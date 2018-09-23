import numpy as np
import ipdb
import torch
import torch.nn as nn
import numpy as np
import random
from models_mujoco import ActGen, StateGen
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from utils import evaluate_actions_sil, select_mj, evaluate_mj, zero_mean_unit_std

class ReplayBuffer:
    """
    Replay buffer...
    """
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, R, obs_next):
        data = (obs_t, action, R, obs_next)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def _encode_sample(self, idxes):
        obses_t, actions, returns, obses_next = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, R, obs_next = data
            obses_t.append(obs_t)
            actions.append(action)
            returns.append(R)
            obses_next.append(obs_next)
        return np.array(obses_t), np.array(actions), np.array(returns), np.array(obses_next)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 1e-6)
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size, beta):
        idxes = self._sample_proportional(batch_size)
        if beta > 0:
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self._storage)) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            weights = np.ones_like(idxes, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

class bw_module:
    def __init__(self, network, args, optimizer, action_shape, obs_shape):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.action_shape = action_shape
        self.obs_shape = obs_shape
        #Create Backward models
        self.bw_actgen = ActGen(self.obs_shape, self.action_shape)
        self.bw_stategen = StateGen(self.obs_shape, self.action_shape)
        if self.args.cuda:
            self.bw_actgen.cuda()
            self.bw_stategen.cuda()
        self.bw_params = list(self.bw_actgen.parameters()) + list(self.bw_stategen.parameters())
        self.bw_optimizer = torch.optim.RMSprop(self.bw_params, lr=self.args.lr, eps=self.args.eps, alpha=self.args.alpha)
        # self.bw_optimizer = torch.optim.Adam(self.bw_params, lr=self.args.lr)
        
        #Create a forward model
        if self.args.consistency:
            self.fw_stategen = StateGen(self.obs_shape, self.action_shape)
            if self.args.cuda:
                self.fw_stategen.cuda()
            self.fw_optimizer = torch.optim.RMSprop(self.fw_stategen.parameters(), lr=self.args.lr, eps=self.args.eps, alpha=self.args.alpha)
            # self.fw_optimizer = torch.optim.Adam(self.fw_stategen.parameters(), lr=self.args.lr)
        #Create an episode buffer of size : # processes
        self.running_episodes = [[] for _ in range(self.args.num_processes)]
        self.buffer = PrioritizedReplayBuffer(self.args.capacity, self.args.sil_alpha)
        # some other parameters...
        self.total_steps = []
        self.total_rewards = []
        # Set the mean, stds. All numpy stuff
        self.obs_delta_mean = None
        self.obs_delta_std = None
        self.obs_next_mean = None
        self.obs_next_std = None
        self.actions_mean = None
        self.actions_std = None

    def train_bw_model(self, update):
        """
        Train the bw_model. Sample (s,a,r,s) from PER Buffer, Compute bw_model loss & Optimize
        """
        obs, actions, _, obs_next_unnormalized, weights, idxes = self.sample_batch(self.args.k_states)
        batch_size = min(self.args.k_states, len(self.buffer))
        if obs is not None and obs_next_unnormalized is not None:
            obs_delta, self.obs_delta_mean, self.obs_delta_std = zero_mean_unit_std(obs-obs_next_unnormalized)
            actions, self.actions_mean, self.actions_std = zero_mean_unit_std(actions)
            obs_next, self.obs_next_mean, self.obs_next_std = zero_mean_unit_std(obs_next_unnormalized)
            # need to get the masks
            # get basic information of network..
            obs_delta = torch.tensor(obs_delta, dtype=torch.float32)
            obs_next = torch.tensor(obs_next, dtype=torch.float32)
            obs_next_unnormalized = torch.tensor(obs_next_unnormalized, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            if self.args.per_weight:
                weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
            if self.args.cuda:
                obs_delta = obs_delta.cuda()
                obs_next = obs_next.cuda()
                obs_next_unnormalized = obs_next_unnormalized.cuda()
                actions = actions.cuda()
                if self.args.per_weight:
                    weights = weights.cuda()
            # Train BW - Model
            a_mu = self.bw_actgen(obs_next)
            s_mu, s_sigma = self.bw_stategen(obs_next, actions)
            s_sigma = torch.ones_like(s_mu)
            a_sigma = torch.ones_like(a_mu)
            if self.args.cuda:
                a_sigma = a_sigma.cuda()
            # Calculate Losses. Losses in terms of everything (mu,sigma,actions/obs_delta) noramlized only.
            action_log_probs, action_entropy = evaluate_mj(self.args, a_mu, a_sigma, actions)
            state_log_probs, state_entropy = evaluate_mj(self.args, s_mu, s_sigma, obs_delta)
            if self.args.per_weight:
                entropy_loss = self.args.entropy_coef*(torch.mean(action_entropy*weights)+torch.mean(state_entropy*weights))
                total_loss = -torch.mean(action_log_probs*weights) - torch.mean(state_log_probs*weights) - entropy_loss
            else:
                entropy_loss = self.args.entropy_coef*(action_entropy.mean()+state_entropy.mean())
                total_loss = -torch.mean(action_log_probs) - torch.mean(state_log_probs) - entropy_loss
            self.bw_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bw_params, self.args.max_grad_norm)
            self.bw_optimizer.step()

            #Now updating the priorities in the PER Buffer. Use Net Value estimates
            with torch.no_grad():
                value, _, _ = self.network(obs_next_unnormalized)
            value = torch.clamp(value, min=0)
            self.buffer.update_priorities(idxes, value.squeeze(1).cpu().numpy())

            # Train FW - Model
            if self.args.consistency:
                f_mu, f_sigma = self.fw_stategen(obs, actions)
                log_probs, dist_entropy = evaluate_mj(self.args, f_mu, f_sigma, obs_next)
                if self.args.per_weight:
                    fw_loss = -torch.mean(log_probs*weights) - self.args.entropy_coef*torch.mean(dist_entropy*weights)
                else:
                    fw_loss = -torch.mean(log_probs) - self.args.entropy_coef*(dist_entropy.mean())
                self.fw_optimizer.zero_grad()
                fw_loss.backward()
                torch.nn.under.clip_grad_norm_(self.fw_stategen.parameters(), self.args.max_grad_norm)
                self.fw_optimizer.step()
                return total_loss.item(), fw_loss.item()
            else:
                return total_loss.item()

        elif self.args.consistency:
            return 0.0, 0.0
        else:
            return 0.0

    def train_imitation(self, update):
        """
        Do these steps
        1. Generate Recall traces from bw_model
        2. Do imitation learning using those recall traces
        """
        # maintain list of sampled episodes(batchwise) and append to list. Then do Imitation learning simply
        _ , _ , _ , states, _ , _ = self.sample_batch(self.args.num_states*5)
        if states is not None:
            states_preprocessed = np.nan_to_num((states-self.obs_next_mean)/self.obs_next_std)
            with torch.no_grad():
                # self.network requires un-normalized states
                states = torch.tensor(states, dtype=torch.float32)
                states_preprocessed = torch.tensor(states_preprocessed, dtype=torch.float32)
                if self.args.cuda:
                    states = states.cuda()
                    states_preprocessed = states_preprocessed.cuda()
                value, _, _ = self.network(states)
                sorted_indices = value.cpu().numpy().reshape(-1).argsort()[-self.args.num_states:][::-1]
            # Select high value states under currect valuation for target states_next
            states_next = states[sorted_indices.tolist()]
            states_next_preprocessed = states_preprocessed[sorted_indices.tolist()]
            mb_actions, mb_states_prev = [], []
            for step in range(self.args.trace_size):
                with torch.no_grad():
                    a_mu = self.bw_actgen(states_next_preprocessed)
                    a_sigma = torch.ones_like(a_mu)
                    if self.args.cuda:
                        a_sigma = a_sigma.cuda()
                    actions = select_mj(a_mu, a_sigma)
                    # if self.args.cuda:
                    #     actions = actions*torch.tensor(self.actions_std, dtype=torch.float32).cuda() + torch.tensor(self.actions_mean, dtype=torch.float32).cuda()
                    # else:
                    #     actions = actions*torch.tensor(self.actions_std, dtype=torch.float32) + torch.tensor(self.actions_mean, dtype=torch.float32)
                    s_mu, s_sigma = self.bw_stategen(states_next_preprocessed, actions)
                    s_sigma = torch.ones_like(s_mu)
                    # s_t = s_t+1 + Δs_t
                    delta_state = select_mj(s_mu, s_sigma)
                    if self.args.cuda:
                        delta_state = delta_state*torch.tensor(self.obs_delta_std, dtype=torch.float32).cuda() + torch.tensor(self.obs_delta_mean, dtype=torch.float32).cuda()
                    else:
                        delta_state = delta_state*torch.tensor(self.obs_delta_std, dtype=torch.float32) + torch.tensor(self.obs_delta_mean, dtype=torch.float32)
                    states_prev = states_next + delta_state
                    states_next = states_prev
                    #np.nan_to_num not available in torch
                    states_next_preprocessed = np.nan_to_num((states_next.cpu().numpy() - self.obs_delta_mean)/self.obs_next_std)
                    states_next_preprocessed = torch.tensor(states_next_preprocessed, dtype=torch.float32)
                    if self.args.cuda:
                        states_next_preprocessed = states_next_preprocessed.cuda()
                # Add to list
                mb_actions.append(actions.cpu().numpy())
                mb_states_prev.append(states_prev.cpu().numpy())
            # Begin to do Imitation Learning
            mb_actions = torch.tensor(mb_actions, dtype=torch.float32).view(self.args.num_states*self.args.trace_size, -1)
            mb_states_prev = torch.tensor(mb_states_prev, dtype=torch.float32).view(self.args.num_states*self.args.trace_size, -1)
            if self.args.cuda:
                mb_actions = mb_actions.cuda()
                mb_states_prev = mb_states_prev.cuda()
            _, mu, sigma = self.network(mb_states_prev)
            try:
                action_log_probs, dist_entropy = evaluate_mj(self.args, mu, sigma, mb_actions)
            except:
                print(mu, sigma)
                import ipdb;ipdb.set_trace()
            total_loss = -torch.mean(action_log_probs) - - self.args.entropy_coef*(dist_entropy.mean())
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # Do the Consistency Bit
            if self.args.consistency:
                print('foo')
            else:
                return total_loss.item()
        elif self.args.consistency:
            return 0.0, 0.0
        else:
            return 0.0

    def step(self, obs, actions, rewards, dones, obs_next):
        """
        Add the batch information into the Buffers
        """
        for n in range(self.args.num_processes):
            self.running_episodes[n].append([obs[n], actions[n], rewards[n], obs_next[n]])
        # to see if can update the episode...
        for n, done in enumerate(dones):
            if done:
                self.update_buffer(self.running_episodes[n])
                # Clear the episode buffer
                self.running_episodes[n] = []

    def update_buffer(self, trajectory):
        """
        Update buffer. Add single episode to PER Buffer and update stuff
        """
        positive_reward = False
        for (ob, a, r, ob_next) in trajectory:
            if r > 0:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)
            self.total_steps.append(len(trajectory))
            self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            while np.sum(self.total_steps) > self.args.capacity and len(self.total_steps) > 1:
                self.total_steps.pop(0)
                self.total_rewards.pop(0)

    def add_episode(self, trajectory):
        """
        Add single episode to PER Buffer
        """
        obs = []
        actions = []
        rewards = []
        dones = []
        obs_next = []
        for (ob, action, reward, ob_next) in trajectory:
            if ob is not None:
                obs.append(ob)
            else:
                obs.append(None)
            if ob_next is not None:
                obs_next.append(ob_next)
            else:
                obs_next.append(None)
            actions.append(action)
            # rewards.append(np.sign(reward))
            rewards.append(reward)
            dones.append(False)
        # Put done at end of trajectory
        dones[len(dones) - 1] = True
        returns = self.discount_with_dones(rewards, dones, self.args.gamma)
        for (ob, action, R, ob_next) in list(zip(obs, actions, returns, obs_next)):
            self.buffer.add(ob, action, R, ob_next)

    def get_best_reward(self):
        if len(self.total_rewards) > 0:
            return np.max(self.total_rewards)
        return 0

    def num_episodes(self):
        return len(self.total_rewards)

    def num_steps(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if len(self.buffer) > 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size, beta=self.args.sil_beta)
        else:
            return None, None, None, None, None, None

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]
