import numpy as np
import torch
import torch.nn as nn
import numpy as np
import random
from models import ActGen, StateGen
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from utils import evaluate_actions_sil, select_actions, select_state

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
    def __init__(self, network, args, optimizer, num_actions, obs_state_shape):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.num_actions = num_actions
        self.obs_state_shape = obs_state_shape
        # obs_state_shape is 84x84x84 but we need ...x4x84x84
        self.batch_obs_state_shape = (self.args.num_states*self.args.trace_size, self.obs_state_shape[-1] ) + self.obs_state_shape[:-1]
        #Create Backward models
        self.bw_actgen = ActGen(num_actions)
        self.bw_stategen = StateGen(num_actions)
        if self.args.cuda:
            self.bw_actgen.cuda()
            self.bw_stategen.cuda()
        self.bw_params = list(self.bw_actgen.parameters()) + list(self.bw_stategen.parameters())
        self.bw_optimizer = torch.optim.RMSprop(self.bw_params, lr=self.args.lr, eps=self.args.eps, alpha=self.args.alpha)
        #Create an episode buffer of size : # processes
        self.running_episodes = [[] for _ in range(self.args.num_processes)]
        self.buffer = PrioritizedReplayBuffer(self.args.capacity, self.args.sil_alpha)
        # some other parameters...
        self.total_steps = []
        self.total_rewards = []

    def train_bw_model(self):
        """
        Train the bw_model. Sample (s,a,r,s) from PER Buffer, Compute bw_model loss & Optimize

        """
        obs, actions, returns, obs_next, weights, idxes = self.sample_batch(self.args.k_states)
        batch_size = min(self.args.k_states, len(self.buffer))
        if obs is not None and obs_next is not None:
            # need to get the masks
            # get basic information of network..
            obs = torch.tensor(obs, dtype=torch.float32)
            obs_next = torch.tensor(obs_next, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
            # returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
            weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
            max_nlogp = torch.tensor(np.ones((len(idxes), 1)) * self.args.max_nlogp, dtype=torch.float32)
            if self.args.cuda:
                obs = obs.cuda()
                obs_next = obs_next.cuda()
                actions = actions.cuda()
                # returns = returns.cuda()
                weights = weights.cuda()
                max_nlogp = max_nlogp.cuda()
            pi = self.bw_actgen(obs_next)
            mu = self.bw_stategen(obs_next, self.indexes_to_one_hot(actions))
            # Naive losses without weighting
            # loss_actgen = torch.nn.LLLoss(pi, actions.unsqueeze(1))
            # loss_stategen = nn.MSELoss(obs-obs_next, mu)

            # Losses with weightings and entropy regularization
            action_log_probs, dist_entropy = evaluate_actions_sil(pi, actions)
            action_log_probs = -action_log_probs
            clipped_nlogp = torch.min(action_log_probs, max_nlogp)
            action_loss = torch.sum(weights * clipped_nlogp) / batch_size
            entropy_reg = torch.sum(weights*dist_entropy) / batch_size
            loss_actgen = action_loss - entropy_reg * self.args.entropy_coef
            square_error = ((obs - obs_next - mu)**2).view(batch_size , -1)
            loss_stategen = torch.sum(torch.sum((square_error),1)*weights) / batch_size

            total_loss = loss_actgen + 0.5*loss_stategen
            self.bw_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bw_params, self.args.max_grad_norm)
            self.bw_optimizer.step()

            #Now updating the priorities in the PER Buffer. Use Net Value estimates
            with torch.no_grad():
                value, _ = self.network(obs_next)
                value = torch.clamp(value, min=0)
                self.buffer.update_priorities(idxes, value.squeeze(1).cpu().numpy())
        return

    def train_imitation(self):
        """
        Do these steps
        1. Generate Recall traces from bw_model
        2. Do imiation learning using those recall traces
        """
        # maintain list of sampled episodes(batchwise) and append to list. Then do Imitation learning simply
        _ , _ , _ , states, _ , _ = self.sample_batch(self.args.num_states*5)
        if states is not None:
            with torch.no_grad():
                states = torch.tensor(states, dtype=torch.float32)
                if self.args.cuda:
                    states = states.cuda()
                value, _ = self.network(states)
                sorted_indices = value.cpu().numpy().reshape(-1).argsort()[-self.args.num_states:][::-1]
                # Select high value states under currect valuation
                hv_states = states[sorted_indices.tolist()]
            for n in range(self.args.num_traces):
                # An iteration of sampling recall traces and doing imitation learning
                mb_actions, mb_states_prev = [], []
                states_next = hv_states
                for step in range(self.args.trace_size):
                    with torch.no_grad():
                        pi = self.bw_actgen(states_next)
                        actions = select_actions(pi)
                        mu = self.bw_stategen(states_next, self.indexes_to_one_hot(actions))
                        # s_t = s_t+1 + Δs_t
                        states_prev = states_next + select_state(mu, True)
                    # Add to list
                    mb_actions.append(actions.cpu().numpy())
                    mb_states_prev.append(states_prev.cpu().numpy())
                    # Update state
                    states_next = states_prev
                # Begin to do Imitation Learning
                mb_actions = torch.tensor(mb_actions, dtype=torch.int64).unsqueeze(1).view(self.args.num_states*self.args.trace_size, -1)
                mb_states_prev = torch.tensor(mb_states_prev, dtype=torch.float32).view(self.batch_obs_state_shape)
                max_nlogp = torch.tensor(np.ones((self.args.num_states*self.args.trace_size, 1)) * self.args.max_nlogp, dtype=torch.float32)
                if self.args.cuda:
                    mb_actions = mb_actions.cuda()
                    mb_states_prev = mb_states_prev.cuda()
                    max_nlogp = max_nlogp.cuda()
                _, pi = self.network(mb_states_prev)
                action_log_probs, dist_entropy = evaluate_actions_sil(pi, mb_actions)
                action_log_probs = -action_log_probs
                clipped_nlogp = torch.min(action_log_probs, max_nlogp)
                total_loss = (torch.sum(clipped_nlogp) + torch.sum(dist_entropy)) / self.args.num_states*self.args.trace_size
                # Start to update Policy Network Parameters
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

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
            rewards.append(np.sign(reward))
            dones.append(False)
        # Put done at end of trajectory
        dones[len(dones) - 1] = True
        returns = self.discount_with_dones(rewards, dones, self.args.gamma)
        for (ob, action, R, ob_next) in list(zip(obs, actions, returns, obs_next)):
            self.buffer.add(ob, action, R, ob_next)

    def fn_reward(self, reward):
        return np.sign(reward)

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

    def indexes_to_one_hot(self, indexes):
        """Converts a vector of indexes to a batch of one-hot vectors. """
        indexes = indexes.type(torch.int64).view(-1, 1)
        one_hots = torch.zeros(indexes.shape[0], self.num_actions)
        # one_hots = one_hots.view(*indexes.shape, -1)
        if self.args.cuda:
            one_hots = one_hots.cuda()
        one_hots = one_hots.scatter_(1, indexes, 1)
        return one_hots
