import numpy as np
import ipdb
import torch
from models_mujoco import Net
from datetime import datetime
from utils import select_mj, evaluate_mj, discount_with_dones
import os
from sil_module import sil_module
from bw_module_mujoco import bw_module
import copy

class a2c_agent:
    def __init__(self, envs, args, vis=None):
        self.envs = envs
        self.args = args
        self.vis = vis
        # define the network. Gives V(s) and π(a|S)
        self.net = Net(self.envs.observation_space,self.envs.action_space)
        if self.args.cuda:
            self.net.cuda()
        # define the optimizer
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.args.lr, eps=self.args.eps, alpha=self.args.alpha)
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # check the saved path for envs..
        self.model_path = self.args.save_dir + self.args.env_name + '/'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # get the obs..
        # Initialize observation seen by the environment. Dim : # of processes(16) X observation_space.shape
        self.obs = np.zeros((self.args.num_processes, self.envs.observation_space.shape[0]), dtype=self.envs.observation_space.dtype.name)
        # env already encapsulated the multiple processes
        self.obs[:] = self.envs.reset()
        # track completed processes
        self.dones = [False for _ in range(self.args.num_processes)]

    def learn(self):
        """
        Train the Agent.
        """
        if self.args.model_type == 'sil':
            sil_model = sil_module(self.net, self.args, self.optimizer)
        elif self.args.model_type == 'bw':
            bw_model = bw_module(self.net, self.args, self.optimizer, self.envs.action_space, self.envs.observation_space)
        num_updates = self.args.total_frames // (self.args.num_processes * self.args.nsteps)
        # get the reward to calculate other information
        episode_rewards = torch.zeros([self.args.num_processes, 1])
        final_rewards = torch.zeros([self.args.num_processes, 1])
        # Stuff for Visdom Plots
        vis_loss = []
        vis_timesteps = []
        vis_rewards = []
        win = None
        win2 = None
        # start to update
        for update in range(num_updates):
            mb_obs, mb_rewards, mb_actions, mb_dones = [], [], [], []
            for step in range(self.args.nsteps):
                # Executing the action after seeing the observation
                with torch.no_grad():
                    input_tensor = self._get_tensors(self.obs)
                    _, a_mu, a_sigma = self.net(input_tensor)
                # select actions
                actions = select_mj(a_mu, a_sigma)
                cpu_actions = actions.squeeze(1).cpu().numpy()
                # step in gym batched environment
                obs, rewards, dones, _ = self.envs.step(cpu_actions)
                # start to store the information
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(cpu_actions)
                mb_dones.append(self.dones)
                # process rewards...
                raw_rewards = copy.deepcopy(rewards)
                # rewards = np.sign(rewards)
                # start to store the rewards
                mb_rewards.append(rewards)
                self.dones = dones
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n]*0
                self.obs = obs

                if self.args.model_type == 'sil':
                    # Update the Buffers after doing the step
                    sil_model.step(input_tensor.detach().cpu().numpy(), cpu_actions, raw_rewards, dones)
                elif self.args.model_type == 'bw':
                    # obs_next = self._get_tensors(self.obs).detach().cpu().numpy()
                    # We pass in only numpy objects
                    bw_model.step(input_tensor.detach().cpu().numpy(), cpu_actions, raw_rewards, dones, self.obs)

                raw_rewards = torch.from_numpy(np.expand_dims(np.stack(raw_rewards), 1)).float()
                episode_rewards += raw_rewards
                # get the masks
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
                # update the obs
            mb_dones.append(self.dones)
            # process the rollouts. List to np array
            # 5x16 To 16x5
            mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1,0)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
            mb_actions = np.asarray(mb_actions, dtype=np.float32).swapaxes(1, 0)
            mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
            mb_masks = mb_dones[:, :-1]
            mb_dones = mb_dones[:, 1:]
            with torch.no_grad():
                input_tensor = self._get_tensors(self.obs)
                last_values, _ , _ = self.net(input_tensor)
            # compute returns via 5-step lookahead.
            for n, (rewards, done_, value) in enumerate(zip(mb_rewards, mb_dones, last_values.detach().cpu().numpy().squeeze())):
                rewards = rewards.tolist()
                done_ = done_.tolist()
                if done_[-1] == 0:
                    # Passed in [value] for the estimated V(curr_obs) in TD Learning
                    rewards = discount_with_dones(rewards+[value], done_ + [0], self.args.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, done_, self.args.gamma)
                mb_rewards[n] = rewards
            # Convert 16 x 5 points to 80 flat points
            mb_obs = mb_obs.reshape(self.args.num_processes * self.args.nsteps,-1)
            mb_rewards = mb_rewards.flatten()
            mb_actions = mb_actions.reshape(self.args.num_processes * self.args.nsteps,-1)
            # start to update network. Doing A2C Update
            vl, al, adv = self._update_network(mb_obs, mb_rewards, mb_actions, update)

            # start to update the sil_module or backtracking model
            if self.args.model_type == 'sil':
                mean_adv, num_samples = sil_model.train_sil_model()
            elif (self.args.model_type == 'bw') and (update % self.args.n_a2c == 0):
                if not self.args.consistency:
                    l_bw, l_imi = 0.0, 0.0
                    for _ in range(self.args.n_bw):
                        l_bw += bw_model.train_bw_model(update)
                    l_bw /= self.args.n_bw
                    for _ in range(self.args.n_imi):
                        l_imi += bw_model.train_imitation(update)
                    l_imi /= self.args.n_imi
                else:
                    l_bw, l_fw = 0.0, 0.0
                    for _ in range(self.args.n_bw):
                        l_bw_, l_fw_ = bw_model.train_bw_model(update)
                        l_bw += l_bw_
                        l_fw += l_fw_
                    l_bw /= self.args.n_bw
                    l_fw /= self.args.n_bw
                    l_imi, l_cons = 0.0, 0.0
                    for _ in range(self.args.n_imi):
                        l_imi_, l_cons_ = bw_model.train_imitation(update)
                        l_imi += l_imi_
                        l_cons_ += l_cons_
                    l_imi /= self.args.n_imi
                    l_cons /= self.args.n_imi
            
            if update % self.args.log_interval == 0:
                if self.args.model_type == 'sil':
                    print('[{}] Update: {} of {} Timesteps: {} Rewards: {:.2f} VL: {:.3f} PL: {:.3f} ' \
                            'Ent: {:.2f} Min: {:.4f} Max: {:.4f} BR: {} E: {} VS: {} S: {}'.format(\
                            datetime.now(), update, num_updates, (update+1)*(self.args.num_processes * self.args.nsteps),\
                            final_rewards.mean(), vl, al, adv, final_rewards.min(), final_rewards.max(), sil_model.get_best_reward(), \
                            sil_model.num_episodes(), num_samples, sil_model.num_steps()))
                elif (self.args.model_type == 'bw') and (self.args.consistency) :
                    print('[{}] Update: {} of {} Timesteps: {} Rewards: {:.2f} VL: {:.4f} PL: {:.4f} ' \
                            'Ent: {:.2f} Min: {:.4f} Max: {:.4f} BR: {} E: {} S: {} BW: {:.4f} IMI: {:.4f} FW: {:.4f} CONS: {:.4f}'.format(\
                            datetime.now(), update, num_updates, (update+1)*(self.args.num_processes * self.args.nsteps),\
                            final_rewards.mean(), vl, al, adv, final_rewards.min(), final_rewards.max(), bw_model.get_best_reward(), \
                            bw_model.num_episodes(), bw_model.num_steps(), l_bw, l_imi, l_fw, l_cons))
                elif self.args.model_type == 'bw':
                    print('[{}] Update: {} of {} Timesteps: {} Rewards: {:.2f} VL: {:.4f} PL: {:.4f} ' \
                            'Ent: {:.2f} Min: {:.4f} Max: {:.4f} BR: {} E: {} S: {} BW: {:.4f} IMI: {:.4f}'.format(\
                            datetime.now(), update, num_updates, (update+1)*(self.args.num_processes * self.args.nsteps),\
                            final_rewards.mean(), vl, al, adv, final_rewards.min(), final_rewards.max(), bw_model.get_best_reward(), \
                            bw_model.num_episodes(), bw_model.num_steps(), l_bw, l_imi))
                else:
                    print('[{}] Update: {} of {} Timesteps: {} Rewards: {:.2f} VL: {:.3f} PL: {:.3f} ' \
                            'Ent:{:.3f} Min: {:.4f} Max: {:.4f}'.format(\
                            datetime.now(), update, num_updates, (update+1)*(self.args.num_processes * self.args.nsteps),\
                            final_rewards.mean(), vl, al, adv, final_rewards.min(), final_rewards.max()))
                torch.save(self.net.state_dict(), self.model_path + 'model.pt')
            # Save to Visdom Plots
            if self.args.vis and (update % self.args.vis_interval == 0):
                if (self.args.model_type == 'bw') and (self.args.consistency):
                    vis_loss.append([vl, al, l_bw, l_imi, l_fw, l_cons])
                    legend=['Value loss','policy loss', 'BW Loss','IMI loss', 'CONST loss']
                elif self.args.model_type == 'bw':
                    vis_loss.append([vl, al, l_bw, l_imi])
                    legend=['Value loss','policy loss', 'BW Loss','IMI loss']
                else:
                    vis_loss.append([vl, al])
                    legend=['Value loss','policy loss']
                vis_timesteps.append((update+1)*(self.args.num_processes * self.args.nsteps))
                vis_rewards.append(final_rewards.mean())
                title = self.args.env_name + '--' + self.args.model_type
                if win is None:
                    win = self.vis.line(Y=np.array(vis_rewards), X=np.array(vis_timesteps), opts=dict(title=title, xlabel='Timesteps',
                                ylabel='Avg Rewards'))
                self.vis.line(Y=np.array(vis_rewards), X=np.array(vis_timesteps), win=win, update='replace', opts=dict(title=title, xlabel='Timesteps',
                                ylabel='Avg Rewards'))
                if win2 is None:
                    win2 = self.vis.line(Y=np.array(vis_loss), X=np.array(vis_timesteps), opts=dict(title=title, xlabel='Timesteps',
                                ylabel='Losses', legend=legend))
                self.vis.line(Y=np.array(vis_loss), X=np.array(vis_timesteps), win=win2, update='replace', opts=dict(title=title, xlabel='Timesteps',
                                ylabel='Losses', legend=legend))
                
    def _update_network(self, obs, returns, actions, update):
        """
        Learning the Policy Network using A2C.
        """
        # evaluate the actions
        input_tensor = self._get_tensors(obs)
        values, mu, sigma = self.net(input_tensor)
        # define the tensor of actions, returns
        # convert to 2D tensor of (#processesxn_steps)x1
        returns = torch.tensor(returns, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        if self.args.cuda:
            returns = returns.cuda()
            actions = actions.cuda()
        # evaluate actions
        action_log_probs, dist_entropy = evaluate_mj(self.args, mu, sigma, actions, False)
        # calculate advantages...
        advantages = returns - values
        # get the value loss
        value_loss = advantages.pow(2).mean()
        # get the action loss. We detach advantages to reduce to standard PG form upon diff
        action_loss = -(advantages.detach() * action_log_probs).mean()
        # total loss
        total_loss = action_loss + self.args.value_loss_coef * value_loss - dist_entropy.mean() * self.args.entropy_coef
        # start to update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        return value_loss.item(), action_loss.item(), dist_entropy.mean()

    def _get_tensors(self, obs):
        """
        Get the input tensors...
        """
        input_tensor = torch.tensor(obs, dtype=torch.float32)
        if self.args.cuda:
            input_tensor = input_tensor.cuda()
        return input_tensor
