import os
import torch
import copy
import time
from statistics import mean

from gym.wrappers import RecordVideo

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import learning.amp_players as amp_players
from learning.value_pose_net import ValuePoseNet

class AMPPlayerContinuousValue(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)
        self.plot_val_reward = self.player_config['plot_val_reward']
        self.use_pose = self.player_config['use_pose']
        self.use_vel = self.player_config['use_vel']
        self.use_vru = self.player_config['use_vru']
        self.step_to_pred = self.env.task.step_to_pred
        self.inversion_penalty_scale = self.config["inversion_penalty_scale"]
        self.visualized_pose = False
        self.gamma = self.config['gamma']
        # import pdb; pdb.set_trace()

        self.valuenet = self._build_valuenet(self.config)
        return

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        total_value_loss = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None
        vals = []
        rewards = []
        rewards_loc = []
        rewards_pow = []
        rewards_disc = []
        max_reward = 100
        min_reward = -10
        max_frame_rew = -100
        min_frame_rew = 100

        # import pdb; pdb.set_trace()

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        print('n_games', n_games)
        print('rendering:', render)
        print('exp_name:', self.video_dir)
        for t in range(n_games):
            rew_lists = {'total': [], 'loc': [], 'disc': [], 'pow': []}
            rew_disc_coef = 1.0 # the discount coefficient for the reward
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()
            inverted_envs = self.env.task.inverted.item()
            if inverted_envs:
                print('inverted')
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            c_task_value = 0
            c_critic_value = 0
            c_disc_reward = 0
            c_loc_reward = 0
            c_pow_reward = 0

            print_game_res = False

            done_indices = []
            waypoint_traj = None
            init_pose = None

            ideal_traj = self.env.task._traj_gen._verts[0, :, :2].cpu().numpy()
            real_traj = []

            # from collections import deque
            # dt_acc = deque(maxlen=100)

            with torch.no_grad():
                # show progress bar
                bar = tqdm.tqdm(total=self.env.task.max_episode_length, desc='steps', leave=False, dynamic_ncols=True)
                for n in range(self.max_steps):
                    t_s = time.time()

                    obs_dict = self.env_reset(done_indices)
                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)
                    obs_dict, r, done, info =  self.env_step(self.env, action)

                    # import pdb; pdb.set_trace()
                    real_traj.append(copy.deepcopy(self.env.task._humanoid_root_states[0,:2].cpu().numpy()))
                    r *= (-self.inversion_penalty_scale) if inverted_envs else 1.0
                    if n == 0:
                        with torch.no_grad():
                            waypoint_traj = self.env.get_waypoint_traj()[:,:13,:].to(self.device)
                            init_pose = self.env.get_init_pose().to(self.device)
                            init_vel = self.env.get_init_vel().to(self.device)
                            # import pdb; pdb.set_trace()
                            valuenet_pred = self.valuenet(waypoint_traj, init_pose, init_vel)
                            if not self.visualized_pose:
                                self.valuenet.visualize_pose(init_pose.cpu(), gt_xy=waypoint_traj.cpu())
                                # self.visualized=True
                    r_raw = self.env.raw_reward()
                    r_loc, r_pow = r_raw[:,0], r_raw[:,1]
                    # print(f'r_loc: {r_loc.mean():.2f}, r_pow: {r_pow.mean():.2f}')
                    # import pdb; pdb.set_trace()

                    post_infos = self._post_step(info)
                    rew_disc_coef *= self.gamma # discount coefficient
                    if post_infos:
                        task_value, critic_value, disc_reward = post_infos
                        c_task_value += task_value
                        c_critic_value += critic_value
                        c_disc_reward += (disc_reward * 0.25) * rew_disc_coef # add discounted reward
                        c_loc_reward += (r_loc * 0.5) * rew_disc_coef # add discounted reward
                        c_pow_reward += (r_pow * 0.5) * rew_disc_coef # add discounted reward
                        cr += ((r_loc + r_pow) * 0.5 + disc_reward * 0.25) * rew_disc_coef # add discounted reward
                        # add discription
                        bar.set_description(f'game: {t}, steps: {n},  traj_reward: {c_loc_reward.mean():.2f}, disc_reward: {c_disc_reward:.2f}, power_reward: {c_pow_reward.mean():.2f}, combined_reward: {cr.mean():.2f}')
                        frame_rew = ((r_loc + r_pow) * 0.5 + disc_reward * 0.25) * rew_disc_coef # add discounted reward
                        max_frame_rew = max(max_frame_rew, frame_rew.max().item())
                        min_frame_rew = min(min_frame_rew, frame_rew.min().item())
                        rew_lists['total'].append(frame_rew.item())
                        rew_lists['loc'].append(r_loc.item()*0.5*rew_disc_coef)
                        rew_lists['disc'].append(disc_reward*0.25*rew_disc_coef)
                        rew_lists['pow'].append(r_pow.item()*0.5*rew_disc_coef)
                    else:
                        bar.set_description(f'game: {t}, steps: {n}, reward: {cr.mean():.2f}')
                        cr += r * rew_disc_coef # add discounted reward

                    steps += 1
                    bar.update(1)

                    # dt = time.time() - t_s
                    # dt_acc.append(dt)
                    # print(1/np.mean(dt_acc))

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[::self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count
                    if (n == self.step_to_pred):
                            # ratio = self.step_to_pred/self.env.task.max_episode_length
                            cr_to_pred = copy.deepcopy(cr)
                            if self.plot_val_reward:
                                # import pdb; pdb.set_trace()
                                rewards_loc.append(c_loc_reward.item())
                                rewards_pow.append(c_pow_reward.item())
                                rewards_disc.append(c_disc_reward)

                    if done_count > 0:
                        if n < self.step_to_pred:
                            cr_to_pred = copy.deepcopy(cr)
                            if self.plot_val_reward:
                                # import pdb; pdb.set_trace()
                                rewards_loc.append(c_loc_reward.item())
                                rewards_pow.append(c_pow_reward.item())
                                rewards_disc.append(c_disc_reward)
                        # import pdb; pdb.set_trace()
                        norm_rewards = (cr_to_pred - min_reward) / (max_reward - min_reward)
                        value_loss = self.criterion(valuenet_pred.squeeze(), norm_rewards.squeeze())
                        print(f'pred_value: {valuenet_pred.mean().item():.2f}, cr_to_pred: {norm_rewards.mean().item():.2f}, value_loss: {value_loss.item():.2f}')
                        total_value_loss += value_loss.item()
                        if self.is_rnn:
                            for s in self.states:
                                s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        vals.append(valuenet_pred.mean().item())
                        # rewards.append(norm_rewards.mean().item())
                        rewards.append(cr_to_pred.mean().item())
                        # if cr_to_pred.mean().item() > max_reward:
                            # max_reward = cr_to_pred.mean().item()
                        # if cr_to_pred.mean().item() < min_reward:
                            # min_reward = cr_to_pred.mean().item()

                        # cr = cr * (1.0 - done.float())
                        cr = 0
                        # steps = steps * (1.0 - done.float())
                        steps = 0
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        # game_res = 0.0
                        # if isinstance(info, dict):
                        #     if 'battle_won' in info:
                        #         print_game_res = True
                        #         game_res = info.get('battle_won', 0.5)
                        #     if 'scores' in info:
                        #         print_game_res = True
                        #         game_res = info.get('scores', 0.5)
                        # if self.print_stats:
                        #     if print_game_res:
                        #         print(f'reward: {cur_rewards/done_count:.2f}, steps: {cur_steps/done_count}, w: {game_res}')
                        #     else:
                        #         print(f'reward: {cur_rewards/done_count:.2f}, steps: {cur_steps/done_count}')

                        # sum_game_res += game_res

                        # save video
                        if render:
                            self.env_save_video(game=t, exp_name=self.video_dir, rew=rew_lists, real_traj=real_traj, ideal_traj=ideal_traj)

                        if games_played % 10 == 0 and self.plot_val_reward:
                            self._plot_val_reward(vals, rewards, rew_type='total')
                            self._plot_val_reward(vals, rewards_loc, rew_type='loc')
                            self._plot_val_reward(vals, rewards_pow, rew_type='pow')
                            self._plot_val_reward(vals, rewards_disc, rew_type='disc')

                        if batch_size//self.num_agents == 1 or games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print('sum_rewards', sum_rewards, 'max_reward', max_reward, 'min_reward', min_reward)
        if print_game_res:
            print(f'av reward: {(sum_rewards / games_played * n_game_life):.3f}, av steps: {(sum_steps / games_played * n_game_life):.3f}, winrate: {(sum_game_res / games_played * n_game_life):.3f}')
        else:
            print(f'av reward: {(sum_rewards / games_played * n_game_life):.3f}, av steps: {(sum_steps / games_played * n_game_life):.3f}, av value loss: {total_value_loss / games_played:.3f}')
            # print(f'av_loc: {rewards_loc.mean():.2f}, av_pow: {rewards_pow.mean():.2f}, av_disc: {rewards_disc.mean():.2f}, av_total: {rewards.mean():.2f}')
            print(f'av_loc: {mean(rewards_loc):.2f}, av_pow: {mean(rewards_pow):.2f}, av_disc: {mean(rewards_disc):.2f}, av_total: {mean(rewards):.2f}')
            # standard deviation
            print(f'std_loc: {np.std(rewards_loc):.2f}, std_pow: {np.std(rewards_pow):.2f}, std_disc: {np.std(rewards_disc):.2f}, std_total: {np.std(rewards):.2f}')

        if self.plot_val_reward:
            self._plot_val_reward(vals, rewards, rew_type='total')
            self._plot_val_reward(vals, rewards_loc, rew_type='loc')
            self._plot_val_reward(vals, rewards_pow, rew_type='pow')
            self._plot_val_reward(vals, rewards_disc, rew_type='disc')
            print('Val-Reward plot saved')
            print(f'Correlation: \n Total reward: {np.corrcoef(vals, rewards)[0, 1]:.3f}')
            print(f'Loc reward: {np.corrcoef(vals, rewards_loc)[0, 1]:.3f}')
            print(f'Pow reward: {np.corrcoef(vals, rewards_pow)[0, 1]:.3f}')
            print(f'Disc reward: {np.corrcoef(vals, rewards_disc)[0, 1]:.3f}')

        self.save_hist(vals, rewards, save_name='val_reward_hist.png')
        print('Hist plot saved to', 'output/plot/val_reward_hist.png')
        return

    def _post_step(self, info):
        super()._post_step(info)
        # import pdb; pdb.set_trace()
        if self.plot_val_reward:
            # self._amp_debug(info)
            return self._task_value_debug(info)
        else:
            return None

    def _task_value_debug(self, info):
        obs = info['obs']
        amp_obs = info['amp_obs']
        # import pdb; pdb.set_trace()
        obs = obs.cuda()
        task_value = self._eval_task_value(obs)
        amp_obs_single = amp_obs[0:1].cuda()

        critic_value = self._eval_critic(obs)
        disc_pred = self._eval_disc(amp_obs_single)
        amp_rewards = self._calc_amp_rewards(amp_obs_single)
        disc_reward = amp_rewards['disc_rewards']
        # plot_all = torch.cat([critic_value, task_value])
        # plotter_names = ("task_value", "task")
        # self.live_plotter(plot_all.cpu().numpy(), plotter_names = plotter_names)
        return (task_value.mean().item(), critic_value.mean().item(), disc_reward.mean().item())

    def _eval_task_value(self, input):
        input = self._preproc_input(input)
        return self.model.a2c_network.eval_task_value(input)

    def _plot_val_reward(self, vals, rewards, rew_type='total'):
        # added by takez.
        # plot the value and reward, and compute the correlation between them
        assert rew_type in ['total', 'loc', 'pow', 'disc']
        corr = np.corrcoef(vals, rewards)
        plt.scatter(vals, rewards)
        plt.xlabel('task value', fontsize=18)
        plt.ylabel(f'{rew_type} reward', fontsize=18)
        plt.title(f'correlation: {corr[0, 1]:.3f}', fontsize=18)
        # save the plot
        save_dir = 'output/plot/'
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        plt.savefig(save_dir + f'val_{rew_type}_reward.png')
        plt.close()

    def live_plotter(self, w, plotter_names,  identifier='', pause_time=0.00000001):
        matplotlib.use("Qt5agg")
        num_lines = len(w)
        if not hasattr(self, 'lines'):
            size = 100
            self.x_vec = np.linspace(0, 1, size + 1)[0:-1]
            self.y_vecs = [np.array([0] * len(self.x_vec)) for i in range(7)]
            self.lines = [[] for i in range(num_lines)]
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()

            self.fig = plt.figure(figsize=(1, 1))
            ax = self.fig.add_subplot(111)
            # create a variable for the line so we can later update it

            for i in range(num_lines):
                l, = ax.plot(self.x_vec, self.y_vecs[i], '-o', alpha=0.8)
                self.lines[i] = l

            # update plot label/title
            plt.ylabel('Values')

            plt.title('{}'.format(identifier))
            plt.ylim((-0.75, 1.5))
            plt.gca().legend(plotter_names)
            plt.show()

        for i in range(num_lines):
            # after the figure, axis, and line are created, we only need to update the y-data
            self.y_vecs[i][-1] = w[i]
            self.lines[i].set_ydata(self.y_vecs[i])
            # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
            self.y_vecs[i] = np.append(self.y_vecs[i][1:], 0.0)

        # plt.pause(pause_time)
        self.fig.canvas.start_event_loop(0.001)

    def save_plotter(self):
        plt.savefig('output/plot/val_reward.png')
        plt.close()
        return

    def _build_valuenet(self, config): # added by takez
        # import pdb; pdb.set_trace()
        self.valuenet = ValuePoseNet(use_pose=self.use_pose, hide_toe=True, normalize=True, use_vel=self.use_vel, vru=self.use_vru)
        self.valuenet.load_state_dict(torch.load(self.player_config['valuenet_path']))
        self.valuenet.to(self.device)
        self.valuenet.eval()
        self.valuenet.requires_grad_ = False
        self.criterion = torch.nn.MSELoss()
        return self.valuenet

    def save_hist(self, vals, rewards, save_dir='output/plot/', save_name='val_reward_hist.png'):
        # binning
        plt.figure()
        # plt.hist(vals, bins=50, alpha=0.5, label='task value')
        plt.hist(rewards, bins=30, label='reward', density=True, range=(-300, 300))
        plt.legend()
        plt.savefig(save_dir + save_name)
        return