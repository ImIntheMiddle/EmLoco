import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml
import glob
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from pacer.utils.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
from torch import optim

import learning.amp_datasets as amp_datasets
from learning.value_pose_net import ValuePoseNet
from learning.scheduler import CosineAnnealingLR

from tensorboardX import SummaryWriter
import wandb


class CommonAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, config):
        a2c_common.A2CBase.__init__(self, base_name, config)
        self.cfg = config
        self.exp_name = self.cfg['train_dir'].split('/')[-1]

        self._load_config_params(config)

        self.is_discrete = False
        self._setup_action_space()
        self._actor_coef = config.get('actor_coef', None)
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        self.clip_actions = config.get('clip_actions', True)
        self._save_intermediate = config.get('save_intermediate', False)
        self._do_finetune = config['player'].get('finetune', False)
        self.use_pose = config['player'].get('use_pose', False)
        self.use_vel = config['player'].get('use_vel', False)
        self.use_vru = config['player'].get('use_vru', False)
        self.step_to_pred = self.vec_env.env.task.step_to_pred
        self.inversion_penalty_scale = config.get('inversion_penalty_scale', None)

        net_config = self._build_net_config()
        if self.normalize_input:
            obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)
        net_config['mean_std'] = self.running_mean_std
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        # import pdb; pdb.set_trace()
        use_except = ['_task_value_mlp', '_value_logits']
        if self._do_finetune: # for finetuning of value network
            for name, param in self.model.named_parameters():
                if not any(ex in name for ex in use_except):
                    param.requires_grad = False
        else: # for normal training
        # use except _task_value_mlp and _value_logits
            for name, param in self.model.named_parameters():
                if any(ex in name for ex in use_except):
                    param.requires_grad = False
        # print('params', use_params)
        self.last_lr = 0 if self._do_finetune else float(self.last_lr)
        self.policy_lr = float(self.last_lr)
        self.value_lr = float(self.last_lr) if self._do_finetune else 0

        self.optimizer = optim.Adam(self.model.parameters(),
                                    self.policy_lr,
                                    eps=1e-08,
                                    weight_decay=self.weight_decay)

        if self._do_finetune:
            print('!!!! Finetuning value network !!!!')
            self.valuenet = self._build_valuenet()
            # self.valuenet.load_state_dict(torch.load("/home/halo/plausibl/pacer/output/exp/pacer/v4_valuenet_lr1e-3mini_valuenet.pth"))
            self.valuenet = self.valuenet.to(self.ppo_device)
            self.vnet_optimizer = torch.optim.AdamW(self.valuenet.parameters(), lr=1e-3, weight_decay=0.0001)
            self.vnet_scheduler = CosineAnnealingLR(self.vnet_optimizer, warmup_epochs=20, max_epochs=20000)
            self.vnet_criterion = torch.nn.MSELoss(reduction='sum') # sum because we are averaging over minibatch
            self.vnet_optimizer.zero_grad()
            # self.step_to_pred = con

            # also turn off policy training
            self._actor_coef = 0
            self.critic_coef = 0
            self.entropy_coef = 0
            self.bouns_loss_coef = 0
            self._tv_coef = 0
            self._disc_coef = 0
            self.sym_loss_coef = 0
            self.mini_epochs_num = 1 # only one epoch for finetuning because we are not training policy

        if self.has_central_value:
            cv_config = {
                'state_shape': torch_ext.shape_whc_to_cwh(self.state_shape),
                'value_size': self.value_size,
                'ppo_device': self.ppo_device,
                'num_agents': self.num_agents,
                'horizon_length': self.horizon_length,
                'num_actors': self.num_actors,
                'num_actions': self.actions_num,
                'seq_len': self.seq_len,
                'model': self.central_value_config['network'],
                'config': self.central_value_config,
                'writter': self.writer,
                'multi_gpu': self.multi_gpu
            }
            self.central_value_net = central_value.CentralValueTrain(
                **cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = amp_datasets.AMPDataset(self.batch_size,
                                               self.minibatch_size,
                                               self.is_discrete, self.is_rnn,
                                               self.ppo_device, self.seq_len)
        self.algo_observer.after_init(self)

        return

    def init_tensors(self):
        super().init_tensors()
        self.current_combined_rewards = torch.zeros_like(self.current_rewards).flatten().to(self.ppo_device)
        self.game_combined_rewards = torch.zeros_like(self.current_rewards).flatten().to(self.ppo_device)
        self.discount_coefs = torch.ones_like(self.current_rewards).flatten().to(self.ppo_device) # initialize gammas to 1
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(
            self.experience_buffer.tensor_dict['values'])
        self.vnet_pred = torch.empty(0).to(self.ppo_device)
        self.vnet_gt = torch.empty(0).to(self.ppo_device)
        self.vnet_loss = 0
        self.tensor_list += ['next_obses']
        return

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        self.max_cum_rewards = 100 # fix to 100 or 0
        self.min_cum_rewards = -10 # fix to -20 or 1000
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        model_output_file = osp.join(self.network_path, self.config['name'])

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        self._init_train()

        while True:
            epoch_start = time.time()

            epoch_num = self.update_epoch()
            train_info = self.train_epoch()

            sum_time = train_info['total_time']
            total_time += sum_time
            frame = self.frame
            if self.multi_gpu:
                self.hvd.sync_stats(self)

            if self.rank == 0:
                scaled_time = sum_time
                scaled_play_time = train_info['play_time']
                curr_frames = self.curr_frames
                self.frame += curr_frames
                fps_step = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time
                vnet_loss = 0
                mean_pred = 0
                mean_combined_rewards = 0

                self.writer.add_scalar('performance/total_fps',curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)
                train_info_dict = self._log_train_info(train_info, frame)
                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self._get_mean_rewards()
                    mean_lengths = self.game_lengths.get_mean()

                    # update value network
                    if self._do_finetune:
                        vnet_loss = self.vnet_loss/len(self.vnet_pred) if len(self.vnet_pred) > 0 else 0
                        mean_pred = self.vnet_pred.mean().item() if len(self.vnet_pred) > 0 else 0
                        mean_combined_rewards = self.vnet_gt.mean().item() if len(self.vnet_gt) > 0 else 0

                        self.vnet_scheduler.step()

                    for i in range(self.value_size):
                        self.writer.add_scalar('rewards{0}/frame'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar('rewards{0}/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar('rewards{0}/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)

                    if not wandb.run is None:
                        train_info_dict.update({
                            "episode_lengths": mean_lengths,
                            "mean_rewards": np.mean(mean_rewards),
                        })
                        if len(self.vnet_pred) > 0:
                            train_info_dict.update({
                                "vnet_loss": vnet_loss,
                                "vnet_pred": mean_pred,
                                "combined_rewards": mean_combined_rewards,
                                "max_cum_rewards": self.max_cum_rewards,
                                "min_cum_rewards": self.min_cum_rewards,
                            })
                        wandb.log(train_info_dict, step=epoch_num)

                    epoch_end = time.time()
                    if self._do_finetune:
                        log_str = f"Ep: {self.epoch_num}\trwd: {np.mean(mean_rewards):.2f}\tvnet_pred: {mean_pred:.2f}\tcombine_rwd: {mean_combined_rewards:.2f}\tvnet_loss: {vnet_loss:.3f}\tfps_step: {fps_step:.1f}\tfps_total: {fps_total:.1f}\tep_time:{epoch_end - epoch_start:.1f}\tframe: {self.frame}\teps_len: {mean_lengths:.1f}"
                    else:
                        log_str = f"Ep: {self.epoch_num}\trwd: {np.mean(mean_rewards):.1f}\tfps_step: {fps_step:.1f}\tfps_total: {fps_total:.1f}\tep_time:{epoch_end - epoch_start:.1f}\tframe: {self.frame}\teps_len: {mean_lengths:.1f}"
                    print(log_str)
                    # self.game_combined_rewards = torch.zeros_like(self.current_rewards).flatten().to(self.ppo_device)
                    self.vnet_pred = torch.empty(0).to(self.ppo_device)
                    self.vnet_gt = torch.empty(0).to(self.ppo_device)
                    self.vnet_loss = 0

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                if self.save_freq > 0 and (epoch_num % self.save_freq == 0):
                    if not self._do_finetune:
                        self.save(model_output_file)
                    else: # save finetuned model
                        torch.save(self.valuenet.state_dict(), model_output_file + '_valuenet.pth')
                    print('latest model saved')

                    if (self._save_intermediate) and (epoch_num % (self.save_freq * 5) == 0):
                        # Save intermediate model every save_freq  epoches
                        if not self._do_finetune:
                            int_model_output_file = model_output_file + '_' + str(
                                epoch_num).zfill(8)
                            self.save(int_model_output_file)
                        else: # save finetuned model
                            int_model_output_file = model_output_file + '_valuenet_' + str(
                                epoch_num).zfill(8)
                            torch.save(self.valuenet.state_dict(), int_model_output_file + '.pth')
                        print('intermediate model saved')

                if epoch_num > self.max_epochs:
                    self.save(model_output_file)
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num

                update_time = 0
        return

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(
                            curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr, self.entropy_coef, self.epoch_num, 0,
                        curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(
                    self.last_lr, self.entropy_coef, self.epoch_num, 0,
                    av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls),
                                                'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(
                self.last_lr, self.entropy_coef, self.epoch_num, 0,
                av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        self._record_train_batch_info(batch_dict, train_info)
        return train_info

    def play_steps(self):
        self.set_eval()

        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)
            import pdb; pdb.set_trace()

            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data('states', n,
                                                   self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n,
                                               self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)

            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(
                1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards,
                                       mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = self._calc_advs(batch_dict)

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

        return

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_info = self._actor_loss(old_action_log_probs_batch,
                                      action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip,
                                       return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss

            a_clip_frac = torch.mean(a_info['actor_clipped'].float())

            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(),
                                          old_mu_batch, old_sigma_batch,
                                          reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist *
                           rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr,
            'lr_mul': lr_mul,
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)

        return

    def discount_values(self, mb_fdones, mb_values, mb_rewards,
                        mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[
                t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    def env_reset(self, env_ids=None):
        obs = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def _get_mean_rewards(self):
        return self.game_rewards.get_mean()

    def _load_config_params(self, config):
        self.last_lr = config['learning_rate']
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
        }
        return config

    def set_full_state_weights(self, weights):
        self.set_weights(weights)
        self.epoch_num = weights['epoch']
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])
        self.optimizer.load_state_dict(weights['optimizer'])
        self.frame = weights.get('frame', 0)
        self.last_mean_rewards = weights.get('last_mean_rewards', -100500)

        env_state = weights.get('env_state', None)
        self.vec_env.set_env_state(env_state)

    def _setup_action_space(self):
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(
            action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(
            action_space.high.copy()).float().to(self.ppo_device)
        return

    def _init_train(self):
        return

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs = obs_dict['obs']
        processed_obs = self._preproc_obs(obs)
        value = self.model.a2c_network.eval_critic(processed_obs)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _actor_loss(self, old_action_log_probs_batch, action_log_probs,
                    advantage, curr_e_clip):
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip,
                                        1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)

        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        clipped = clipped.detach()

        info = {'actor_loss': a_loss, 'actor_clipped': clipped.detach()}
        return info

    def _critic_loss(self, value_preds_batch, values, curr_e_clip,
                     return_batch, clip_value):
        if clip_value:
            value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2

        info = {'critic_loss': c_loss}
        return info

    def _calc_advs(self, batch_dict):
        returns = batch_dict['returns']
        values = batch_dict['values']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() +
                                                             1e-8)

        return advantages

    def _record_train_batch_info(self, batch_dict, train_info):
        return

    def _log_train_info(self, train_info, frame):
        self.writer.add_scalar('performance/update_time',
                               train_info['update_time'], frame)
        self.writer.add_scalar('performance/play_time',
                               train_info['play_time'], frame)
        self.writer.add_scalar(
            'losses/a_loss',
            torch_ext.mean_list(train_info['actor_loss']).item(), frame)
        self.writer.add_scalar(
            'losses/c_loss',
            torch_ext.mean_list(train_info['critic_loss']).item(), frame)

        self.writer.add_scalar(
            'losses/bounds_loss',
            torch_ext.mean_list(train_info['b_loss']).item(), frame)
        self.writer.add_scalar(
            'losses/entropy',
            torch_ext.mean_list(train_info['entropy']).item(), frame)
        self.writer.add_scalar(
            'info/last_lr',
            train_info['last_lr'][-1] * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/lr_mul', train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/e_clip',
                               self.e_clip * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar(
            'info/clip_frac',
            torch_ext.mean_list(train_info['actor_clip_frac']).item(), frame)
        self.writer.add_scalar('info/kl',
                               torch_ext.mean_list(train_info['kl']).item(),
                               frame)

        train_info_dict = {
            "update_time": train_info['update_time'],
            "play_time": train_info['play_time'],
            "a_loss": torch_ext.mean_list(train_info['actor_loss']).item(),
            "c_loss": torch_ext.mean_list(train_info['critic_loss']).item(),
            "bounds_loss": torch_ext.mean_list(train_info['b_loss']).item(),
            "entropy": torch_ext.mean_list(train_info['entropy']).item(),
            "last_lr": train_info['last_lr'][-1] * train_info['lr_mul'][-1],
            "lr_mul": train_info['lr_mul'][-1],
            "e_clip": self.e_clip * train_info['lr_mul'][-1],
            "clip_frac":
            torch_ext.mean_list(train_info['actor_clip_frac']).item(),
            "kl": torch_ext.mean_list(train_info['kl']).item(),
        }
        return train_info_dict

    def post_epoch(self, epoch_num):
        pass

    def _build_valuenet(self): # added by takez
        # import pdb; pdb.set_trace()
        config = self.cfg
        self.valuenet = ValuePoseNet(use_pose=self.use_pose, use_vel=self.use_vel, vru=self.use_vru)
        self.valuenet.to(self.device)
        self.valuenet.train()
        return self.valuenet