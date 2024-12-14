# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import sys
import os
import numpy as np
import joblib
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.flags import flags
from env.tasks.base_task import PORT, SERVER
import torch

# (flags.fixed_path, flags.real_path, flags.pred_path, flags.slow, flags.init_heading) = (False, False, False, False, True)

class TrajGenerator():
    def __init__(self, num_envs, episode_dur, num_verts, device, dtheta_max,
                 speed_min, speed_max, accel_max, sharp_turn_prob, motion_lib, hybridInitProb, flags=None):

        self._device = device
        self._dt = episode_dur / (num_verts - 1)
        self._dtheta_max = dtheta_max
        self._speed_min = speed_min
        self._speed_max = speed_max
        self._accel_max = accel_max
        self._sharp_turn_prob = sharp_turn_prob
        self._motion_lib = motion_lib
        self._hybrid_init_prob = hybridInitProb
        self._flags = flags
        self.inverted = torch.zeros(num_envs, dtype=torch.bool, device=self._device) # all False

        self._verts_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self._verts = self._verts_flat.view((num_envs, num_verts, 3))

        env_ids = torch.arange(self.get_num_envs(), dtype=np.int)

        if self._flags.real_path:
            print('!!!!loading real-world trajectories!!!!')
            # self.traj_data = joblib.load("../social-transmotion/data/saved_trajs/traj_data.pkl")
            self.traj_data = []
            if self._flags.jta_path: # dummy data
                self.traj_data_jta = joblib.load("../social-transmotion/data/saved_trajs/jta_all_visual_cues_train_trajs.pkl")
                # self.traj_data_jta = joblib.load("../social-transmotion/data/saved_trajs/jta_all_visual_cues_test_trajs.pkl")
                self.traj_data.append(self.traj_data_jta)
            if self._flags.jrdb_path:
                # self.traj_data_jrdb = joblib.load("../social-transmotion/data/saved_trajs/jrdb_all_visual_cues_train_trajs.pkl")
                # self.traj_data_jrdb = joblib.load("../social-transmotion/data/saved_trajs/jrdb_all_visual_cues_test_trajs_filterv2.pkl")
                self.traj_data_jrdb = joblib.load("../social-transmotion/data/saved_trajs/jrdb_all_visual_cues_train_trajs_filterv2.pkl")
                self.traj_data.append(self.traj_data_jrdb)
        if self._flags.pred_path:
            self.traj_pred_data = joblib.load("data/traj/traj_pred_data.pkl")
        self.heading = torch.zeros(num_envs, 1)

        # self.hybrid_init_prob = 0.5
        return

    def reset(self, env_ids, init_pos, root_vel=None, motion_ids=None, motion_times=None):
        n = len(env_ids)
        if (n > 0):
            num_verts = self.get_num_verts()
            dtheta = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0 # Sample the angles at each waypoint
            dtheta *= self._dtheta_max * self._dt

            dtheta_sharp = np.pi * (2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0) # Sharp Angles Angle
            sharp_probs = self._sharp_turn_prob * torch.ones_like(dtheta)
            sharp_mask = torch.bernoulli(sharp_probs) == 1.0
            dtheta[sharp_mask] = dtheta_sharp[sharp_mask]

            dtheta[:, 0] = np.pi * (2 * torch.rand([n], device=self._device) - 1.0) # Heading


            dspeed = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0
            dspeed *= self._accel_max * self._dt
            dspeed[:, 0] = (self._speed_max - self._speed_min) * torch.rand([n], device=self._device) + self._speed_min # Speed

            speed = torch.zeros_like(dspeed)
            speed[:, 0] = dspeed[:, 0]
            for i in range(1, dspeed.shape[-1]):
                speed[:, i] = torch.clip(speed[:, i - 1] + dspeed[:, i], self._speed_min, self._speed_max)

            ################################################
            if self._flags.fixed_path:
                dtheta[:, :] = 0 # ZL: Hacking to make everything 0
                dtheta[0, 0] = 0 # ZL: Hacking to create collision
                if len(dtheta) > 1:
                    dtheta[1, 0] = -np.pi # ZL: Hacking to create collision
                speed[:] = (self._speed_min + self._speed_max)/2
            ################################################

            if self._flags.slow:
                speed[:] = speed/4

            # import pdb; pdb.set_trace()
            if self._flags.adjust_root_vel:
                # import pdb; pdb.set_trace()
                root_speed = torch.norm(root_vel[:,:2], dim=-1)
                init_speed = speed[:, 0]
                speed_ratio = root_speed / init_speed # (envs) speed: (envs, 100)
                speed = speed_ratio.unsqueeze(-1) * speed
                speed = torch.clip(speed, self._speed_min, self._speed_max)

            dtheta = torch.cumsum(dtheta, dim=-1)

            # speed[:] = 6
            seg_len = speed * self._dt

            dpos = torch.stack([torch.cos(dtheta), -torch.sin(dtheta), torch.zeros_like(dtheta)], dim=-1)
            dpos *= seg_len.unsqueeze(-1)
            dpos[..., 0, 0:2] += init_pos[..., 0:2]
            vert_pos = torch.cumsum(dpos, dim=-2)

            self._verts[env_ids, 0, 0:2] = init_pos[..., 0:2]
            self._verts[env_ids, 1:] = vert_pos

            # import pdb; pdb.set_trace()

            ####### ZL: Loading random real-world trajectories #######
            if self._flags.real_path:
                # hybrid_init_prob = 0.5
                real_data_prob = torch.rand(n, device=self._device)
                real_data_num = int(torch.sum(real_data_prob > self._hybrid_init_prob))
                # print(f'generating with {real_data_num} real data')
                real_data_mask = real_data_prob > self._hybrid_init_prob

                jta_num = len(self.traj_data_jta) if self._flags.jta_path else 0
                jrdb_num = len(self.traj_data_jrdb) if self._flags.jrdb_path else 0
                data_num = jta_num + jrdb_num # jta_num or jrdb_num or jta_num+jrdb_num
                # randomly sample from the real-data trajectories
                rids = random.sample(range(data_num), real_data_num)
                traj = torch.zeros([real_data_num, num_verts, 3], device=self._device) # initialize

                if len(self.traj_data) == 1:
                    for i, id in enumerate(rids):
                        traj[i] = torch.from_numpy(self.traj_data[0][id]['traj'])[:num_verts]
                elif len(self.traj_data) == 2:
                    for i, id in enumerate(rids):
                        if id < jta_num:
                            traj[i] = torch.from_numpy(self.traj_data[0][id]['traj'])[:num_verts]
                        else:
                            traj[i] = torch.from_numpy(self.traj_data[1][id - jta_num]['traj'])[:num_verts]
                else:
                    raise ValueError('Invalid traj_data length')
                traj = traj.to(self._device).float()

                # import pdb; pdb.set_trace()
                # offset = traj[..., 0, 0:2] - init_pos[real_data_mask, 0:2]
                traj[..., 0:2] = traj[..., 0:2] - traj[..., 0, 0:2].unsqueeze(1)

                if self._flags.adjust_root_vel:
                    init_speed = torch.norm(traj[:, 1] - traj[:, 0], dim=-1)
                    init_speed = torch.clamp(init_speed, min=self._speed_min*self._dt)
                    root_speed = torch.norm(root_vel[real_data_mask, :2].clone(), dim=-1)
                    speed_ratio = root_speed.div(init_speed) * self._dt
                    speed_ratio = speed_ratio.to(self._device)
                    traj[..., 0:2] = (speed_ratio * traj[..., 0:2].T).T

                traj[..., 0:2] += init_pos[real_data_mask, 0:2].unsqueeze(1)
                self._verts[env_ids[real_data_mask]] = traj

            elif self._flags.pred_path:
                traj_data = self.traj_pred_data
                rids = random.sample(traj_data.keys(), n)
                print(f'generating with {rids}')
                # rids = [1]
                traj = torch.stack([
                    torch.from_numpy(
                        traj_data[id]['coord_dense'])[:num_verts]
                    for id in rids
                ], dim=0).to(self._device).float()

                traj[..., 0:2] = traj[..., 0:2] - (traj[..., 0, 0:2] - init_pos[..., 0:2])[:, None]
                self._verts[env_ids] = traj

            if self._flags.init_heading:
                # import pdb; pdb.set_trace()
                # calculate the heading of the first segment
                copied_verts = self._verts[env_ids].clone()
                dinit = (copied_verts[:, 1, :2].clone() - copied_verts[:, 0, :2].clone()) # initial velocity

                # root_vel と dinit の両方がゼロベクトルでないかチェック
                root_vel_mag = torch.sqrt(torch.sum(root_vel**2, dim=1))
                dinit_mag = torch.sqrt(torch.sum(dinit**2, dim=1))

                # ゼロベクトルである場合の処理（例：0 に置き換える）
                root_rot = torch.where(root_vel_mag > 0, torch.atan2(root_vel[..., 1], root_vel[..., 0]), torch.zeros_like(root_vel[..., 0]))
                init_heading = torch.where(dinit_mag > 0, torch.atan2(dinit[..., 1], dinit[..., 0]), torch.zeros_like(dinit[..., 0]))

                # init_heading = torch.atan2(dinit[..., 1], dinit[..., 0])
                # calculate the difference between the root rotation and the initial heading
                # root_rot = torch.atan2(root_vel[..., 1], root_vel[..., 0])
                rot_diff = init_heading - root_rot # この角度だけ回転．通常の処理
                if self._flags.heading_inversion: # 半数のデータを反転させる
                    inversion_flag = torch.rand(n, device=self._device) > 0.5
                    self.inverted[env_ids[inversion_flag]] = True
                    self.inverted[env_ids[~inversion_flag]] = False
                    # init_headingの反対方向に回転する
                    rot_diff[inversion_flag] = init_heading[inversion_flag] - root_rot[inversion_flag] + np.pi # rootに対し反対方向に回転
                # translate the trajectory to the origin
                origin = copied_verts[:, 0, 0:2].clone()
                origin = origin.unsqueeze(1)
                origin = origin.expand(-1, num_verts, 2)
                copied_verts[:, :, 0:2] -= origin # copied verts transformed
                # rotate the trajectory by the rotation matrix
                c, s = torch.cos(rot_diff), torch.sin(rot_diff)
                R = torch.stack([c, -s, s, c], dim=-1).view(-1, 2, 2)

                # おそらく，ここの行列処理がおかしい．
                # 書き下して確認する．
                copied_verts[:, :, 0:2] = torch.bmm(copied_verts[:, :, 0:2].clone(), R) # copied verts transformed

                # assert alignment
                # calculate here because the difference can be too small
                daligned = copied_verts[:, 1, :2].clone() - copied_verts[:, 0, :2].clone()
                daligned_mag = torch.sqrt(torch.sum(daligned**2, dim=1))
                aligned_heading = torch.where(daligned_mag > 0, torch.atan2(daligned[..., 1], daligned[..., 0]), torch.zeros_like(daligned[..., 0]))
                # aligned_heading = torch.atan2(daligned[..., 1], daligned[..., 0])
                fixed_rot_diff = root_rot - aligned_heading
                masked_rot_diff = fixed_rot_diff[~torch.all(dinit==0, dim=1)]
                # import pdb; pdb.set_trace()
                if len(masked_rot_diff) > 0:
                    if self._flags.heading_inversion:
                        rot_assert_flag1 = (masked_rot_diff[~inversion_flag].abs().max() > 1e-4) if len(masked_rot_diff[~inversion_flag]) > 0 else False
                        rot_assert_flag2 = (~torch.isclose(masked_rot_diff[inversion_flag].abs().max(), torch.tensor(np.pi))).any() if len(masked_rot_diff[inversion_flag]) > 0 else False
                        rot_assert = rot_assert_flag1 or rot_assert_flag2
                    else:
                        rot_assert = masked_rot_diff.abs().max() > 1e-4
                    if rot_assert:
                        print(f'rot_diff: {masked_rot_diff} is too large! alignment failed!')
                # translate the trajectory back
                copied_verts[:, :, 0:2] += origin
                self._verts[env_ids] = copied_verts
            if self._flags.add_noise:
                self._verts[env_ids] += torch.randn_like(self._verts[env_ids]) * 0.5
        return

    def show_inverted(self):
        return self.inverted

    def input_new_trajs(self, env_ids):
        import json
        import requests
        from scipy.interpolate import interp1d
        x = requests.get(
            f'http://{SERVER}:{PORT}/path?num_envs={len(env_ids)}')
        import pdb; pdb.set_trace()
        data_lists = [value for idx, value in x.json().items()]
        coord = np.array(data_lists)
        x = np.linspace(0, coord.shape[1] - 1, num = coord.shape[1])
        fx = interp1d(x, coord[..., 0], kind='linear')
        fy = interp1d(x, coord[..., 1], kind='linear')
        x4 = np.linspace(0, coord.shape[1] - 1, num = coord.shape[1] * 10)
        coord_dense = np.stack([fx(x4), fy(x4), np.zeros([len(env_ids), x4.shape[0]])], axis = -1)
        coord_dense = np.concatenate([coord_dense, coord_dense[..., -1:, :]], axis = -2)
        self._verts[env_ids] = torch.from_numpy(coord_dense).float().to(env_ids.device)
        return self._verts[env_ids]


    def get_num_verts(self):
        return self._verts.shape[1]

    def get_num_segs(self):
        return self.get_num_verts() - 1

    def get_num_envs(self):
        return self._verts.shape[0]

    def get_traj_duration(self):
        num_verts = self.get_num_verts()
        dur = num_verts * self._dt
        return  dur

    def get_traj_verts(self, traj_id):
        return self._verts[traj_id]

    def calc_pos(self, traj_ids, times):
        # import pdb; pdb.set_trace()
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self._verts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self._verts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        return pos

    def mock_calc_pos(self, env_ids, traj_ids, times, query_value_gradient):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self._verts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self._verts_flat[traj_ids * num_verts + seg_id1]
        import pdb; pdb.set_trace()

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        new_obs, func = query_value_gradient(env_ids, pos)
        if not new_obs is None:
            # ZL: computes grad
            with torch.enable_grad():
                new_obs.requires_grad_(True)
                new_val = func(new_obs)

                disc_grad = torch.autograd.grad(
                    new_val,
                    new_obs,
                    grad_outputs=torch.ones_like(new_val),
                    create_graph=False,
                    retain_graph=True,
                    only_inputs=True)
        return pos
