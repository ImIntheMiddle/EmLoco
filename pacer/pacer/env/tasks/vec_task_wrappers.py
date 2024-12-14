# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from gym import spaces
import numpy as np
import torch
from pacer.env.tasks.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython


class VecTaskCPUWrapper(VecTaskCPU):

    def __init__(self, task, rl_device, sync_frame_time=False, clip_observations=5.0):
        super().__init__(task, rl_device, sync_frame_time, clip_observations)
        return


class VecTaskGPUWrapper(VecTaskGPU):

    def __init__(self, task, rl_device, clip_observations=5.0):
        super().__init__(task, rl_device, clip_observations)
        return


class VecTaskPythonWrapper(VecTaskPython):

    def __init__(self, task, rl_device, clip_observations=5.0):
        super().__init__(task, rl_device, clip_observations)

        self._amp_obs_space = spaces.Box(np.ones(task.get_num_amp_obs()) * -np.Inf, np.ones(task.get_num_amp_obs()) * np.Inf)
        return

    def reset(self, env_ids=None):
        self.task.reset(env_ids)
        obs = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        return obs

    def get_value_func_obs(self, env_ids=None):
        return self.task.get_value_func_obs(env_ids)

    def save_video(self, game, exp_name, rew, real_traj, pred_traj):
        return self.task.save_video(game, exp_name, rew, real_traj, pred_traj)

    def raw_reward(self):
        return self.task.reward_raw.to(self.rl_device)

    def get_waypoint_traj(self):
        waypoint_traj = self.task.waypoint_traj.clone()
        # import pdb; pdb.set_trace()
        # normalize by origin
        waypoint_traj -= waypoint_traj[:,0].clone().unsqueeze(1)
        return waypoint_traj

    def get_init_pose(self):
        # import pdb; pdb.set_trace()
        init_pose = self.task.init_pose.clone()
        # normalize by origin
        init_pose -= init_pose[:,0].clone().unsqueeze(1)
        return init_pose

    def get_init_vel(self):
        init_vel = self.task.init_vel.clone()
        return init_vel

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)