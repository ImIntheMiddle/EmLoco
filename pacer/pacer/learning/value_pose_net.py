# This file is a part of the Plausibl.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

DISC_LOGIT_INIT_SCALE = 1.0

class ValuePoseNet(nn.Module):
    def __init__(self, use_pose, use_vel, hide_toe=True, hide_spine=True, normalize=True, vru=False, **kwargs):
        super().__init__(**kwargs)
        self.use_pose = use_pose
        self.hide_toe = hide_toe
        self.hide_spine = hide_spine
        self.normalize = normalize
        self.use_vel = use_vel
        self.use_vru = vru
        self._build_task_value_pose_mlp()
        self.smpl_skeleton = [[0,1],[1,2],[2,3],[0,5],[5,6],[6,7],[0,12],[12,13],[0,14],[14,15],[15,16],[16,17],[17,18],[0,19],[19,20],[20,21],[21,22],[22,23]]
        self.criterion = nn.MSELoss()
        if self.use_pose and self.use_vel:
            print("Using Pose and Velocity")
            self.net_forward = self.forward_full
        elif self.use_vel:
            print("Using Velocity")
            self.net_forward = self.forward_vel
        elif self.use_pose:
            print("Using Pose")
            self.net_forward = self.forward_pose
        else:
            print("!!!! ValuePoseNet is NOT Using Pose !!!!")
            self.net_forward = self.forward_traj
        return

    def _build_task_value_pose_mlp(self):
        self.traj_size = 13*2 if not self.use_vru else 5*2 # 13 or 5 waypoints, (x, y) TODO: modify this to use config
        assert (self.traj_size % 2 == 0)
        self.pose_size = 24*3 # 24 joints, (x, y, z) TODO: modify this to use config
        assert (self.pose_size % 3 == 0)
        self.vel_size = 2

        if self.use_pose and self.use_vel:
            mlp_input_shape = self.traj_size + self.pose_size + self.vel_size # 26 + 72 + 2 = 100
        elif self.use_pose:
            mlp_input_shape = self.traj_size + self.pose_size # 26 + 72 = 98
        elif self.use_vel:
            mlp_input_shape = self.traj_size + self.vel_size # 26 + 2 = 28
        else:
            mlp_input_shape = self.traj_size # 26
        fc1_out = int(mlp_input_shape/2) - 1 # 12 or 48 or 49
        fc2_out = int(fc1_out/2) # 6 or 24 or 25

        self._network = nn.Sequential()
        self._network.add_module('fc1', nn.Linear(mlp_input_shape, fc1_out))
        self._network.add_module('relu1', nn.ReLU())
        self._network.add_module('fc2', nn.Linear(fc1_out, fc2_out))
        self._network.add_module('relu2', nn.ReLU())
        self._network.add_module('fc3', nn.Linear(fc2_out, 1))
        self._network.add_module('sigmoid', nn.Sigmoid()) # output 0-1 value

        # initialize the network
        for m in self._network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        # show the network
        # print("ValuePoseNet: ", self._network)
        # print("Parameters: ", sum(p.numel() for p in self._network.parameters()))
        return

    def _rotate_normalization(self, waypoint_traj, init_pose=None, init_vel=None):
        # import pdb; pdb.set_trace()
        # self.visualize_pose(init_pose.clone().detach().cpu(), gt_xy=waypoint_traj.clone().detach().cpu(), label='before')
        x_vel = waypoint_traj[:, 1, 0]
        y_vel = waypoint_traj[:, 1, 1]

        # Add epsilon to denominator to avoid NaN in backward pass
        epsilon = 1e-10
        near_zeros = x_vel.abs() < epsilon
        x_vel = x_vel * (near_zeros.logical_not())
        x_vel = x_vel + (near_zeros * epsilon)
        angles = torch.atan2(y_vel, x_vel)
        rotation_matrix = torch.zeros(len(angles), 2, 2)
        rotation_matrix[:, 0, 0] = torch.cos(angles)
        rotation_matrix[:, 0, 1] = -torch.sin(angles)
        rotation_matrix[:, 1, 0] = torch.sin(angles)
        rotation_matrix[:, 1, 1] = torch.cos(angles)
        rotation_matrix = rotation_matrix.to(waypoint_traj.device)
        # rotate trajectory xy coordinates so that the first waypoint is at (0, 0)
        waypoint_traj_rotated = torch.bmm(waypoint_traj[...,:2], rotation_matrix)
        # print(torch.atan2(waypoint_traj[:, 1, 1], waypoint_traj[:, 1, 0]))
        # assert torch.allclose(torch.atan2(waypoint_traj_rotated[:, 1, 1], waypoint_traj_rotated[:, 1, 0]), torch.zeros(len(waypoint_traj_rotated)).to(waypoint_traj_rotated.device), atol=1e-4), "First waypoint should be at (0, 0)"
        # rotate pose xy coordinates as well
        if init_pose is not None:
            init_pose[..., :2] = torch.bmm(init_pose[:,:,:2].clone(), rotation_matrix)
        if init_vel is not None:
            # import pdb; pdb.set_trace()
            init_vel = torch.bmm(init_vel[:,:2].clone().unsqueeze(1), rotation_matrix)[:,0]

        # self.visualize_pose(init_pose.clone().detach().cpu(), gt_xy=waypoint_traj_rotated.clone().detach().cpu(), label='after')
        return waypoint_traj_rotated, init_pose, init_vel

    def forward(self, waypoint_traj, init_pose=None, init_vel=None):
        if self.normalize:
            waypoint_traj, init_pose, init_vel = self._rotate_normalization(waypoint_traj, init_pose, init_vel)
        return self.net_forward(waypoint_traj, init_pose, init_vel)

    def forward_traj(self, waypoint_traj, init_pose=None, init_vel=None): # Ensure that poses are not included
        # assert init_pose is None, "init_pose should not be included"
        waypoint_traj = waypoint_traj.reshape(-1, self.traj_size)
        value = self._network(waypoint_traj)
        return value

    def forward_pose(self, waypoint_traj, init_pose, init_vel=None):
        assert init_pose is not None, "init_pose should be included"
        # import pdb; pdb.set_trace()
        waypoint_traj = waypoint_traj.reshape(-1, self.traj_size)
        if self.hide_toe: # hide 10, 11
            init_pose[:,[4, 8]] = 0
        if self.hide_spine:
            init_pose[:,[9, 10, 11]] = 0
        init_pose = init_pose.reshape(-1, self.pose_size)
        obs = torch.cat([waypoint_traj, init_pose], dim=-1)
        value = self._network(obs)
        return value

    def forward_vel(self, waypoint_traj, init_pose=None, init_vel=None):
        assert init_vel is not None, "init_vel should be included"
        waypoint_traj = waypoint_traj.reshape(-1, self.traj_size)
        init_vel = init_vel.reshape(-1, self.vel_size)
        obs = torch.cat([waypoint_traj, init_vel], dim=-1)
        value = self._network(obs)
        return value

    def forward_full(self, waypoint_traj, init_pose, init_vel):
        assert init_pose is not None, "init_pose should be included"
        assert init_vel is not None, "init_vel should be included"
        waypoint_traj = waypoint_traj.reshape(-1, self.traj_size)
        if self.hide_toe: # hide 10, 11
            init_pose[:,[4, 8]] = 0
        if self.hide_spine:
            init_pose[:,[9, 10, 11]] = 0
        init_pose = init_pose.reshape(-1, self.pose_size)
        init_vel = init_vel.reshape(-1, self.vel_size)
        obs = torch.cat([waypoint_traj, init_pose, init_vel], dim=-1)
        value = self._network(obs)
        return value

    def calc_embodied_motion_loss(self, pred_traj, init_pose=None, init_vel=None):
        # import pdb; pdb.set_trace()
        if self.normalize:
            pred_traj, init_pose, init_vel = self._rotate_normalization(pred_traj, init_pose, init_vel)
        pred_value = self.net_forward(pred_traj, init_pose, init_vel)
        # loss: mse between pred_value and 1
        loss = self.criterion(pred_value, torch.ones_like(pred_value)) # ideally, the value should be 1
        # loss = -pred_value # maximize the value
        return pred_value, loss

    def visualize_pose(self, init_pose, past_xy=None, gt_xy=None, bb_size=None, bb_order=None, frame_id=None, ped_id=None, label='', before_pose=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-0.6,0.6)
        ax.set_ylim(-0.6,0.6)
        ax.set_zlim(-0.6,0.6)

        for i, joint in enumerate(init_pose[0]):
            if i in [4,8,9,10,11]:
                continue
            ax.scatter(joint[0], joint[1], joint[2], c='k', marker='o')
            ax.text(joint[0], joint[1], joint[2], f'{i}', color='k')
        for i, edge in enumerate(self.smpl_skeleton):
            ax.plot(init_pose[0,edge,0], init_pose[0,edge,1], init_pose[0,edge,2], c='r', linewidth=1.5)
        if before_pose is not None:
            import pdb; pdb.set_trace()
            for t, pose in enumerate(before_pose):
                if np.isnan(pose).any():
                    continue
                root = past_xy[0][t]
                for i, joint in enumerate(pose):
                    if i in [4,8,9,10,11]:
                        continue
                    joint[0] += root[0]
                    joint[1] += root[1]
                    ax.scatter(joint[0], joint[1], joint[2], c='b', marker='o')
                    ax.text(joint[0], joint[1], joint[2], f'{i}', color='b')
                for i, edge in enumerate(self.smpl_skeleton):
                    ax.plot(pose[edge,0], pose[edge,1], pose[edge,2], c='b', linewidth=1.5)
                break
        if past_xy is not None:
            ax.plot(past_xy[0,:,0], past_xy[0,:,1], np.zeros(len(past_xy)), c='b', linewidth=1.5, label='Past')
        if gt_xy is not None:
            ax.plot(gt_xy[0,:,0], gt_xy[0,:,1], np.zeros(len(gt_xy)), c='g', linewidth=1.5, label='GT')
        title = 'Pose in ValuePoseNet'
        if (bb_size is not None) and (bb_order is not None):
            title += f', BB size: {int(bb_size)}, BB order: {int(bb_order)}'
        if (frame_id is not None) and (ped_id is not None):
            title += f', Frame: {int(frame_id)}, Ped: {int(ped_id)}'
        ax.set_title(title)
        plt.legend()
        savename = f'pose_VPNet_{label}.png' if label != '' else 'pose_VPNet.png'
        plt.savefig(savename)
        plt.close()