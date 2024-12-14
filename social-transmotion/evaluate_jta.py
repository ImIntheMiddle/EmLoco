import sys
sys.path.append("/misc/dl00/halo/plausibl/pacer/pacer")
import os
import json
import argparse
import torch
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from progress.bar import Bar
from torch.utils.data import DataLoader

from dataset_jta import batch_process_coords, create_dataset, collate_batch
from model_jta import create_model
from utils.utils import create_logger, load_default_config, load_config
from utils.metrics import calculate_initial_yaw_error, calculate_velocity, calculate_acceleration, calculate_ang_velocity, calculate_ang_acceleration, calculate_chi_distance

from learning.value_pose_net import ValuePoseNet

def inference(model, config, input_joints, padding_mask, out_len=14, limit_obs=False):
    model.eval()
    with torch.no_grad():
        if torch.isnan(input_joints).any():
            # logger.info('Nan detected!')
            # masking nan values with zeros
            input_joints = torch.where(torch.isnan(input_joints), torch.zeros_like(input_joints), input_joints)
        # add noise only to the input trajectory
        if config["NOISY_TRAJ"]: # gaussian(0, 0.25)
            input_joints[:,:,0,:2] = input_joints[:,:,0,:2] + torch.randn_like(input_joints[:,:,0,:2]) * config["NOISY_TRAJ"]
        # import pdb; pdb.set_trace()
        pred_joints = model(input_joints, padding_mask, limit_obs=limit_obs)

    output_joints = pred_joints[:,-out_len:]

    return output_joints

class Visualizer_3D():
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.smpl_skeleton = [[0,1],[1,2],[2,3],[0,5],[5,6],[6,7],[0,12],[12,13],[0,14],[14,15],[15,16],[16,17],[17,18],[0,19],[19,20],[20,21],[21,22],[22,23]]
        self.color = ['g', 'blue', 'darkorange', 'gold']
        # self.color = ['g', 'magenta', 'gold']
        # self.color = ['k', 'k', 'g', 'gold']

    def plot_3d(self, fig, gt_xy, pred_xy, past_xy, init_pose, id_b, id_k, ade, values, names=['Pred'], past_len=9):
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=40, azim=-50) # 正面ver
        # ax.view_init(elev=30, azim=-150) # 正面ver
        # 反対
        # ax.view_init(elev=40, azim=-140) # 背面ver
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        xlim = (-3, 6)
        ylim = (-3, 1)
        zlim = (-2, 1.5)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_zlim(zlim[0], zlim[1])
        ax.set_xticks(np.arange(xlim[0], xlim[1], 1))
        ax.set_yticks(np.arange(ylim[0], ylim[1], 2))
        ax.set_zticks(np.arange(zlim[0], zlim[1], 1))
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.axes.zaxis.set_visible(False)
        ax.axes.zaxis.set_tick_params(width=0)
        ax.set_zticks([])
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]
        ax.set_box_aspect([x_range, y_range, z_range])  # x, y, zのアスペクト比を1:1:1に設定

        # plot past trajectory
        if past_len == 1:
            ax.scatter(past_xy[-1, 0], past_xy[-1, 1], np.zeros(1)-1.5, c='k', label='Past trajectory (1 frame)', marker='o', s=10)
            # ax.scatter(past_xy[-1, 1], -past_xy[-1, 0], np.zeros(1), c='k', label='Past trajectory (1 frame)', marker='o')
        elif past_len > 1:
            ax.plot(past_xy[-past_len:, 0], past_xy[-past_len:, 1], np.zeros(past_len)-1.5, c='k', label=f'Past Trajectory ({past_len}frame)', linewidth=1.5, linestyle='-', marker='o', markersize=3)
            # ax.plot(past_xy[-past_len:, 0], past_xy[-past_len:, 1], np.zeros(past_len)-1.5, c='k', label=f'Past trajectory ({past_len}frame)', linewidth=1.5)
            # ax.scatter(past_xy[-1, 0], past_xy[-1, 1], np.zeros(1)-1.5, c='k', marker='o', s=20)
        else:
            raise ValueError('past_len should be greater than 0')

        # scaling pose for better visualization
        init_pose *= 2

        # plot skeleton
        for i, edge in enumerate(self.smpl_skeleton):
            ax.plot(init_pose[edge,0], init_pose[edge,1], init_pose[edge,2], c='magenta', linewidth=1.5, marker='o', markersize=2)
            # ax.plot(init_pose[edge,0], init_pose[edge,1], init_pose[edge,2], c='r', linewidth=1.5)

        # plot GT trajectory
        ax.plot(gt_xy[:,0], gt_xy[:,1], np.zeros(13)-1.5, c='r', label='Ground Truth', linewidth=2, linestyle='-', marker='o', markersize=3)
        # ax.plot(gt_xy[:,0], gt_xy[:,1], np.zeros(13)-1.5, c='b', label='GT', linewidth=2)

        for id, pred in enumerate(pred_xy): # for predictions of each model
            # import pdb; pdb.set_trace()
            if id == 0:
                # 凡例にだけ加える
                label_i = f'{names[id]}'
                ax.plot(np.zeros(1), np.zeros(1), np.zeros(1), c=self.color[id], label=label_i, linestyle='--', linewidth=2.5, marker='o', markersize=0)
                continue
            # pred = np.expand_dims(pred, axis=1) if len(pred.shape) == 2 else pred
            for i in range(pred.shape[1]): #for each mode
                # if i in [0, 1]:
                #     continue
                if (values[id][0] is not None) and (values[id][1] is not None):
                    label_i = f'{names[id]}' if i == 0 else None
                    # label_i = f'{names[id]}, Value: {values[id][0][i]:.2f}'
                    # change color based on the value (value is 0 to 1)
                    color = plt.cm.viridis(values[id][0][i])
                    # ax.plot(pred[:,i,0], pred[:,i,1], np.zeros(13)-1.5, c=self.color[id], label=label_i, linestyle='--', linewidth=2.5, marker='o', markersize=3)
                    # ax.plot(pred[:,i,0], pred[:,i,1], np.zeros(13)-1.5, c=color, label=label_i, linestyle='--', linewidth=2.5, marker='o', markersize=3)
                    ax.plot(pred[:,i,0], pred[:,i,1], np.zeros(13)-1.5, c=self.color[id], label=label_i, linestyle='--', linewidth=2.5)
                else:
                    label_i = f'{names[id]}' if i == 0 else None
                    ax.plot(pred[:,i,0], pred[:,i,1], np.zeros(13)-1.5, c=self.color[id], label=label_i, linestyle='--', linewidth=2, marker='o', markersize=3)
                    # ax.plot(pred[:,i,1], -pred[:,i,0], np.zeros(13), c=self.color[id], label=f'{names[id]}', linestyle='--', linewidth=2)

        # fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, shrink=0.6)
        ax.legend(fontsize=12, ncol=2)

        return fig, ax

    def save_plot(self, gt_xy, pred_xy, past_xy, init_pose, id_b, id_k, ade, values=[[None, None]], names=['Pred'], past_len=9):
        fig = plt.figure(figsize=(4,4))
        fig, ax = self.plot_3d(fig, gt_xy, pred_xy, past_xy, init_pose, id_b, id_k, ade, values, names, past_len=past_len)
        # save
        plt.savefig(os.path.join(self.save_dir, f'batch{id_b}_person{id_k}_ade{ade[0]:.2f}.png'))
        plt.savefig(os.path.join(self.save_dir, f'batch{id_b}_person{id_k}_ade{ade[0]:.2f}.pdf'), bbox_inches="tight")
        plt.close()

def evaluate_ade_fde(model, valuenet, split, modality_selection, dataloader, bs, config, logger, exp_name, return_all=False, visualize=False, limit_obs=False, bar_prefix="", per_joint=False, show_avg=False):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    # bar = Bar(f"EVAL ADE_FDE", fill="#", max=len(dataloader))
    bar = tqdm.tqdm(dataloader, desc="EVAL ADE_FDE", dynamic_ncols=True)

    batch_size = bs
    batch_id = 0
    ade = 0
    fde = 0
    des = np.zeros(12)
    min_ade = 0
    min_fde = 0
    max_ade = 0
    max_fde = 0
    iye = 0 # initial yaw error
    ade_batch = 0
    fde_batch = 0
    ade_batch_min = 0
    fde_batch_min = 0
    ade_batch_max = 0
    fde_batch_max = 0
    filter_threshold = config["MODEL"]["value_threshold"]
    ade_value = 0
    fde_value = 0
    ade_random = 0
    fde_random = 0
    minade_value = 0
    minfde_value = 0
    ade_filtered = 0
    fde_filtered = 0
    des_batch = np.zeros(12)
    sample_num = 0
    sample_num_value_sampling = 0
    sample_num_filtered = 0
    iye_batch = 0
    largest_ade = 0
    value_list = []
    value_gt_list = []
    value_loss_list = []
    value_loss_gt_list = []
    ade_list = []
    fde_list = []
    gt_primitive = {'velocity': [], 'acceleration': [], 'ang_velocity': [], 'ang_acceleration': []}
    pred_primitive = {'velocity': [], 'acceleration': [], 'ang_velocity': [], 'ang_acceleration': []}
    vis_dict = {'label':['gt_xy', 'pred_xy', 'past_xy', 'id_b', 'id_k', 'ade', 'pred_values', 'pred_value_gt', 'fde'], 'data':[]}

    if visualize:
        save_dir = os.path.join('./experiments/JTA', exp_name, 'visualization', '3d_plot', split, modality_selection)
        if valuenet is not None:
            valuenet_path = config["MODEL"]["valuenet_checkpoint"]
            save_dir = os.path.join(save_dir, valuenet_path.split('/')[-1].split('.')[0])
        logger.info(f"save_dir: {save_dir}")
        vis_3d = Visualizer_3D(save_dir)

    break_flag = False
    for i, batch in enumerate(bar):
        joints, masks, padding_mask = batch
        padding_mask = padding_mask.to(config["DEVICE"])
        primary_init_pose = joints[:, 0, 8, 3:27, :3]
        # import pdb; pdb.set_trace()
        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config, modality_selection)
        # primary_init_pose = in_joints[:, 8, 1:20, :3] # for jta dataset
        # zeros_pose = torch.zeros(primary_init_pose.size(0), primary_init_pose.size(1)+5, 3).to(config["DEVICE"])
        # needed_joints = list(range(0, 4)) + list(range(5, 8)) + list(range(12, 24))
        # zeros_pose[:,needed_joints] = primary_init_pose
        # primary_init_pose = zeros_pose
        pred_joints = inference(model, config, in_joints, padding_mask, out_len=out_F, limit_obs=limit_obs)

        in_joints = in_joints.cpu()
        out_joints = out_joints.cpu() # ground truth
        pred_joints = pred_joints.cpu().reshape(out_joints.size(0), 12, -1, 2)

        iye_batch += calculate_initial_yaw_error(out_joints[:,0,0,:2], pred_joints[:,0,0,:2]).sum() # all scene, initial frame, primary person

        for k in range(len(out_joints)):
            person_past_joints = in_joints[k,:,0:1]
            person_out_joints = out_joints[k,:,0:1]
            person_pred_joints = pred_joints[k,:,:]

            # copy appropriately
            init_pose = primary_init_pose[k].clone().detach().to(config["DEVICE"]).float()
            init_pose[..., 2] = -init_pose[..., 2] # transformation for jta dataset
            init_vel = (in_joints[k,8,0,:2] - in_joints[k,7,0,:2]).clone().detach() * 2.5
            init_vel = init_vel.to(config["DEVICE"])

            gt_xy = person_out_joints[:,0,:2] # ground truth xy
            pred_xys = person_pred_joints[:,:,:2] # predicted xy
            past_xy = person_past_joints[:,0,:2] # past xy

            # calculate motion primitive for gt
            gt_vel = calculate_velocity(gt_xy)
            gt_primitive['velocity'] = np.concatenate((gt_primitive['velocity'], gt_vel), axis=0) if len(gt_primitive['velocity']) else gt_vel
            gt_accel = calculate_acceleration(gt_vel)
            gt_primitive['acceleration'] = np.concatenate((gt_primitive['acceleration'], gt_accel), axis=0) if len(gt_primitive['acceleration']) else gt_accel
            gt_angvel = calculate_ang_velocity(gt_xy)
            gt_primitive['ang_velocity'] = np.concatenate((gt_primitive['ang_velocity'], gt_angvel), axis=0) if len(gt_primitive['ang_velocity']) else gt_angvel
            gt_angaccel = calculate_ang_acceleration(gt_angvel)
            gt_primitive['ang_acceleration'] = np.concatenate((gt_primitive['ang_acceleration'], gt_angaccel), axis=0) if len(gt_primitive['ang_acceleration']) else gt_angaccel

            # calculate motion primitive for pred
            sum_ade_mean = 0
            sum_ade_min = 1e5
            sum_ade_max = 0
            scene_fde_min = 1e5
            scene_fde_mean = 0
            scene_fde_max = 0
            des_mean = np.zeros(12)
            gt_xy = gt_xy.detach().cpu().numpy()
            num_mode = pred_xys.size(1)
            # import pdb; pdb.set_trace()
            candidates = []
            pred_values = []
            for p in range(num_mode):
                pred_xy = pred_xys[:,p]
                pred_vel = calculate_velocity(pred_xy)
                pred_primitive['velocity'] = np.concatenate((pred_primitive['velocity'], pred_vel), axis=0) if len(pred_primitive['velocity']) else pred_vel
                pred_accel = calculate_acceleration(pred_vel)
                pred_primitive['acceleration'] = np.concatenate((pred_primitive['acceleration'], pred_accel), axis=0) if len(pred_primitive['acceleration']) else pred_accel
                pred_angvel = calculate_ang_velocity(pred_xy)
                pred_primitive['ang_velocity'] = np.concatenate((pred_primitive['ang_velocity'], pred_angvel), axis=0) if len(pred_primitive['ang_velocity']) else pred_angvel
                pred_angaccel = calculate_ang_acceleration(pred_angvel)
                pred_primitive['ang_acceleration'] = np.concatenate((pred_primitive['ang_acceleration'], pred_angaccel), axis=0) if len(pred_primitive['ang_acceleration']) else pred_angaccel

                # import pdb; pdb.set_trace()
                sum_ade = 0
                des = []
                for t in range(12):
                    d1 = (gt_xy[t,0] - pred_xy[t,0].detach().cpu().numpy())
                    d2 = (gt_xy[t,1] - pred_xy[t,1].detach().cpu().numpy())

                    dist_ade = np.linalg.norm([d1,d2])
                    des.append(dist_ade)
                    sum_ade += np.linalg.norm(dist_ade)
                sum_ade /= 12
                sum_ade_mean += sum_ade
                if sum_ade < sum_ade_min:
                    sum_ade_min = sum_ade
                if sum_ade > sum_ade_max:
                    sum_ade_max = sum_ade
                d3 = (gt_xy[-1,0] - pred_xy[-1,0].detach().cpu().numpy())
                d4 = (gt_xy[-1,1] - pred_xy[-1,1].detach().cpu().numpy())
                dist_fde = [d3,d4]
                scene_fde = np.linalg.norm(dist_fde)
                scene_fde_mean += scene_fde
                if scene_fde < scene_fde_min:
                    scene_fde_min = scene_fde
                if scene_fde > scene_fde_max:
                    scene_fde_max = scene_fde

                # import pdb; pdb.set_trace()
                des_mean += np.array(des) # sum of all des for each mode

                # concat origin to trajectory
                gt_xy = np.concatenate((np.zeros((1,2)), gt_xy), axis=0) if len(gt_xy) == out_F else gt_xy
                gt_traj = torch.tensor(gt_xy).to(config["DEVICE"]).float()
                pred_xy = np.concatenate((np.zeros((1,2)), pred_xy), axis=0)
                pred_traj = torch.tensor(pred_xy).to(config["DEVICE"]).float()

                if (valuenet is not None) and (not torch.isnan(init_pose).any()) and (not torch.isnan(pred_traj).any()):
                    # import pdb; pdb.set_trace()
                    with torch.no_grad():
                        pred_value, value_loss = valuenet.calc_embodied_motion_loss(pred_traj.unsqueeze(0), init_pose.unsqueeze(0), init_vel.unsqueeze(0))
                        pred_value_gt, value_loss_gt = valuenet.calc_embodied_motion_loss(gt_traj.unsqueeze(0), init_pose.unsqueeze(0), init_vel.unsqueeze(0))
                        # import pdb; pdb.set_trace()
                        # valuenet.visualize_pose(init_pose.unsqueeze(0).cpu(), past_xy.unsqueeze(0).cpu(), gt_traj.unsqueeze(0).cpu())
                        pred_value = pred_value.item()
                        pred_value_gt = pred_value_gt.item()
                        value_list.append(pred_value)
                        value_gt_list.append(pred_value_gt)
                        value_loss_list.append(value_loss.item())
                        value_loss_gt_list.append(value_loss_gt.item())

                        if num_mode > 1:
                            candidates.append((sum_ade, scene_fde))
                            pred_values.append(pred_value)

                else:
                    value_loss, value_loss_gt = None, None
                    pred_value, pred_value_gt = None, None

            if num_mode > 1 and valuenet is not None:
                # import pdb; pdb.set_trace()
                ade_list.extend([c[0] for c in candidates])
                fde_list.extend([c[1] for c in candidates])
                id_random = random.randint(0, num_mode-1)
                ade_random += candidates[id_random][0]
                fde_random += candidates[id_random][1]

                id_maxvalue = np.argmax(pred_values)

                # filter the candidate with values higher than threshold
                filtered_candidates = [c for c in candidates if pred_values[candidates.index(c)] >= filter_threshold]
                out_candidates = [c for c in candidates if pred_values[candidates.index(c)] < filter_threshold]
                if len(filtered_candidates) == 0:
                    ade_value += candidates[id_maxvalue][0]
                    fde_value += candidates[id_maxvalue][1]
                    minade_value += candidates[id_maxvalue][0]
                    minfde_value += candidates[id_maxvalue][1]
                    ade_filtered += candidates[id_maxvalue][0]
                    fde_filtered += candidates[id_maxvalue][1]
                    sample_num_value_sampling += 1
                else: # if there are candidates with values higher than threshold
                    minade_tmp = 1e5
                    minfde_tmp = 1e5
                    for c in filtered_candidates:
                        ade_value += c[0]
                        fde_value += c[1]
                        sample_num_value_sampling += 1
                        if c[0] < minade_tmp:
                            minade_tmp = c[0]
                        if c[1] < minfde_tmp:
                            minfde_tmp = c[1]
                    minade_value += minade_tmp
                    minfde_value += minfde_tmp
                for c in out_candidates:
                    ade_filtered += c[0]
                    fde_filtered += c[1]
                    sample_num_filtered += 1

            # if visualize and sum_ade > largest_ade/1.5:
            if visualize:
                # import pdb; pdb.set_trace()
                pred_xys = pred_xys.detach().cpu().numpy()
                pred_xys = np.concatenate((np.zeros((1,pred_xys.shape[1],2)), pred_xys), axis=0)
                largest_ade = sum_ade if sum_ade > largest_ade else largest_ade
                # vis_i = [0,2]
                # vis_k = [10,49]
                # if i in vis_i and k in vis_k:
                # vis_dict['data'].append([gt_xy, pred_xys, past_xy, init_pose.cpu(), i, k, sum_ade, pred_values, pred_value_gt, scene_fde])
                vis_dict['data'].append([gt_xy, pred_xys, past_xy, init_pose.cpu(), i, k, sum_ade_mean/num_mode, pred_values, pred_value_gt, scene_fde])
                # vis_dict['data'].append([None, None, None, None, i, k, sum_ade, pred_value, pred_value_gt, scene_fde])
                    # 3d plot of initial pose and pred/GT trajectories
                    # valuenet.visualize_pose(init_pose.unsqueeze(0).cpu(), past_xy.unsqueeze(0).cpu(), gt_traj.unsqueeze(0).cpu(), bbox_sizes[k].cpu(), bbox_order[k], frame_id, ped_id)
                # if i > max(vis_i):
                if i > 10:
                    break_flag = True
                    break
            # import pdb; pdb.set_trace()
            ade_batch += sum_ade_mean / num_mode
            fde_batch += scene_fde_mean / num_mode
            des_batch += des_mean / num_mode
            ade_batch_min += sum_ade_min
            fde_batch_min += scene_fde_min
            ade_batch_max += sum_ade_max
            fde_batch_max += scene_fde_max
            sample_num += 1
        if break_flag:
            break
        batch_id+=1

    # import pdb; pdb.set_trace()
    ade = ade_batch / sample_num if sample_num > 0 else 0
    fde = fde_batch / sample_num if sample_num > 0 else 0
    des = des_batch / sample_num if sample_num > 0 else 0
    min_ade = ade_batch_min / sample_num if sample_num > 0 else 0
    min_fde = fde_batch_min / sample_num if sample_num > 0 else 0
    max_ade = ade_batch_max / sample_num if sample_num > 0 else 0
    max_fde = fde_batch_max / sample_num if sample_num > 0 else 0
    iye = iye_batch / sample_num if sample_num > 0 else 0
    chi_square_dict = calculate_chi_distance(gt_primitive, pred_primitive)

    logger.info(f'Total samples: {sample_num}')

    logger.info(f'ADE: {ade:.5f}')
    logger.info(f'FDE: {fde:.5f}')
    logger.info(f'Min ADE: {min_ade:.5f}')
    logger.info(f'Min FDE: {min_fde:.5f}')
    logger.info(f'Worst ADE: {max_ade:.5f}')
    logger.info(f'Worst FDE: {max_fde:.5f}')
    logger.info(f'IYE: {iye:.5f}')
    logger.info(f'DES: {np.round(des, 5)}')
    logger.info(f'Chi-square distance:\n Velocity: {chi_square_dict["velocity"]:.5f},\n Acceleration: {chi_square_dict["acceleration"]:.5f},\n Angular velocity: {chi_square_dict["ang_velocity"]:.5f},\n Angular acceleration: {chi_square_dict["ang_acceleration"]:.5f}')

    if num_mode > 1:
        ade_value_sample = ade_value / sample_num_value_sampling if sample_num_value_sampling > 0 else 0
        fde_value_sample = fde_value / sample_num_value_sampling if sample_num_value_sampling > 0 else 0
        ade_random_sample = ade_random / sample_num if sample_num > 0 else 0
        fde_random_sample = fde_random / sample_num if sample_num > 0 else 0
        minade_value_sample = minade_value / sample_num if sample_num > 0 else 0
        minfde_value_sample = minfde_value / sample_num if sample_num > 0 else 0
        ade_filtered = ade_filtered / sample_num_filtered if sample_num_filtered > 0 else 0
        fde_filtered = fde_filtered / sample_num_filtered if sample_num_filtered > 0 else 0
        logger.info(f'Threadhold: {filter_threshold}')
        logger.info(f'ADE with Value sampling: {ade_value_sample:.5f}')
        logger.info(f'FDE with Value sampling: {fde_value_sample:.5f}')
        logger.info(f'ADE with Random sampling: {ade_random_sample:.5f}')
        logger.info(f'FDE with Random sampling: {fde_random_sample:.5f}')
        logger.info(f'Min ADE with Value sampling: {minade_value_sample:.5f}')
        logger.info(f'Min FDE with Value sampling: {minfde_value_sample:.5f}')
        logger.info(f'ADE of rejected samples: {ade_filtered:.5f}')
        logger.info(f'FDE of rejected samples: {fde_filtered:.5f}')

        # import pdb; pdb.set_trace()
        value_array = np.array(value_list) # 0 to 1
        ade_array = np.array(ade_list)
        fde_array = np.array(fde_list)

        bins = np.arange(0, 1.05, 0.1) # 0 to 1
        bin_centers = (bins[:-1] + bins[1:]) / 2 # 0.025 to 0.975

        # valueの各ビンごとにade，fdeの平均値を計算
        value_indices = np.digitize(value_array, bins)
        # カラーマップの正規化
        # norm = Normalize(vmin=0, vmax=1)  # カラーマップの範囲を0から1に設定
        # colors = plt.cm.viridis(norm(value_array))  # カラーマップを使って色を作成

        # valueの各ビンごとにade，fdeの平均値を計算
        ade_mean_values = [ade_array[value_indices == i].mean() if np.any(value_indices == i) else np.nan for i in range(1, len(bins))]
        fde_mean_values = [fde_array[value_indices == i].mean() if np.any(value_indices == i) else np.nan for i in range(1, len(bins))]
        plt.bar(bin_centers, ade_mean_values, color=plt.cm.viridis(bin_centers), width=0.1, alpha=0.9, edgecolor='white')
        plt.xlabel('Plausibility score', fontsize=14)
        plt.ylabel('ADE', fontsize=14)
        plt.xticks(bins)
        plt.rcParams["font.size"] = 12
        # y軸の範囲を限定
        plt.ylim(0, 6)
        # bar label
        for i in range(len(bin_centers)):
            if ade_mean_values[i] > 0:
                # plt.text(bin_centers[i], ade_mean_values[i], f'{len(ade_array[value_indices == (i+1)])}', ha='center', va='bottom', fontsize=11)
                plt.text(bin_centers[i], ade_mean_values[i], f'{ade_mean_values[i]:.2f}', ha='center', va='bottom', fontsize=11)
        plt.savefig(os.path.join('./experiments/JTA', exp_name, 'value_ade_barplot.png'))
        # pdf
        plt.savefig(os.path.join('./experiments/JTA', exp_name, 'value_ade_barplot.pdf'), bbox_inches="tight")
        plt.close()
        plt.bar(bin_centers, fde_mean_values, color=plt.cm.viridis(bin_centers), width=0.1, alpha=0.9, edgecolor='white')
        plt.xlabel('Plausibility score', fontsize=14)
        plt.ylabel('FDE', fontsize=14)
        plt.xticks(bins)
        for i in range(len(bin_centers)):
            if fde_mean_values[i] > 0:
                plt.text(bin_centers[i], fde_mean_values[i], f'{len(fde_array[value_indices == (i+1)])}', ha='center', va='bottom', fontsize=11)
        plt.savefig(os.path.join('./experiments/JTA', exp_name, 'value_fde_barplot.png'))
        plt.savefig(os.path.join('./experiments/JTA', exp_name, 'value_fde_barplot.pdf'), bbox_inches="tight")
        plt.close()

        # ヒストグラムを作成
        # import pdb; pdb.set_trace()
        counts, bins = np.histogram(value_array, bins=10, range=(0, 1))
        # 自然対数でカウントを変換
        counts_log = np.log10(counts+1) # 0の場合はlog10(1)=0
        # viridisのカラーマップを使って色を作成
        plt.ylim(0, 4.5)
        plt.bar(bin_centers, counts_log, width=0.1, color=plt.cm.viridis(bin_centers), alpha=0.9, edgecolor='white')
        for i in range(len(bin_centers)):
            if counts[i] > 0:
                # plt.text(bin_centers[i], ade_mean_values[i], f'{len(ade_array[value_indices == (i+1)])}', ha='center', va='bottom', fontsize=11)
                plt.text(bin_centers[i], counts_log[i], f'{counts[i]}', ha='center', va='bottom', fontsize=11)
        plt.xlabel('Plausibility score', fontsize=14)
        plt.ylabel('Number of samples (log)', fontsize=14)
        plt.xticks(bins)
        plt.savefig(os.path.join('./experiments/JTA', exp_name, 'value_hist.png'))
        plt.savefig(os.path.join('./experiments/JTA', exp_name, 'value_hist.pdf'), bbox_inches="tight")

    # save vis_dict
    if visualize:
        past_frame_num = limit_obs if limit_obs!=0 else in_F
        with open(os.path.join(save_dir, f'vis_dict_{past_frame_num}frame.pkl'), 'wb') as f:
            pickle.dump(vis_dict, f)
        logger.info(f'vis_dict saved to {os.path.join(save_dir, f"vis_dict_{past_frame_num}frame.pkl")}!')

    if args.valueloss:
        # import pdb; pdb.set_trace()
        logger.info(f'Value:  {np.mean(value_list):.3f}')
        logger.info(f'Value GT: {np.mean(value_gt_list):.3f}')
        logger.info(f'Value Loss: {np.mean(value_loss_list):.3f}')
        logger.info(f'Value Loss GT: {np.mean(value_loss_gt_list):.3f}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,  help="checkpoint path")
    parser.add_argument("--split", type=str, default="test", help="Split to use. one of [train, test, valid]")
    parser.add_argument("--metric", type=str, default="ade_fde", help="Evaluation metric. One of (vim, mpjpe)")
    parser.add_argument("--modality", type=str, default="traj+all", help="available modality combination from['traj','traj+2dbox','traj+3dpose','traj+2dpose','traj+3dpose+3dbox','traj+all', 'traj+2dbox+2dpose', 'traj+2dbox+2dpose+3dbox]")
    parser.add_argument("--vis", action="store_true", help="Visualize the predictions")
    parser.add_argument('--limit_obs', type=int, default=0, help="Limit the number of observations")
    parser.add_argument("--valueloss", action="store_true", help="Use value loss")
    parser.add_argument("--all_frames", action="store_true", help="Evaluate all observation frames")
    parser.add_argument("--noisy_traj", type=float, default=0, help="Add noise to the trajectory to mimic real data")
    parser.add_argument("--multi_modal", action="store_true", help="Use multi-modal model")
    parser.add_argument("--last_epoch", action="store_true", help="Use last epoch checkpoint")
    parser.add_argument("--no_pose", action="store_true", help="No pose in valuenet")
    parser.add_argument("--no_vel", action="store_true", help="No velocity in valuenet")
    parser.add_argument("--epoch", type=str, default=0, help="Epoch to evaluate")
    parser.add_argument("--filter_threshold", type=float, default=0.7, help="Threshold for filtering samples")
    args = parser.parse_args()

    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    ################################
    # Load checkpoint
    ################################

    if args.last_epoch:
        ckpt_name = f'./experiments/JTA/{args.exp_name}/checkpoints/checkpoint.pth.tar'
    elif args.epoch != 0:
        if os.path.exists(f'./experiments/JTA/{args.exp_name}/checkpoints/best_val_checkpoint_{args.epoch}epoch.pth.tar'):
            ckpt_name = f'./experiments/JTA/{args.exp_name}/checkpoints/best_val_checkpoint_{args.epoch}epoch.pth.tar'
        elif os.path.exists(f'./experiments/JTA/{args.exp_name}/checkpoints/checkpoint_{args.epoch}epoch.pth.tar'):
            ckpt_name = f'./experiments/JTA/{args.exp_name}/checkpoints/checkpoint_{args.epoch}epoch.pth.tar'
        elif os.path.exists(f'./experiments/default/{args.exp_name}/checkpoints/best_val_{args.epoch}.pth.tar'):
            ckpt_name = f'./experiments/default/{args.exp_name}/checkpoints/best_val_{args.epoch}.pth.tar'
        elif os.path.exists(f'./experiments/JTA/{args.exp_name}/checkpoints/best_val_{args.epoch}.pth.tar'):
            ckpt_name = f'./experiments/JTA/{args.exp_name}/checkpoints/best_val_{args.epoch}.pth.tar'
        else:
            raise FileNotFoundError("Checkpoint not found")
    else:
        ckpt_name = f'./experiments/JTA/{args.exp_name}/checkpoints/best_val_checkpoint.pth.tar'

    logdir = os.path.join('./experiments/JTA', args.exp_name, 'eval_logs')
    os.makedirs(logdir, exist_ok=True)
    logger = create_logger(logdir)
    logger.info(f'Loading checkpoint from {ckpt_name}')
    ckpt = torch.load(ckpt_name, map_location = torch.device('cpu'))
    config = ckpt['config']
    new_cfg = load_config("configs/jta_all_visual_cues.yaml")
    exp_name = args.exp_name

    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed(0)
    else:
        config["DEVICE"] = "cpu"

    config["NOISY_TRAJ"] = args.noisy_traj
    config["MULTI_MODAL"] = args.multi_modal
    config["MODEL"]["value_threshold"] = args.filter_threshold
    use_pose = not args.no_pose
    use_vel = not args.no_vel

    if args.valueloss:
        valuenet = ValuePoseNet(use_pose=use_pose, use_vel=use_vel)
        valuenet_ckpt = new_cfg["MODEL"].get("valuenet_checkpoint", "/home/halo/plausibl/pacer/output/exp/pacer/valuenet_realpath_JTA+JRDB_valuenet_00025000.pth")
        config["MODEL"]["valuenet_checkpoint"] = valuenet_ckpt
        if valuenet_ckpt != "":
            logger.info(f"Loading checkpoint from {valuenet_ckpt}")
            valuenet.load_state_dict(torch.load(valuenet_ckpt, map_location = torch.device('cpu')))
        else:
            logger.info("No checkpoint provided for valuenet. Using random weights.")
        valuenet.eval()
        valuenet.requires_grad_ = False
        valuenet.to(config["DEVICE"])
    else:
        valuenet = None

    logger.info("Initializing with config:")
    logger.info(config)

    ################################
    # Initialize model
    ################################

    model = create_model(config, logger)
    model = torch.nn.DataParallel(model).to(config["DEVICE"])
    # import pdb; pdb.set_trace()
    model.load_state_dict(ckpt['model'])

    ################################
    # Load data
    ################################

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    assert in_F == 9
    assert out_F == 12

    name = config['DATA']['train_datasets']

    dataset = create_dataset(name[0], logger, split=args.split, track_size=(in_F+out_F), track_cutoff=in_F, preprocessed=config['DATA']['preprocessed'])

    bs = new_cfg['TRAIN']['batch_size']*10
    # bs = 13
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=config['TRAIN']['num_workers'], shuffle=False, collate_fn=collate_batch)
    logger.info(f'Evaluating with modality: {args.modality}')

    if args.all_frames:
        for obs_i in [1,2,3,4,5,6,7,8,0]: # 0 to 8
            obs_len = 9 if obs_i == 0 else obs_i
            logger.info(f"Evaluating with {obs_len} frames")
            evaluate_ade_fde(model, valuenet, args.split, args.modality, dataloader, bs, config, logger, exp_name, return_all=True, visualize=args.vis, limit_obs=obs_i)
    else:
        obs_len = 9 if args.limit_obs == 0 else args.limit_obs
        logger.info(f"Evaluating with {obs_len} frames")
        evaluate_ade_fde(model, valuenet, args.split, args.modality, dataloader, bs, config, logger, exp_name, return_all=True, visualize=args.vis, limit_obs=args.limit_obs)