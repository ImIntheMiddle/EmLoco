import sys
sys.path.append("/misc/dl00/halo/plausibl/pacer/pacer")
import os
import argparse
import torch
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from progress.bar import Bar
from torch.utils.data import DataLoader

from dataset_jrdb import collate_batch, batch_process_coords, get_datasets, create_dataset
from model_jta import create_model
from evaluate_jta import Visualizer_3D, inference
from evaluate_jrdb import evaluate_ade_fde
from utils.utils import create_logger, load_default_config, load_config
from utils.metrics import calculate_initial_yaw_error, calculate_velocity, calculate_acceleration, calculate_ang_velocity, calculate_ang_acceleration, calculate_chi_distance

from learning.value_pose_net import ValuePoseNet

def evaluate_ade_fde(model, valuenet, split, modality_selection, dataloader, bs, config, logger, exp_name, return_all=False, visualize=False, limit_obs=False, bar_prefix="", per_joint=False, show_avg=False):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    # bar = Bar(f"EVAL ADE_FDE", fill="#", max=len(dataloader))
    bar = tqdm.tqdm(dataloader, desc="EVAL ADE_FDE", dynamic_ncols=True)

    batch_size = bs
    batch_id = 0
    ade = 0
    fde = 0
    iye = 0 # initial yaw error
    ade_batch = 0
    fde_batch = 0
    iye_batch = 0
    largest_ade = 0
    value_list = []
    value_gt_list = []
    value_loss_list = []
    value_loss_gt_list = []
    gt_primitive = {'velocity': [], 'acceleration': [], 'ang_velocity': [], 'ang_acceleration': []}
    pred_primitive = {'velocity': [], 'acceleration': [], 'ang_velocity': [], 'ang_acceleration': []}
    vis_dict = {'label':['gt_xy', 'pred_xy', 'past_xy', 'id_b', 'id_k', 'ade', 'pred_values', 'pred_value_gt'], 'data':[]}

    if visualize:
        save_dir = os.path.join('./experiments/JRDB', exp_name, 'visualization', '3d_plot', split, modality_selection)
        if valuenet is not None:
            valuenet_path = config["MODEL"]["valuenet_checkpoint"]
            save_dir = os.path.join(save_dir, valuenet_path.split('/')[-1].split('.')[0])
        logger.info(f"save_dir: {save_dir}")
        vis_3d = Visualizer_3D(save_dir)

    break_flag = False
    for i, batch in enumerate(bar):
        joints, masks, padding_mask = batch
        padding_mask = padding_mask.to(config["DEVICE"])
        # import pdb; pdb.set_trace()
        primary_init_pose = joints[:, 0, 8, 2:, :3]
        primary_bbox = joints[:, 0, 8, 1]

        # import pdb; pdb.set_trace()
        jta_like_joints = torch.zeros(joints.shape[0], joints.shape[1], joints.shape[2], 49, joints.shape[4])
        # fill in the joints
        jta_like_joints[:, :, :, :2, :] = joints[:, :, :, :2, :]
        jta_like_joints[:, :, :, 3:27, :] = joints[:, :, :, 2:, :]
        # mask should be [batch, person, frame, 49]
        filled_mask = torch.zeros(masks.shape[0], masks.shape[1], masks.shape[2], 49)
        filled_mask[:, :, :, :2] = masks[:, :, :, :2]
        filled_mask[:, :, :, 3:27] = masks[:, :, :, 2:]

        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(jta_like_joints, filled_mask, padding_mask, config, modality_selection)
        pred_joints = inference(model, config, in_joints, padding_mask, out_len=out_F, limit_obs=limit_obs)

        in_joints = in_joints.cpu()
        out_joints = out_joints.cpu()
        pred_joints = pred_joints.cpu().reshape(out_joints.size(0), 12, 1, 2)

        iye_batch += calculate_initial_yaw_error(out_joints[:,0,0,:2], pred_joints[:,0,0,:2]).sum() # all scene, initial frame, primary person

        for k in range(len(out_joints)):
            person_past_joints = in_joints[k,:,0:1]
            person_out_joints = out_joints[k,:,0:1]
            person_pred_joints = pred_joints[k,:,0:1]

            init_pose = primary_init_pose[k].clone().detach().to(config["DEVICE"]).float()
            init_pose[..., 0] = -init_pose[..., 0] # transformation for jrdb dataset
            init_vel = (in_joints[k,8,0,:2] - in_joints[k,7,0,:2]).clone().detach() * 2.5
            init_vel = init_vel.to(config["DEVICE"])

            gt_xy = person_out_joints[:,0,:2]
            pred_xy = person_pred_joints[:,0,:2]
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
            pred_vel = calculate_velocity(pred_xy)
            pred_primitive['velocity'] = np.concatenate((pred_primitive['velocity'], pred_vel), axis=0) if len(pred_primitive['velocity']) else pred_vel
            pred_accel = calculate_acceleration(pred_vel)
            pred_primitive['acceleration'] = np.concatenate((pred_primitive['acceleration'], pred_accel), axis=0) if len(pred_primitive['acceleration']) else pred_accel
            pred_angvel = calculate_ang_velocity(pred_xy)
            pred_primitive['ang_velocity'] = np.concatenate((pred_primitive['ang_velocity'], pred_angvel), axis=0) if len(pred_primitive['ang_velocity']) else pred_angvel
            pred_angaccel = calculate_ang_acceleration(pred_angvel)
            pred_primitive['ang_acceleration'] = np.concatenate((pred_primitive['ang_acceleration'], pred_angaccel), axis=0) if len(pred_primitive['ang_acceleration']) else pred_angaccel

            sum_ade = 0
            for t in range(12):
                d1 = (gt_xy[t,0].detach().cpu().numpy() - pred_xy[t,0].detach().cpu().numpy())
                d2 = (gt_xy[t,1].detach().cpu().numpy() - pred_xy[t,1].detach().cpu().numpy())

                dist_ade = [d1,d2]
                sum_ade += np.linalg.norm(dist_ade)
            sum_ade /= 12
            ade_batch += sum_ade
            d3 = (gt_xy[-1,0].detach().cpu().numpy() - pred_xy[-1,0].detach().cpu().numpy())
            d4 = (gt_xy[-1,1].detach().cpu().numpy() - pred_xy[-1,1].detach().cpu().numpy())
            dist_fde = [d3,d4]
            scene_fde = np.linalg.norm(dist_fde)
            fde_batch += scene_fde

            # concat origin to trajectory
            gt_xy = np.concatenate((np.zeros((1,2)), gt_xy), axis=0)
            gt_traj = torch.tensor(gt_xy).to(config["DEVICE"]).float()
            pred_xy = np.concatenate((np.zeros((1,2)), pred_xy), axis=0)
            pred_traj = torch.tensor(pred_xy).to(config["DEVICE"]).float()

            if (valuenet is not None) and (not torch.isnan(init_pose).any()) and (not torch.isnan(pred_traj).any()):
                with torch.no_grad():
                    # import pdb; pdb.set_trace()
                    pred_value, value_loss = valuenet.calc_embodied_motion_loss(pred_traj.unsqueeze(0), init_pose.unsqueeze(0), init_vel.unsqueeze(0))
                    pred_value_gt, value_loss_gt = valuenet.calc_embodied_motion_loss(gt_traj.unsqueeze(0), init_pose.unsqueeze(0), init_vel.unsqueeze(0))
                    # import pdb; pdb.set_trace()
                    # frame_id, ped_id = int(frame_pedids[k][0][8][0]), int(frame_pedids[k][0][8][1])
                    # valuenet.visualize_pose(init_pose.unsqueeze(0).cpu(), past_xy.unsqueeze(0).cpu(), gt_traj.unsqueeze(0).cpu(), bbox_sizes[k].cpu(), bbox_order[k], frame_id, ped_id)
                    pred_value = pred_value.item()
                    pred_value_gt = pred_value_gt.item()
                    value_list.append(pred_value)
                    value_gt_list.append(pred_value_gt)
                    value_loss_list.append(value_loss.item())
                    value_loss_gt_list.append(value_loss_gt.item())
            else:
                value_loss, value_loss_gt = None, None
                pred_value, pred_value_gt = None, None

                # if visualize and sum_ade > largest_ade/1.5:
                if visualize and (not torch.isnan(init_pose).any()) and (not torch.isnan(pred_traj).any()):
                    # import pdb; pdb.set_trace()
                    largest_ade = sum_ade if sum_ade > largest_ade else largest_ade
                    if i < 50:
                        vis_dict['data'].append([gt_xy, pred_xy, past_xy, init_pose.cpu(), i, k, sum_ade, pred_value, pred_value_gt])
                        # 3d plot of initial pose and pred/GT trajectories
                        # valuenet.visualize_pose(init_pose.unsqueeze(0).cpu(), past_xy.unsqueeze(0).cpu(), gt_traj.unsqueeze(0).cpu(), bbox_sizes[k].cpu(), bbox_order[k], frame_id, ped_id)
                    else:
                        break_flag = True

        if break_flag:
            break
        batch_id+=1

    ade = ade_batch/((batch_id-1)*batch_size+len(out_joints))
    fde = fde_batch/((batch_id-1)*batch_size+len(out_joints))
    iye = iye_batch/((batch_id-1)*batch_size+len(out_joints)) # initial yaw error
    chi_square_dict = calculate_chi_distance(gt_primitive, pred_primitive)

    logger.info(f'ADE: {ade:.5f}')
    logger.info(f'FDE: {fde:.5f}')
    logger.info(f'IYE: {iye:.5f}')
    logger.info(f'Chi-square distance:\n Velocity: {chi_square_dict["velocity"]:.5f},\n Acceleration: {chi_square_dict["acceleration"]:.5f},\n Angular velocity: {chi_square_dict["ang_velocity"]:.5f},\n Angular acceleration: {chi_square_dict["ang_acceleration"]:.5f}')

    # save vis_dict
    if visualize:
        frame_len = args.limit_obs if args.limit_obs!=0 else in_F
        with open(os.path.join(save_dir, f'vis_dict_{frame_len}frame.pkl'), 'wb') as f:
            pickle.dump(vis_dict, f)
        logger.info('vis_dict saved!')

    if args.valueloss:
        # import pdb; pdb.set_trace()
        logger.info(f'Data with value:  {len(value_list)}/{len(dataloader.dataset)}')
        logger.info(f'Value:  {np.mean(value_list):.3f}')
        logger.info(f'Value GT: {np.mean(value_gt_list):.3f}')
        logger.info(f'Value Loss: {np.mean(value_loss_list):.3f}')
        logger.info(f'Value Loss GT: {np.mean(value_loss_gt_list):.3f}')

        # make histogram
        import matplotlib.pyplot as plt
        plt.hist(value_gt_list, bins=50, label='pred')
        plt.savefig('value_gt_hist.png')

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
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ################################
    # Load checkpoint
    ################################

    # ckpt_name = f'./experiments/JTA/{args.exp_name}/checkpoints/checkpoint.pth.tar'
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

    if args.valueloss:
        valuenet = ValuePoseNet(use_pose=True, use_vel=True, hide_toe=True, normalize=True)
        valuenet_ckpt = config["MODEL"].get("valuenet_checkpoint", "/home/halo/plausibl/pacer/output/exp/pacer/valuenet_realpath_JTA+JRDB_valuenet_00025000.pth")
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

    name = "jrdb_all_visual_cues"

    dataset = create_dataset(name, logger, split=args.split, track_size=(in_F+out_F), track_cutoff=in_F, preprocessed=config['DATA']['preprocessed'])

    bs = new_cfg['TRAIN']['batch_size']*5
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=config['TRAIN']['num_workers'], shuffle=False, collate_fn=collate_batch)

    if args.all_frames:
        for obs_i in [1,2,3,4,5,6,7,8,0]: # 0 to 8
            obs_len = 9 if obs_i == 0 else obs_i
            logger.info(f"Evaluating with {obs_len} frames")
            evaluate_ade_fde(model, valuenet, args.split, args.modality, dataloader, bs, config, logger, exp_name, return_all=True, visualize=args.vis, limit_obs=obs_i)
    else:
        evaluate_ade_fde(model, valuenet, args.split, args.modality, dataloader, bs, config, logger, exp_name, return_all=True, visualize=args.vis, limit_obs=args.limit_obs)