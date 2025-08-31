import sys
sys.path.append("/misc/dl00/halo/plausibl/pacer/pacer")
import os
import json
import argparse
import torch
import random
import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from torch.utils.data import DataLoader
from scipy.stats import rankdata

from dataset_amass import batch_process_coords, create_dataset, collate_batch
from model_amass import create_model
from utils.utils import create_logger, load_default_config, load_config
from utils.metrics import calculate_initial_yaw_error, calculate_velocity, calculate_acceleration, calculate_ang_velocity, calculate_ang_acceleration, calculate_chi_distance

from evaluate_jta import inference, Visualizer_3D
from learning.value_pose_net import ValuePoseNet

def evaluate_ade_fde(model, valuenet, split, modality_selection, dataloader, bs, config, logger, exp_name, return_all=False, visualize=False, limit_obs=False, bar_prefix="", per_joint=False, show_avg=False):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    # bar = Bar(f"EVAL ADE_FDE", fill="#", max=len(dataloader))
    bar = tqdm.tqdm(dataloader, desc="EVAL ADE_FDE", dynamic_ncols=True)

    batch_size = bs
    batch_id = 0
    iye = 0 # initial yaw error
    total_ade = 0
    total_fde = 0
    total_des = np.zeros(out_F)
    total_sample_num = 0
    sample_with_pose = 0
    sample_num_value_sampling = 0
    iye_batch = 0
    largest_ade = 0
    min_ade = 0
    min_fde = 0
    ade_batch_min = 0
    fde_batch_min = 0
    filter_threshold = config["MODEL"]["value_threshold"]
    sample_num = 0
    sample_num_value_sampling = 0
    sample_num_filtered = 0
    ade_value = 0
    fde_value = 0
    ade_random = 0
    fde_random = 0
    minade_value = 0
    minfde_value = 0
    ade_filtered = 0
    fde_filtered = 0
    minade_filtered = 0
    mindfe_filtered = 0
    sample_num_minfilt = 0
    ade_list = []
    fde_list = []
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
        # import pdb; pdb.set_trace()
        joints, masks, padding_mask = batch
        padding_mask = padding_mask.to(config["DEVICE"])
        primary_init_pose = joints[:, 0, 11, 1:, :3]

        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config, modality_selection)
        pred_joints = inference(model, config, in_joints, padding_mask, out_len=out_F, limit_obs=limit_obs)

        in_joints = in_joints.cpu()
        out_joints = out_joints.cpu()
        pred_joints = pred_joints.cpu().reshape(out_joints.size(0), out_F, -1, 2)

        iye_batch += calculate_initial_yaw_error(out_joints[:,0,0,:2], pred_joints[:,0,0,:2]).sum() # all scene, initial frame, primary person

        for k in range(len(out_joints)):
            person_past_joints = in_joints[k,:,0:1]
            person_out_joints = out_joints[k,:,0:1] # GT
            person_pred_joints = pred_joints[k,:,:]

            init_pose = primary_init_pose[k].clone().detach().to(config["DEVICE"]).float()
            init_pose[..., 0] = -init_pose[..., 0] # transformation for jrdb dataset
            init_vel = (in_joints[k,11,0,:2] - in_joints[k,10,0,:2]).clone().detach() * 30 # 30fps
            init_vel = init_vel.to(config["DEVICE"])

            gt_xy = person_out_joints[:,0,:2]
            pred_xys = person_pred_joints[:,:,:2]
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
            scene_fde_min = 1e5
            scene_fde_mean = 0
            des_mean = np.zeros(out_F)
            gt_xy = gt_xy.detach().cpu().numpy()
            num_mode = pred_xys.size(1)
            candidates = []
            pred_values = []
            for p in range(pred_xys.size(1)):
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
                scene_des = []
                for t in range(out_F):
                    d1 = (gt_xy[t,0] - pred_xy[t,0].detach().cpu().numpy())
                    d2 = (gt_xy[t,1] - pred_xy[t,1].detach().cpu().numpy())

                    dist_ade = np.linalg.norm([d1, d2])
                    scene_des.append(dist_ade)
                    sum_ade += dist_ade
                sum_ade /= out_F
                sum_ade_mean += sum_ade
                if sum_ade < sum_ade_min:
                    sum_ade_min = sum_ade
                d3 = (gt_xy[-1,0] - pred_xy[-1,0].detach().cpu().numpy())
                d4 = (gt_xy[-1,1] - pred_xy[-1,1].detach().cpu().numpy())
                dist_fde = [d3, d4]
                scene_fde = np.linalg.norm(dist_fde)
                scene_fde_mean += scene_fde
                if scene_fde < scene_fde_min:
                    scene_fde_min = scene_fde

                des_mean += np.array(scene_des)
                total_sample_num += 1
                total_ade += sum_ade
                total_fde += scene_fde
                total_des += np.array(scene_des)

                # concat origin to trajectory
                gt_xy = np.concatenate((np.zeros((1,2)), gt_xy), axis=0) if len(gt_xy)==out_F else gt_xy
                gt_traj = torch.tensor(gt_xy).to(config["DEVICE"]).float() if type(gt_xy) != torch.Tensor else gt_xy
                pred_xy = np.concatenate((np.zeros((1,2)), pred_xy), axis=0)
                pred_traj = torch.tensor(pred_xy).to(config["DEVICE"]).float()
                if (not torch.isnan(init_pose).any()): sample_with_pose += 1
                if (valuenet is not None) and (not torch.isnan(init_pose).any()) and (not torch.isnan(pred_traj).any()):
                # if (valuenet is not None) and (not torch.isnan(pred_traj).any()):
                    with torch.no_grad():
                        sum_ade_mean += sum_ade
                        scene_fde_mean += scene_fde
                        # import pdb; pdb.set_trace()
                        pred_value, value_loss = valuenet.calc_embodied_motion_loss(pred_traj.unsqueeze(0), init_pose.unsqueeze(0), init_vel.unsqueeze(0))
                        pred_value_gt, value_loss_gt = valuenet.calc_embodied_motion_loss(gt_traj.unsqueeze(0), init_pose.unsqueeze(0), init_vel.unsqueeze(0))
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
                if len(out_candidates) > 0:
                    minade_filtered += min([c[0] for c in out_candidates])
                    mindfe_filtered += min([c[1] for c in out_candidates])
                    sample_num_minfilt += 1

            # if visualize and sum_ade > largest_ade/1.5:
            if visualize and (not torch.isnan(init_pose).any()) and (not torch.isnan(pred_traj).any()):
                # import pdb; pdb.set_trace()
                pred_xys = pred_xys.detach().cpu().numpy()
                pred_xys = np.concatenate((np.zeros((1,pred_xys.shape[1],2)), pred_xys), axis=0)
                largest_ade = sum_ade if sum_ade > largest_ade else largest_ade
                vis_dict['data'].append([gt_xy, pred_xys, past_xy, init_pose.cpu(), i, k, sum_ade, pred_value, pred_value_gt])

                if i > 30:
                    break_flag = True
                    break

            ade_batch_min += sum_ade_min
            fde_batch_min += scene_fde_min
            sample_num += 1

        if break_flag:
            break
        batch_id+=1

    total_ade /= total_sample_num
    total_fde /= total_sample_num
    total_des /= total_sample_num
    min_ade = ade_batch_min / sample_num if sample_num > 0 else 0
    min_fde = fde_batch_min / sample_num if sample_num > 0 else 0

    iye = iye_batch/((batch_id-1)*batch_size+len(out_joints)) # initial yaw error
    chi_square_dict = calculate_chi_distance(gt_primitive, pred_primitive)

    logger.info(f'Total sample num: {total_sample_num}')
    logger.info(f'Total sample num with pose: {sample_with_pose}')
    logger.info(f'Total ADE: {total_ade:.5f}')
    logger.info(f'Total FDE: {total_fde:.5f}')
    logger.info(f'Min ADE: {min_ade:.5f}')
    logger.info(f'Min FDE: {min_fde:.5f}')
    logger.info(f'Total DES: {np.round(total_des, 3)}')
    logger.info(f'IYE: {iye:.5f}')
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
        minade_filtered = minade_filtered / sample_num_minfilt if sample_num_minfilt > 0 else 0
        mindfe_filtered = mindfe_filtered / sample_num_minfilt if sample_num_minfilt > 0 else 0
        logger.info(f'Threadhold: {filter_threshold}')
        logger.info(f'ADE with Value sampling: {ade_value_sample:.5f}')
        logger.info(f'FDE with Value sampling: {fde_value_sample:.5f}')
        logger.info(f'ADE with Random sampling: {ade_random_sample:.5f}')
        logger.info(f'FDE with Random sampling: {fde_random_sample:.5f}')
        logger.info(f'Min ADE with Value sampling: {minade_value_sample:.5f}')
        logger.info(f'Min FDE with Value sampling: {minfde_value_sample:.5f}')
        logger.info(f'ADE of rejected samples: {ade_filtered:.5f}')
        logger.info(f'FDE of rejected samples: {fde_filtered:.5f}')
        logger.info(f'Min ADE of rejected samples: {minade_filtered:.5f}')
        logger.info(f'Min FDE of rejected samples: {mindfe_filtered:.5f}')

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
        plt.hist(value_gt_list, bins=50, label='pred')
        plt.savefig('value_gt_hist.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,  help="checkpoint path")
    parser.add_argument("--split", type=str, default="test", help="Split to use. one of [train, test, valid]")
    parser.add_argument("--metric", type=str, default="ade_fde", help="Evaluation metric. One of (vim, mpjpe)")
    parser.add_argument("--modality", type=str, default="traj+all", help="available modality combination from['traj','traj+all']")
    parser.add_argument("--vis", action="store_true", help="Visualize the predictions")
    parser.add_argument("--limit_obs", type=int, default=0, help="Limit the number of observations")
    parser.add_argument("--valueloss", action="store_true", help="Use value loss")
    parser.add_argument("--epoch", type=str, default=0, help="Epoch to evaluate")
    parser.add_argument("--multi_modal", action="store_true", help="Use multi-modal model")
    # parser.add_argument("--all_frames", action="store_true", help="Use all frames for evaluation")
    parser.add_argument("--filter_threshold", type=float, default=0.7, help="Threshold for filtering samples")

    args = parser.parse_args()

    SEED = 40
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ################################
    # Load checkpoint
    ################################
    print(f"loading checkpoint from {args.exp_name}, epoch {args.epoch}...")
    if args.epoch == 0:
        if os.path.exists(f'./experiments/AMASS/{args.exp_name}/checkpoints/best_val_checkpoint.pth.tar'):
            ckpt_name = f'./experiments/AMASS/{args.exp_name}/checkpoints/best_val_checkpoint.pth.tar'
    elif args.epoch != 0:
        if os.path.exists(f'./experiments/AMASS/{args.exp_name}/checkpoints/best_val_checkpoint_{args.epoch}epoch.pth.tar'):
            ckpt_name = f'./experiments/AMASS/{args.exp_name}/checkpoints/best_val_checkpoint_{args.epoch}epoch.pth.tar'
        elif os.path.exists(f'./experiments/AMASS/{args.exp_name}/checkpoints/checkpoint_{args.epoch}epoch.pth.tar'):
            ckpt_name = f'./experiments/AMASS/{args.exp_name}/checkpoints/checkpoint_{args.epoch}epoch.pth.tar'
        else:
            raise FileNotFoundError("Checkpoint not found")
    else:
        raise FileNotFoundError("Checkpoint not found")

    logdir = os.path.join('./experiments/AMASS', args.exp_name, 'eval_logs')
    os.makedirs(logdir, exist_ok=True)
    logger = create_logger(logdir)
    logger.info(f'Loading checkpoint from {ckpt_name}')
    ckpt = torch.load(ckpt_name, map_location = torch.device('cpu'))
    config = ckpt['config']
    new_cfg = load_config("configs/amass_all_visual_cues.yaml")
    exp_name = args.exp_name

    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed(SEED)
    else:
        config["DEVICE"] = "cpu"

    config['NOISY_TRAJ'] = False
    config["MULTI_MODAL"] = args.multi_modal
    config["MODEL"]["value_threshold"] = args.filter_threshold

    if args.valueloss:
        valuenet = ValuePoseNet(use_pose=True, use_vel=True, amass=True)
        # valuenet_ckpt = new_cfg["MODEL"].get("valuenet_checkpoint", "/home/halo/plausibl/pacer/output/exp/pacer/v3_init_last15_valuenet.pth")
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
    logger.info("Input Modality: " + args.modality)

    ################################
    # Initialize model
    ################################

    model = create_model(config, logger).to(config["DEVICE"])
    # model = torch.nn.DataParallel(model).to(config["DEVICE"])
    logger.info(f"Model loaded from {ckpt_name}, epoch {ckpt['epoch']}")
    # import pdb; pdb.set_trace()
    model.load_state_dict(ckpt['model'])

    ################################
    # Load data
    ################################

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    assert in_F == 12
    assert out_F == 30

    name = config['DATA']['train_datasets']

    dataset = create_dataset(name[0], logger, split=args.split, track_size=(in_F+out_F), track_cutoff=in_F, preprocessed=config["DATA"]["preprocessed"])

    bs = new_cfg['TRAIN']['batch_size']*6
    # bs = int(new_cfg['TRAIN']['batch_size']*6)
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=config['TRAIN']['num_workers'], shuffle=False, collate_fn=collate_batch)

    logger.info(f'Evaluating with modality: {args.modality}')

    obs_len = 12 if args.limit_obs == 0 else args.limit_obs
    logger.info(f"Evaluating with {obs_len} frames")
    evaluate_ade_fde(model, valuenet, args.split, args.modality, dataloader, bs, config, logger, exp_name, return_all=True, visualize=args.vis, limit_obs=args.limit_obs)