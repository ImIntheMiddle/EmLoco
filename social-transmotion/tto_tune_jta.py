import sys
sys.path.append("/misc/dl00/halo/plausibl/pacer/pacer")
import os
import argparse
import torch
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import optuna
from progress.bar import Bar
from torch.utils.data import DataLoader

from dataset_jta import batch_process_coords, create_dataset, collate_batch
from model_jta import create_model
from train_jta import compute_loss, nan_handler
from utils.utils import create_logger
from utils.utils import create_logger, load_default_config, load_config, AverageMeter
from utils.metrics import MSE_LOSS

from learning.value_pose_net import ValuePoseNet

def optuna_opt(model, dataloader, valuenet, config, logger, exp_name):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    bar = Bar(f"EVAL", fill="#", max=len(dataloader))
    # bar = tqdm.tqdm(dataloader, desc="EVAL", dynamic_ncols=True)
    mse_loss_avg = AverageMeter()
    value_loss_avg = AverageMeter()

    dataiter = iter(dataloader)
    model.eval()
    traj_pool = torch.empty(0, 13, 2).to(config["DEVICE"])
    pose_pool = torch.empty(0, 24, 3).to(config["DEVICE"])
    vel_pool = torch.empty(0, 2).to(config["DEVICE"])
    outjoint_pool = torch.empty(0, 12, 2).to(config["DEVICE"])

    for i in range(len(dataloader)):
        with torch.no_grad():
            try:
                joints, masks, padding_mask = next(dataiter)
            except StopIteration:
                break

            in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config)
            padding_mask = padding_mask.to(config["DEVICE"])

            primary_init_pose = joints[:, 0, 8, 3:27, :3].to(config["DEVICE"])
            primary_init_pose[..., 2] *= -1 # Flip z-axis: transformation for jta dataset
            primary_init_pose.requires_grad = False

            primary_init_vel = (in_joints[:,8,0,:2] - in_joints[:,7,0,:2]) * 2.5
            primary_init_vel = primary_init_vel.to(config["DEVICE"])
            primary_init_vel.requires_grad = False

            mse_loss, pred_joints = compute_loss(model, config, in_joints, out_joints, in_masks, out_masks, padding_mask)
            mse_loss_avg.update(mse_loss.item(), len(in_joints))

            pred_traj = pred_joints[:,in_F:].squeeze(2)
            pred_traj = torch.cat([torch.zeros(pred_traj.size(0), 1, 2).to(config["DEVICE"]), pred_traj], dim=1)
            pred_traj, primary_init_pose, primary_init_vel = nan_handler(pred_traj, primary_init_pose, primary_init_vel)
            pred_traj.requires_grad = True

            traj_pool = torch.cat([traj_pool, pred_traj], dim=0)
            pose_pool = torch.cat([pose_pool, primary_init_pose], dim=0)
            vel_pool = torch.cat([vel_pool, primary_init_vel], dim=0)
            outjoint_pool = torch.cat([outjoint_pool, out_joints[:,:,0,:2]], dim=0)

    mse_loss = torch.nn.functional.mse_loss(traj_pool[:,:12], outjoint_pool)
    print(f"Initial MSE Loss: {mse_loss}")
    del dataiter
    del dataloader
    study = optuna.create_study(direction='minimize')
    study.optimize(valuenet_opt(traj_pool, pose_pool, vel_pool, outjoint_pool, valuenet, config), n_trials=200)
    print("Number of finished trials: {}".format(len(study.trials)))
    print(f"Best MSE: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_params}")

def valuenet_opt(traj_pool, pose_pool, vel_pool, outjoint_pool, valuenet, config):
    smooth_loss = torch.nn.SmoothL1Loss()

    def objective(trial):
        def closure():
            # import pdb; pdb.set_trace()
            optimizer.zero_grad()
            trajs = parameters.clone().reshape(-1, 13, 2)
            pred_value, value_loss = valuenet.calc_embodied_motion_loss(trajs, pose_pool.clone(), vel_pool.clone())
            loss = alpha * value_loss.mean() + (1 - alpha) * smooth_loss(trajs, original_traj_pool.clone())
            loss.backward()
            assert loss.requires_grad, "Loss should require grad"
            assert parameters.grad is not None, "Loss should have grad"
            bar.set_description(f"Loss: {loss.mean().item():.5f}")
            return loss

        valuenet.eval()
        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        lr = trial.suggest_float('lr', 1e-3, 1e-1)
        # steps = trial.suggest_int('steps', 2, 3)
        history_size = trial.suggest_int('history_size', 1, 3000)
        max_iter = trial.suggest_int('max_iter', 1, 3000)
        # line_search_fn = trial.suggest_categorical('line_search_fn', ['strong_wolfe', None])

        pose_pool.requires_grad = False
        vel_pool.requires_grad = False
        original_traj_pool = traj_pool.clone().detach()
        original_traj_pool.requires_grad = False
        parameters = original_traj_pool.clone().detach().flatten()
        parameters.requires_grad = True
        optimizer = torch.optim.LBFGS([parameters],
                    lr=lr,
                    history_size=history_size,
                    max_iter=max_iter,
                    line_search_fn=None)

        bar = tqdm.tqdm(range(2), leave=False, dynamic_ncols=True)
        for i in bar:
            optimizer.step(closure)

        optimized_params = optimizer.param_groups[0]['params'][0].detach().reshape(-1, 13, 2)
        # assert (optimized_params - original_traj_pool).sum() != 0, "Optimization failed"
        mse_loss = torch.nn.functional.mse_loss(optimized_params[:,:12], outjoint_pool)
        del optimizer
        del parameters
        return mse_loss.item()
    return objective

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,  help="checkpoint path")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ################################
    # Load checkpoint
    ################################
    ckpt_name = f'./experiments/JTA/{args.exp_name}/checkpoints/best_val_checkpoint.pth.tar'

    logger = create_logger('')
    logger.info(f'Loading checkpoint from {ckpt_name}')
    ckpt = torch.load(ckpt_name, map_location = torch.device('cpu'))
    config = ckpt['config']
    new_cfg = load_config("configs/jta_all_visual_cues.yaml")
    exp_name = args.exp_name
    # print(exp_name)

    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed(0)
    else:
        config["DEVICE"] = "cpu"

    assert config["MODEL"]["valuenet_checkpoint"] != "", "Please provide a checkpoint for valuenet"
    valuenet = ValuePoseNet(use_pose=True, use_vel=True, hide_toe=True, normalize=True)
    valuenet_ckpt = config["MODEL"]["valuenet_checkpoint"]
    logger.info(f"Loading checkpoint from {valuenet_ckpt}")
    valuenet.load_state_dict(torch.load(valuenet_ckpt, map_location = torch.device('cpu')))
    valuenet.to(config["DEVICE"])

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

    dataset = create_dataset(name[0], logger, split='val', track_size=(in_F+out_F), track_cutoff=in_F, preprocessed=config['DATA']['preprocessed'])

    bs = new_cfg['TRAIN']['batch_size']*4
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=config['TRAIN']['num_workers'], shuffle=False, collate_fn=collate_batch)

    optuna_opt(model, dataloader, valuenet, config, logger, exp_name)