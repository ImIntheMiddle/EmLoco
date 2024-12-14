import sys
sys.path.append("/misc/dl00/halo/plausibl/pacer/pacer")
import argparse
from datetime import datetime
import numpy as np
import os
import random
import time
import torch

import tqdm
from progress.bar import Bar
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from dataset_jta import collate_batch, batch_process_coords, get_datasets, create_dataset
from model_jta import create_model
from utils.utils import create_logger, load_default_config, load_config, AverageMeter
from utils.metrics import MSE_LOSS_LSTM
from train_jta import prepare, save_checkpoint, adjust_learning_rate, nan_handler, dataloader_for, dataloader_for_val
from social_lstm import LSTM, LSTMPredictor, drop_distant_far, keep_valid_neigh
from gridbased_pooling import GridBasedPooling

from learning.value_pose_net import ValuePoseNet

def evaluate_loss(model, dataloader, valuenet, config, limit_obs=False):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    bar = Bar(f"EVAL", fill="#", max=len(dataloader))
    # bar = tqdm.tqdm(dataloader, desc="EVAL", total=len(dataloader))
    loss_avg = AverageMeter()
    mse_loss_avg = AverageMeter()
    value_loss_avg = AverageMeter()

    dataiter = iter(dataloader)
    model.eval()
    with torch.no_grad():
        # import pdb; pdb.set_trace()
        for i in range(len(dataloader)):
            try:
                joints, masks, padding_mask = next(dataiter)
            except StopIteration:
                break

            in_joints, _, out_joints, _, _ = batch_process_coords(joints, masks, padding_mask, config)

            primary_init_pose = joints[:, 0, 8, 3:27,:3].to(config["DEVICE"])
            primary_init_pose[..., 2] *= -1 # Flip z-axis: transformation for jta dataset
            primary_init_vel = (in_joints[:,8,0,:2] - in_joints[:,7,0,:2]) * 2.5
            primary_init_vel = primary_init_vel.to(config["DEVICE"])

            mse_loss, pred_joints = compute_loss(model, config, in_joints, out_joints, limit_obs=limit_obs)
            loss = mse_loss.clone()

            if valuenet is not None:
                pred_traj = pred_joints[:,in_F:].squeeze(2)
                pred_traj = torch.cat([torch.zeros(pred_traj.size(0), 1, 2).to(config["DEVICE"]), pred_traj], dim=1)
                pred_traj, primary_init_pose, primary_init_vel = nan_handler(pred_traj, primary_init_pose, primary_init_vel)
                pred_value, value_loss = valuenet.calc_embodied_motion_loss(pred_traj, primary_init_pose, primary_init_vel)
                value_loss *= config["TRAIN"]["valuenet_weight"]
                loss = loss + value_loss if not torch.isnan(value_loss.mean()) else loss
                if not torch.isnan(value_loss):
                    value_loss_avg.update(value_loss, len(in_joints))

            loss_avg.update(loss.item(), len(in_joints))
            mse_loss_avg.update(mse_loss.item(), len(in_joints))

            summary = [
                f"({i + 1}/{len(dataloader)})",
                f"LOSS: {loss_avg.avg:.4f}",
                f"MSE_LOSS: {mse_loss_avg.avg:.4f}",
                f"VALUE_LOSS: {value_loss_avg.avg:.4f}",
                f"T-TOT: {bar.elapsed_td}",
                f"T-ETA: {bar.eta_td:}"
            ]

            bar.suffix = " | ".join(summary)
            bar.next()
            # bar.set_postfix_str(" | ".join(summary))
        bar.finish()
    return mse_loss_avg.avg

def compute_loss(model, config, in_joints, out_joints, epoch=None, mode='val', loss_last=True, optimizer=None, limit_obs=False):
    bs, in_F, _, _ = in_joints.shape
    token_num = config["MODEL"]["token_num"]
    injoints = in_joints.to(config["DEVICE"])
    out_joints = out_joints.to(config["DEVICE"])

    if torch.isnan(in_joints).any():
        # print('Nan detected!')
        # masking nan values with zeros
        in_joints = torch.where(torch.isnan(in_joints), torch.zeros_like(in_joints), in_joints)
        # import pdb; pdb.set_trace()

    # xy_position is at the first token for every 49 tokens
    # import pdb; pdb.set_trace()
    position_tokens = torch.arange(0, in_joints.shape[2], token_num).to(config["DEVICE"])
    observed_traj = in_joints[:, :, position_tokens, :2].reshape(in_joints.shape[1], -1, 2) # (obs_len, num_tracks, 2)
    observed_traj = torch.cat([observed_traj, torch.ones(observed_traj.size(0), observed_traj.size(1), 1).to(config["DEVICE"])], dim=-1) # add visibility == 1

    batch_split = torch.arange(0, observed_traj.shape[1]+1, len(position_tokens)).to(config["DEVICE"])

    out_joints = out_joints[:, :, position_tokens, :2]
    prediction_truth = out_joints[:, :-1].reshape(out_joints.shape[1]-1, -1, 2).clone().to(config["DEVICE"])
    prediction_truth = torch.cat([prediction_truth, torch.ones(prediction_truth.size(0), prediction_truth.size(1), 1).to(config["DEVICE"])], dim=-1) # add visibility == 1
    out_target = out_joints.reshape(out_joints.shape[1], -1, 2).clone().to(config["DEVICE"]) # (pred_len, num_tracks, 2)

    rel_pred_joints, pred_joints = model(observed_traj, batch_split, prediction_truth)
    # import pdb; pdb.set_trace()
    pred_joints = pred_joints[-len(out_target):, :, :2]
    loss = MSE_LOSS_LSTM(pred_joints, out_target, batch_size=bs)
    pred_joints = pred_joints.reshape(bs, len(out_target), -1, 2) # (bs, pred_len, num_tracks, 2)
    return loss, pred_joints

def train(config, epoch, model, optimizer, logger, valuenet, dataloader_train, dataloader_val, min_val_loss, experiment_name="", dataset_name="", limit_obs=0):

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    start_time = time.time()
    dataiter = iter(dataloader_train)

    timer = {"DATA": 0, "FORWARD": 0, "BACKWARD": 0}

    loss_avg = AverageMeter()
    disc_loss_avg = AverageMeter()
    disc_acc_avg = AverageMeter()
    mse_loss_avg = AverageMeter()
    value_loss_avg = AverageMeter()

    if config["TRAIN"]["optimizer"] == "adam":
        adjust_learning_rate(optimizer, epoch, config)

    train_steps =  len(dataloader_train)

    bar = Bar(f"TRAIN {epoch}/{config['TRAIN']['epochs'] - 1}", fill="#", max=train_steps)

    for i in range(train_steps):
        model.train()
        optimizer.zero_grad()

        ################################
        # Load a batch of data
        ################################
        start = time.time()

        try:
            joints, masks, padding_mask = next(dataiter)

        except StopIteration:
            dataiter = iter(dataloader_train)
            joints, masks, padding_mask = next(dataiter)

        in_joints, _, out_joints, _, _ = batch_process_coords(joints, masks, padding_mask, config, training=True)
        primary_init_pose = joints[:, 0, 8, 3:27, :3].clone().to(config["DEVICE"])
        primary_init_pose[..., 2] *= -1 # flip z axis: transformation for jta dataset
        primary_init_vel = (in_joints[:,8,0,:2] - in_joints[:,7,0,:2]) * 2.5
        primary_init_vel = primary_init_vel.clone().to(config["DEVICE"])

        timer["DATA"] = time.time() - start

        ################################
        # Forward Pass
        ################################
        start = time.time()
        mse_loss, pred_joints = compute_loss(model, config, in_joints, out_joints, epoch=epoch, mode='train', optimizer=None, limit_obs=limit_obs)
        loss = mse_loss.clone()
        # mse_loss can be backpropagated by loss, so we can free the memory of mse_loss here
        mse_loss = mse_loss.detach().item() # detach the tensor to free the memory
        mse_loss_avg.update(mse_loss, len(joints))
        if valuenet is not None:
            pred_traj = pred_joints[:,in_F:].squeeze()
            pred_traj = torch.cat([torch.zeros(pred_traj.size(0), 1, 2).to(config["DEVICE"]), pred_traj], dim=1)

            # import pdb; pdb.set_trace()
            pred_traj, primary_init_pose, primary_init_vel = nan_handler(pred_traj, primary_init_pose, primary_init_vel)
            pred_value, value_loss = valuenet.calc_embodied_motion_loss(pred_traj, primary_init_pose, primary_init_vel)

            value_loss  *= config["TRAIN"]["valuenet_weight"]
            loss = loss + value_loss if not torch.isnan(value_loss) else loss

        timer["FORWARD"] = time.time() - start

        ################################
        # Backward Pass + Optimization
        ################################
        start = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN"]["max_grad_norm"])
        optimizer.step()

        timer["BACKWARD"] = time.time() - start

        ################################
        # Logging
        ################################

        loss_avg.update(loss.item(), len(joints))
        if valuenet is not None:
            if not torch.isnan(value_loss):
                value_loss_avg.update(value_loss.mean().item(), len(joints))

        summary = [
            f"{str(epoch).zfill(3)} ({i + 1}/{train_steps})",
            f"LOSS: {loss_avg.avg:.4f}",
            f"MSE_LOSS: {mse_loss_avg.avg:.4f}",
            f"VALUE_LOSS: {value_loss_avg.avg:.4f}",
            f"T-TOT: {bar.elapsed_td}",
            f"T-ETA: {bar.eta_td:}"
        ]

        # for key, val in timer.items():
        #      summary.append(f"{key}: {val:.2f}")

        bar.suffix = " | ".join(summary)
        bar.next()

        if config['dry_run']:
            break

    bar.finish()

    ################################
    # Tensorboard logs
    ################################

    # global_step += train_steps
    # writer_train.add_scalar("loss", loss_avg.avg, epoch)

    val_loss = evaluate_loss(model, dataloader_val, valuenet, config, limit_obs=limit_obs)
    # writer_valid.add_scalar("loss", val_loss, epoch)

    val_ade = val_loss/100
    logger.info(f"Epoch {epoch} | Train Loss: {loss_avg.avg:.3f} | Val ADE: {val_ade:.3f}")

    logger.info(f'time for training: {time.time()-start_time}')
    print('epoch ', epoch, ' finished!')

    # if not config['dry_run']:
    #     save_checkpoint(model, optimizer, epoch, config, 'checkpoint.pth.tar', logger)
    return val_ade

def main(config, logger, valuenet, dataloader_train, dataloader_val, experiment_name, dataset_name, limit_obs):
    ################################
    # Create model, loss, optimizer
    ################################
    pool = GridBasedPooling(
            type_=config["MODEL"]["pool"],
            hidden_dim=config["MODEL"]["hidden-dim"],
            cell_side=config["MODEL"]["cell_side"],
            n=config["MODEL"]["n"],
            front=config["MODEL"]["front"],
            out_dim=config["MODEL"]["out_dim"],
            embedding_arch=config["MODEL"]["embedding_arch"],
            constant=config["MODEL"]["constant"],
            pretrained_pool_encoder=None,
            norm=config["MODEL"]["norm"],
            layer_dims=config["MODEL"]["layer_dims"],
            latent_dim=config["MODEL"]["latent_dim"],
        )

    model = LSTM(
        pool=pool,
        embedding_dim=config["MODEL"]["coordinate-embedding-dim"],
        hidden_dim=config["MODEL"]["hidden-dim"],
        goal_flag=config["MODEL"]["goal_flag"],
        goal_dim=config["MODEL"]["goal_dim"],
    )
    model = torch.nn.DataParallel(model).to(config["DEVICE"])

    if config["RESUME"]!=-1:
        if cfg["MODEL"]["checkpoint"] == "":
            logger.info("Using the latest checkpoint.")
            last_epoch = cfg["RESUME"]
            if os.path.exists(os.path.join(cfg["OUTPUT"]['ckpt_dir'], f"checkpoint.pth.tar")):
                cfg["MODEL"]["checkpoint"] = os.path.join(cfg["OUTPUT"]['ckpt_dir'], f"checkpoint.pth.tar")
            elif os.path.exists(os.path.join(cfg["OUTPUT"]['ckpt_dir'], "best_val_checkpoint.pth.tar")):
                cfg["MODEL"]["checkpoint"] = os.path.join(cfg["OUTPUT"]['ckpt_dir'], "best_val_checkpoint.pth.tar")
            else:
                logger.info("No checkpoint found.")
                raise ValueError("No checkpoint found.")
        logger.info(f"Loading checkpoint from {config['MODEL']['checkpoint']}")
        checkpoint = torch.load(config["MODEL"]["checkpoint"])
        model.load_state_dict(checkpoint["model"])
    else:
        logger.info("Training from scratch.")

    optimizer = torch.optim.Adam(model.module.parameters(), lr=config['TRAIN']['lr'], weight_decay=config['TRAIN']['weight_decay'])

    num_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} parameters.")

    writer_name = experiment_name + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    writer_train = SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_TRAIN"))
    writer_valid =  SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_VALID"))

    ################################
    # Begin Training
    ################################
    global_step = 0
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    min_val_loss = 1e6 if not config["RESUME"] else evaluate_loss(model, dataloader_val, valuenet, config)/100
    logger.info(f"Initial validation loss: {min_val_loss:.3f}")
    if valuenet is not None:
        logger.info(f'Using Value Loss weight: {float(config["TRAIN"]["valuenet_weight"]):.3f}')

    for epoch in range(config['RESUME']+1, config["TRAIN"]["epochs"]):
        val_ade = train(config, epoch, model, optimizer, logger, valuenet, dataloader_train, dataloader_val, min_val_loss, experiment_name=experiment_name, dataset_name=dataset, limit_obs=limit_obs)
        if val_ade < min_val_loss:
            min_val_loss = val_ade
            print('------------------------------BEST MODEL UPDATED------------------------------')
            logger.info(f'Best ADE: {val_ade}')
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, config, 'best_val'+'_checkpoint.pth.tar', logger)
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, config, 'checkpoint.pth.tar', logger)
        if config['dry_run']:
            break

    return min_val_loss, best_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name. Otherwise will use timestamp")
    parser.add_argument("--cfg", type=str, default="configs/jta_social-lstm.yaml", help="Config name. Otherwise will use default config")
    parser.add_argument('--dry-run', action='store_true', help="Run just one iteration")
    parser.add_argument('--valueloss_w', type=float, default=0, help="Use value loss")
    parser.add_argument('--resume', type=int, default=-1, help="Resume training from a checkpoint")
    parser.add_argument('--not_pose', action='store_true', help="Not using pose input for value function")
    parser.add_argument('--not_vel', action='store_true', help="Not using velocity input for value function")
    parser.add_argument('--limit_obs', type=int, default=0, help="Limit the number of observations")
    parser.add_argument('--value_path', type=str, default="", help="Path to the value network checkpoint")
    parser.add_argument('--value_dir', type=str, default="/home/halo/plausibl/pacer/output/exp/pacer/", help="Directory to the value network checkpoint")
    args = parser.parse_args()

    if args.cfg != "":
        cfg = load_config(args.cfg, exp_name=args.exp_name, dataset_name="JTA")
    else:
        cfg = load_default_config()

    cfg['dry_run'] = args.dry_run
    cfg['RESUME'] = args.resume
    cfg["USE_VALUELOSS"] = args.valueloss_w > 0
    cfg["USE_POSE"] = not args.not_pose
    cfg["USE_VELOCITY"] = not args.not_vel
    cfg["TRAIN"]["valuenet_weight"] = args.valueloss_w

    if args.value_path != "":
        cfg["MODEL"]["valuenet_checkpoint"] = os.path.join(args.value_dir, args.value_path)
    else:
        cfg["MODEL"]["valuenet_checkpoint"] = os.path.join(args.value_dir, cfg["MODEL"]["valuenet_checkpoint"])

    if torch.cuda.is_available():
        cfg["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
    else:
        cfg["DEVICE"] = "cpu"

    random.seed(cfg['SEED'])
    torch.manual_seed(cfg['SEED'])
    np.random.seed(cfg['SEED'])

    dataset = cfg["DATA"]["train_datasets"]

    logger = create_logger(cfg["OUTPUT"]["log_dir"])
    logger.info("Initializing with config:")
    logger.info(cfg)

    valuenet, dataloader_train, dataloader_val = prepare(cfg, logger, args.exp_name, dataset)

    min_val_loss, best_epoch = main(cfg, logger, valuenet, dataloader_train, dataloader_val, experiment_name=args.exp_name, dataset_name=dataset, limit_obs=args.limit_obs)
    logger.info(f"Best validation loss: {min_val_loss:.3f} at epoch {best_epoch}")
    logger.info("All done.")