import sys
sys.path.append("/misc/dl00/halo/plausibl/pacer/pacer")
import argparse
from datetime import datetime
import numpy as np
import os
import random
import time
import torch
import optuna

import tqdm
from progress.bar import Bar
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from dataset_jrdb import collate_batch, batch_process_coords, get_datasets, create_dataset
from model_jrdb import create_model
from utils.utils import create_logger, load_default_config, load_config, AverageMeter
from utils.metrics import MSE_LOSS
from train_jta import nan_handler, adjust_learning_rate, compute_loss, save_checkpoint, dataloader_for, dataloader_for_val
from evaluate_jrdb import get_action_label

from learning.value_pose_net import ValuePoseNet

def evaluate_loss(model, dataloader, valuenet, config, limit_obs=False, modality_selection='traj+all'):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    bar = Bar(f"EVAL", fill="#", max=len(dataloader))
    loss_avg = AverageMeter()
    mse_loss_avg = AverageMeter()
    value_loss_avg = AverageMeter()

    dataiter = iter(dataloader)
    model.eval()
    with torch.no_grad():
        for i in range(len(dataloader)):
            try:
                joints, masks, padding_mask, idxs_list = next(dataiter)
            except StopIteration:
                break

            in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config, modality_selection=modality_selection)
            padding_mask = padding_mask.to(config["DEVICE"])

            primary_init_pose = in_joints[:,8,2:26,:3].to(config["DEVICE"])
            # primary_init_pose = in_joints[:, 8, 1:20, :3].clone().to(config["DEVICE"])
            # zeros_pose = torch.zeros(primary_init_pose.size(0), primary_init_pose.size(1)+5, 3).to(config["DEVICE"])
            # needed_joints = list(range(0, 4)) + list(range(5, 8)) + list(range(12, 24))
            # zeros_pose[:,needed_joints] = primary_init_pose
            # primary_init_pose = zeros_pose

            primary_init_vel = (in_joints[:,8,0,:2] - in_joints[:,7,0,:2]) * 2.5
            primary_init_vel = primary_init_vel.to(config["DEVICE"])

            mse_loss, pred_joints = compute_loss(model, config, in_joints, out_joints, in_masks, out_masks, padding_mask, limit_obs=limit_obs)
            loss = mse_loss.clone()
            mse_loss = mse_loss.detach().item()
            mse_loss_avg.update(mse_loss, len(in_joints))

            if valuenet is not None:
                if config["MULTI_MODAL"]:
                    pred_trajs = pred_joints[:,in_F:]
                    pred_trajs = torch.cat([torch.zeros(pred_trajs.size(0), 1, pred_trajs.size(2), 2).to(config["DEVICE"]), pred_trajs], dim=1)
                    pred_trajs, primary_init_pose, primary_init_vel = nan_handler(pred_trajs, primary_init_pose, primary_init_vel)
                    value_losses = 0
                    for i in range(pred_trajs.size(2)):
                        pred_value, value_loss = valuenet.calc_embodied_motion_loss(pred_trajs[:,:,i], primary_init_pose, primary_init_vel)
                        value_losses += value_loss
                    value_losses *= config["TRAIN"]["valuenet_weight"]
                    value_loss = value_losses / pred_trajs.size(2) if pred_trajs.size(2)>0 else 0
                else:
                    pred_traj = pred_joints[:,in_F:].squeeze(2)
                    pred_traj = torch.cat([torch.zeros(pred_traj.size(0), 1, 2).to(config["DEVICE"]), pred_traj], dim=1)
                    pred_traj, primary_init_pose, primary_init_vel = nan_handler(pred_traj, primary_init_pose, primary_init_vel)
                    pred_value, value_loss = valuenet.calc_embodied_motion_loss(pred_traj, primary_init_pose, primary_init_vel)
                    value_loss *= config["TRAIN"]["valuenet_weight"]
                # loss = (1-config["TRAIN"]["valuenet_weight"]) * loss + value_loss.mean() if not torch.isnan(value_loss.mean()) else loss

                # import pdb; pdb.set_trace()
                if len(value_loss[~torch.isnan(value_loss)]) > 0:
                    value_loss_avg.update(value_loss[~torch.isnan(value_loss)].mean().item(), len(value_loss[~torch.isnan(value_loss)]))
                    loss = loss + value_loss[~torch.isnan(value_loss)].mean()

            loss_avg.update(loss.item(), len(in_joints))

            summary = [
                f"({i + 1}/{len(dataloader)})",
                f"LOSS: {loss_avg.avg:.4f}",
                f"MSE_LOSS: {(mse_loss_avg.avg):.4f}",
                f"VALUE_LOSS: {value_loss_avg.avg:.4f}",
                f"T-TOT: {bar.elapsed_td}",
                f"T-ETA: {bar.eta_td:}"
            ]
            bar.suffix = " | ".join(summary)
            bar.next()

        bar.finish()
    return mse_loss_avg.avg

def dataloader_for(dataset, config, **kwargs):
    return DataLoader(dataset,
                      batch_size=config['TRAIN']['batch_size'],
                      num_workers=config['TRAIN']['num_workers'],
                      collate_fn=collate_batch,
                      **kwargs)

def dataloader_for_val(dataset, config, **kwargs):
    return DataLoader(dataset,
                      batch_size=config['TRAIN']['batch_size']*3,
                      num_workers=config['TRAIN']['num_workers'],
                      collate_fn=collate_batch,
                      **kwargs)

def prepare(config, logger, experiment_name, dataset_name):
    ################################
    # Create valuenet
    ################################

    if config["USE_VALUELOSS"]:
        valuenet = ValuePoseNet(use_pose=config["USE_POSE"], use_vel=config["USE_VELOCITY"])
        assert config["MODEL"]["valuenet_checkpoint"] != "", "Please provide a checkpoint for the valuenet!"
        logger.info(f"Loading checkpoint from {config['MODEL']['valuenet_checkpoint']}")
        valuenet.load_state_dict(torch.load(config["MODEL"]["valuenet_checkpoint"]))
        valuenet.eval()
        valuenet.requires_grad_ = False
        valuenet.to(config["DEVICE"])
    else:
        valuenet = None

    ################################
    # Load data
    ################################

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    dataset_train = ConcatDataset(get_datasets(config['DATA']['train_datasets'], config, logger))
    dataloader_train = dataloader_for(dataset_train, config, shuffle=True, pin_memory=True)
    logger.info(f"Training on a total of {len(dataset_train)} annotations.")
    logger.info(f"Created training dataset!")

    dataset_val = create_dataset(config['DATA']['train_datasets'][0], logger, split="val", track_size=(in_F+out_F), track_cutoff=in_F, preprocessed=config['DATA']['preprocessed'])
    dataloader_val = dataloader_for_val(dataset_val, config, shuffle=True, pin_memory=True)
    logger.info(f"Created validation dataset!")

    return valuenet, dataloader_train, dataloader_val

def train(config, epoch, model, optimizer, logger, valuenet, dataloader_train, dataloader_val, min_val_loss, experiment_name="", dataset_name="", limit_obs=False, modality_selection='traj+all'):

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
            joints, masks, padding_mask, idxs_list = next(dataiter)

        except StopIteration:
            dataiter = iter(dataloader_train)
            joints, masks, padding_mask, idxs_list = next(dataiter)

        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config, training=True, modality_selection=modality_selection)
        padding_mask = padding_mask.to(config["DEVICE"])
        primary_init_pose = in_joints[:, 8, 2:26, :3].clone().to(config["DEVICE"])
        # primary_init_pose = in_joints[:, 8, 1:20, :3].detach().clone().to(config["DEVICE"])
        # primary_init_pose[..., 2] *= -1 # flip z axis: transformation for jta dataset
        # zeros_pose = torch.zeros(primary_init_pose.size(0), primary_init_pose.size(1)+5, 3).to(config["DEVICE"])
        # needed_joints = list(range(0, 4)) + list(range(5, 8)) + list(range(12, 24))
        # zeros_pose[:,needed_joints] = primary_init_pose
        # primary_init_pose = zeros_pose

        primary_init_vel = (in_joints[:,8,0,:2] - in_joints[:,7,0,:2]) * 2.5
        primary_init_vel = primary_init_vel.clone().to(config["DEVICE"])

        timer["DATA"] = time.time() - start

        ################################
        # Forward Pass
        ################################
        start = time.time()
        mse_loss, pred_joints = compute_loss(model, config, in_joints, out_joints, in_masks, out_masks, padding_mask, epoch=epoch, mode='train', optimizer=None, limit_obs=limit_obs)
        if config["VAL_LOSS_ONLY"]:
            mse_loss *= 0
        loss = mse_loss.clone()
        if valuenet is not None:
            if config["MULTI_MODAL"]:
                pred_trajs = pred_joints[:,in_F:]
                pred_trajs = torch.cat([torch.zeros(pred_trajs.size(0), 1, pred_trajs.size(2), 2).to(config["DEVICE"]), pred_trajs], dim=1)
                value_losses = 0
                pred_trajs, primary_init_pose, primary_init_vel = nan_handler(pred_trajs, primary_init_pose, primary_init_vel)
                for i in range(pred_trajs.size(2)):
                    pred_value, value_loss = valuenet.calc_embodied_motion_loss(pred_trajs[:,:,i], primary_init_pose, primary_init_vel)
                    value_losses += value_loss
                value_losses *= config["TRAIN"]["valuenet_weight"]
                value_loss = value_losses / pred_trajs.size(2) if pred_trajs.size(2) > 0 else 0
            else:
                pred_traj = pred_joints[:,in_F:].squeeze(2)
                pred_traj = torch.cat([torch.zeros(pred_traj.size(0), 1, 2).to(config["DEVICE"]), pred_traj], dim=1)

                pred_traj, primary_init_pose, primary_init_vel = nan_handler(pred_traj, primary_init_pose, primary_init_vel)
                pred_value, value_loss = valuenet.calc_embodied_motion_loss(pred_traj, primary_init_pose, primary_init_vel)

                # if not torch.isnan(value_loss.mean()):
                #     before_pose = in_joints[0, :8, 2:26, :3].clone().detach().cpu()
                #     valuenet.visualize_pose(primary_init_pose[0].clone().detach().unsqueeze(0).cpu(), in_joints[0,:,0,:2].clone().detach().unsqueeze(0).cpu().numpy(), torch.tensor(np.concatenate((np.zeros((1,2)),out_joints[0,:,0,:2].clone().detach().cpu()),axis=0)).unsqueeze(0).cpu(), before_pose=before_pose)

                value_loss  *= config["TRAIN"]["valuenet_weight"]

                if len(value_loss[~torch.isnan(value_loss)]) > 0:
                    value_loss_avg.update(value_loss[~torch.isnan(value_loss)].mean().item(), len(value_loss[~torch.isnan(value_loss)]))
                    loss = loss + value_loss[~torch.isnan(value_loss)].mean()

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
        mse_loss_avg.update(mse_loss.item(), len(joints))
        if (valuenet is not None) and len((~torch.isnan(value_loss)).shape) > 0:
            # update with non-nan values
            value_loss_avg.update(value_loss[~torch.isnan(value_loss)].mean().item(), len(value_loss[~torch.isnan(value_loss)]))

        summary = [
            f"{str(epoch).zfill(3)} ({i + 1}/{train_steps})",
            f"LOSS: {loss_avg.avg:.3f}",
            f"MSE: {(mse_loss_avg.avg):.3f}",
            f"VALLOSS: {value_loss_avg.avg:.3f}",
            f"TOT: {bar.elapsed_td}",
            f"ETA: {bar.eta_td:}"
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
    # writer_valid.add_scalar("loss", val_loss, epoch)

    val_loss = evaluate_loss(model, dataloader_val, valuenet, config, limit_obs=limit_obs, modality_selection=modality_selection)

    val_ade = val_loss/100
    logger.info(f"Epoch {epoch} | Train Loss: {loss_avg.avg:.3f} | Val ADE: {val_ade:.3f}")

    logger.info(f'time for training: {time.time()-start_time}')
    print('epoch ', epoch, ' finished!')

    # if not config['dry_run']:
        # save_checkpoint(model, optimizer, epoch, config, 'checkpoint.pth.tar', logger)
    return val_ade

def main(cfg, logger, valuenet, dataloader_train, dataloader_val, experiment_name, dataset_name, limit_obs):
    ################################
    # Create model, loss, optimizer
    ################################

    model = create_model(cfg, logger)
    model = torch.nn.DataParallel(model).to(cfg["DEVICE"])

    if cfg["RESUME"]!=-1:
        if cfg["MODEL"]["checkpoint"] == "":
            logger.info("Using the latest checkpoint.")
            last_epoch = cfg["RESUME"]
            if os.path.exists(os.path.join(cfg["OUTPUT"]['ckpt_dir'], f"checkpoint_{last_epoch}epoch.pth.tar")):
                cfg["MODEL"]["checkpoint"] = os.path.join(cfg["OUTPUT"]['ckpt_dir'], f"checkpoint_{last_epoch}epoch.pth.tar")
            elif os.path.exists(os.path.join(cfg["OUTPUT"]['ckpt_dir'], "best_val_checkpoint.pth.tar")):
                cfg["MODEL"]["checkpoint"] = os.path.join(cfg["OUTPUT"]['ckpt_dir'], "best_val_checkpoint.pth.tar")
            else:
                logger.info("No checkpoint found.")
                raise ValueError("No checkpoint found.")
        logger.info(f"Loading checkpoint from {cfg['MODEL']['checkpoint']}")
        checkpoint = torch.load(cfg["MODEL"]["checkpoint"])
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        logger.info("Training from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['TRAIN']['lr'])

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} parameters.")

    writer_name = experiment_name + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    writer_train = SummaryWriter(os.path.join(cfg["OUTPUT"]["runs_dir"], f"{writer_name}_TRAIN"))
    writer_valid =  SummaryWriter(os.path.join(cfg["OUTPUT"]["runs_dir"], f"{writer_name}_VALID"))

    ################################
    # Begin Training
    ################################
    min_val_loss = 1e6 if not cfg["RESUME"] else evaluate_loss(model, dataloader_val, valuenet, cfg)/100
    logger.info(f"Initial validation loss: {min_val_loss:.3f}")
    if valuenet is not None:
        logger.info(f'Using Value Loss weight: {float(cfg["TRAIN"]["valuenet_weight"]):.3f}')

    for epoch in range(cfg['RESUME']+1, cfg["TRAIN"]["epochs"]):
        val_ade = train(cfg, epoch, model, optimizer, logger, valuenet, dataloader_train, dataloader_val, min_val_loss, experiment_name=experiment_name, dataset_name=dataset, limit_obs=False, modality_selection=cfg["MODALITY"])
        if val_ade < min_val_loss:
            min_val_loss = val_ade
            print('------------------------------BEST MODEL UPDATED------------------------------')
            logger.info(f'Best ADE: {val_ade}')
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, cfg, 'best_val'+'_checkpoint.pth.tar', logger)
            save_checkpoint(model, optimizer, epoch, cfg, 'best_val'+f'_checkpoint_{epoch}epoch.pth.tar', logger)
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, cfg, f'checkpoint_{epoch}epoch.pth.tar', logger)
        if cfg['dry_run']:
            break

    return min_val_loss, best_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name. Otherwise will use timestamp")
    parser.add_argument("--cfg", type=str, default="configs/jrdb_all_visual_cues.yaml", help="Config name. Otherwise will use default config")
    parser.add_argument('--dry-run', action='store_true', help="Run just one iteration")
    parser.add_argument('--valueloss_w', type=float, default=0, help="Weight for value loss")
    parser.add_argument('--resume', type=int, default=-1, help="Resume training from a checkpoint")
    parser.add_argument('--not_pose', action='store_true', help="Not using pose input for value function")
    parser.add_argument('--not_vel', action='store_true', help="Not using velocity input for value function")
    parser.add_argument('--limit_obs', type=int, default=0, help="Limit the number of past observations")
    parser.add_argument('--frame_mask', type=bool, default=True, help="Use frame masking")
    parser.add_argument('--value_path', type=str, default="", help="Path to the value network checkpoint. For example, 'valuenet_realpath_JTA+JRDB_valuenet_00025000.pth'. ")
    parser.add_argument('--value_dir', type=str, default="/home/halo/plausibl/pacer/output/exp/pacer/", help="Directory to the value network checkpoint")
    parser.add_argument('--noisy_traj', action='store_true', help="Add noise to the trajectory to mimic real data")
    parser.add_argument('--use_hypara_best', action='store_true', help="Use the best hyperparameters")
    parser.add_argument('--multi_modal', action='store_true', help="Use multimodal model")
    parser.add_argument('--valueloss_only', action='store_true', help="Train with the value loss only")
    parser.add_argument("--modality", type=str, default="traj+all", help="available modality combination from['traj','traj+all', 'traj+2dbox','traj+3dpose']")
    args = parser.parse_args()

    if args.cfg != "":
        cfg = load_config(args.cfg, exp_name=args.exp_name, dataset_name="JRDB")
    else:
        cfg = load_default_config()

    cfg['dry_run'] = args.dry_run
    cfg['RESUME'] = args.resume
    cfg["USE_VALUELOSS"] = args.valueloss_w > 0 or args.use_hypara_best
    cfg["USE_POSE"] = not args.not_pose
    cfg["USE_VELOCITY"] = not args.not_vel
    cfg["USE_FRAME_MASK"] = args.frame_mask
    cfg["NOISY_TRAJ"] = args.noisy_traj
    cfg["MULTI_MODAL"] = args.multi_modal
    cfg["VAL_LOSS_ONLY"] = args.valueloss_only
    cfg["MODALITY"] = args.modality
    if args.use_hypara_best:
        study = optuna.load_study(study_name="my_study", storage=f"sqlite:///./experiments/JRDB/hypara_dist_v4/study.db",)
        cfg["TRAIN"]["valuenet_weight"] = study.best_params["valuenet_weight"]
    else:
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

    # if not os.path.exists(f"experiments/{args.exp_name}"):
    #     os.makedirs(f"experiments/{args.exp_name}")

    min_val_loss, best_epoch = main(cfg, logger, valuenet, dataloader_train, dataloader_val, experiment_name=args.exp_name, dataset_name=dataset, limit_obs=args.limit_obs)

    logger.info(f"Best validation loss: {min_val_loss:.3f} at epoch {best_epoch}")
    logger.info("All done.")