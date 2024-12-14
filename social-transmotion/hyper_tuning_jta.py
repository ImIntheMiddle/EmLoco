import sys
sys.path.append("/misc/dl00/halo/plausibl/pacer/pacer")
import os
import argparse
import random
from datetime import datetime
import numpy as np
import torch
import optuna
import sqlite3
from torch.utils.tensorboard import SummaryWriter
from learning.value_pose_net import ValuePoseNet

from torch.utils.data import DataLoader, ConcatDataset
from utils.utils import create_logger, load_default_config, load_config
from train_jta import dataloader_for, dataloader_for_val, save_checkpoint
from dataset_jta import get_datasets, create_dataset
from model_jta import create_model
from train_jta import train, evaluate_loss

def prepare(config, logger, experiment_name, dataset_name):

    ################################
    # Create valuenet
    ################################

    valuenet = ValuePoseNet(use_pose=config["USE_POSE"], use_vel=config["USE_VELOCITY"])
    assert config["MODEL"]["valuenet_checkpoint"] != "", "Please provide a checkpoint for the valuenet!"
    config["MODEL"]["valuenet_checkpoint"] = os.path.join(args.value_dir, config["MODEL"]["valuenet_checkpoint"])
    logger.info(f"Loading checkpoint from {config['MODEL']['valuenet_checkpoint']}")
    valuenet.load_state_dict(torch.load(config["MODEL"]["valuenet_checkpoint"]))
    valuenet.eval()
    valuenet.to(config["DEVICE"])

    ################################
    # Load data
    ################################

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    dataset_train = ConcatDataset(get_datasets(config['DATA']['train_datasets'], config, logger))
    dataloader_train = dataloader_for(dataset_train, config, shuffle=True, pin_memory=True)
    logger.info(f"Training on a total of {len(dataset_train)} annotations.")
    print(f"Created training dataset!")

    dataset_val = create_dataset(config['DATA']['train_datasets'][0], logger, split="val", track_size=(in_F+out_F), track_cutoff=in_F, preprocessed=config['DATA']['preprocessed'])
    dataloader_val = dataloader_for(dataset_val, config, shuffle=True, pin_memory=True)
    print(f"Created validation dataset!")

    return valuenet, dataloader_train, dataloader_val

def valuenet_opt(cfg, logger, exp_name, valuenet, dataloader_train, dataloader_val, study):
    def objective(trial):
        try:
            best_score = study.best_value
        except ValueError:
            best_score = 1e6
        finally:
            # valuenet_weight = trial.suggest_float("valuenet_weight", 0.7, 1)
            valuenet_weight = trial.suggest_discrete_uniform("valuenet_weight", 0.8, 0.98, 0.02)
            cfg["TRAIN"]["valuenet_weight"] = valuenet_weight
            # cfg["NOISY_TRAJ"] = trial.suggest_float("noisy_traj", 0.0, 2.0)

            ################################
            # Create model, loss, optimizer
            ################################

            model = create_model(cfg, logger)
            model = torch.nn.DataParallel(model).to(cfg["DEVICE"])

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['TRAIN']['lr'])

            num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model has {num_parameters} parameters.")

            writer_name = exp_name + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            writer_train = SummaryWriter(os.path.join(cfg["OUTPUT"]["runs_dir"], f"{writer_name}_TRAIN"))
            writer_valid =  SummaryWriter(os.path.join(cfg["OUTPUT"]["runs_dir"], f"{writer_name}_VALID"))

            ################################
            # Begin Training
            ################################
            min_val_loss = 1e6 if not cfg["RESUME"] else evaluate_loss(model, dataloader_val, valuenet, cfg)/100
            logger.info(f"Initial validation loss: {min_val_loss:.3f}")
            if valuenet is not None:
                logger.info(f'Using Value Loss weight: {float(cfg["TRAIN"]["valuenet_weight"]):.3f}')
            valuenet.eval()
            for epoch in range(cfg["TRAIN"]["epochs"]):
                print(f"Now we are using valuenet weight: {valuenet_weight}")
                val_ade = train(cfg, epoch, model, optimizer, logger, valuenet, dataloader_train, dataloader_val, min_val_loss, experiment_name=exp_name, dataset_name=dataset, limit_obs=False)
                if val_ade < min_val_loss:
                    min_val_loss = val_ade
                    best_epoch = epoch
                    print('------------------------------BEST MODEL UPDATED------------------------------')
                    # torch.save(model.state_dict(), os.path.join(cfg["OUTPUT"]["model_dir"], f"{exp_name}_best.pth"))
                    # logger.info(f"Saving model at epoch {epoch} with validation loss {min_val_loss:.3f}")
                trial.report(val_ade, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                # if cfg['dry_run']:
                #     break

            logger.info(f"Best validation loss: {min_val_loss:.3f} at epoch {best_epoch}")
            if min_val_loss < best_score:
                logger.info(f"Saving model with value_w {valuenet_weight:.3f} and validation loss {min_val_loss:.3f}")
                save_checkpoint(model, optimizer, best_epoch, cfg, f"best_val_0{int(valuenet_weight)*100}_0{int(min_val_loss*1000)}.pth.tar", logger)
            return min_val_loss
    return objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name. Otherwise will use timestamp")
    parser.add_argument("--cfg", type=str, default="configs/jta_all_visual_cues.yaml", help="Config name. Otherwise will use default config")
    parser.add_argument('--dry-run', action='store_true', help="Run just one iteration")
    parser.add_argument('--not_pose', action='store_true', help="Not using pose input for value function")
    parser.add_argument('--not_vel', action='store_true', help="Not using velocity input for value function")
    parser.add_argument('--value_dir', type=str, default="/home/halo/plausibl/pacer/output/exp/pacer/", help="Directory to the value network checkpoint")
    parser.add_argument('--noisy_traj', action='store_true', help="Add noise to trajectory")
    args = parser.parse_args()

    if args.cfg != "":
        cfg = load_config(args.cfg, exp_name=args.exp_name, dataset_name="JTA")
    else:
        cfg = load_default_config()

    cfg['dry_run'] = args.dry_run
    cfg['RESUME'] = False
    cfg["USE_VALUELOSS"] = True
    cfg["USE_POSE"] = not args.not_pose
    cfg["USE_VELOCITY"] = not args.not_vel
    cfg["USE_FRAME_MASK"] = True
    cfg["hypara_tune"] = True
    cfg["MULTI_MODAL"] = False
    cfg["NOISY_TRAJ"] = 0
    cfg["VAL_LOSS_ONLY"] = False

    if torch.cuda.is_available():
        cfg["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
    else:
        cfg["DEVICE"] = "cpu"

    # torch.autograd.set_detect_anomaly(True)

    random.seed(cfg['SEED'])
    torch.manual_seed(cfg['SEED'])
    np.random.seed(cfg['SEED'])

    dataset = cfg["DATA"]["train_datasets"]

    # cfg["TRAIN"]["epochs"] = 20
    cfg["TRAIN"]["batch_size"] = 20

    logger = create_logger(cfg["OUTPUT"]["log_dir"])
    logger.info("Initializing with config:")
    logger.info(cfg)

    logger.info(f"!!!! Train for {cfg['TRAIN']['epochs']} epochs in hyper-parameter tuning !!!!")

    valuenet, dataloader_train, dataloader_val = prepare(cfg, logger, args.exp_name, dataset)

    if not os.path.exists(f"experiments/JTA/{args.exp_name}"):
        os.makedirs(f"experiments/JTA/{args.exp_name}")

    search_space = {"valuenet_weight": np.arange(0.8, 0.99, 0.02)}
    study = optuna.create_study(direction='minimize', study_name="my_study", storage=f"sqlite:///./experiments/JTA/{args.exp_name}/study.db", load_if_exists=True, sampler=optuna.samplers.GridSampler(search_space), pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(n_warmup_steps=5), patience=3))
    study = optuna.load_study(study_name="my_study", storage=f"sqlite:///./experiments/JTA/{args.exp_name}/study.db",)
    study.optimize(valuenet_opt(cfg, logger, args.exp_name, valuenet, dataloader_train, dataloader_val, study), n_trials=len(search_space["valuenet_weight"]))
    logger.info("Number of finished trials: {}".format(len(study.trials)))
    logger.info(f"Best ADE: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    fig1, fig2 = optuna.visualization.plot_slice(study), optuna.visualization.plot_optimization_history(study)
    fig1.write_image(f"experiments/JTA/{args.exp_name}/optuna_slice.png")
    fig2.write_image(f"experiments/JTA/{args.exp_name}/optuna_optimization_history.png")

    logger.info("All done.")