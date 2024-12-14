import sys
sys.path.append("/misc/dl00/halo/plausibl/pacer/pacer")
import argparse
from datetime import datetime
import numpy as np
import os
import random
import time
import torch
import pickle

import tqdm
from scipy.interpolate import interp1d, CubicSpline
from matplotlib import pyplot as plt

from progress.bar import Bar
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from dataset_jrdb import collate_batch, batch_process_coords, get_datasets, create_dataset
from utils.utils import create_logger, load_default_config, load_config
from load_jta_traj import dataloader_for_val, save_to_pkl

def get_primary_init_pose_jrdb(joints):
    return joints[:, 0, 8, 2:, :3]

def dataloader_for_val(dataset, config, **kwargs):
    return DataLoader(dataset,
                      batch_size=1,
                      num_workers=0,
                      collate_fn=collate_batch,
                      **kwargs)

def save_to_pkl(traj_dict, dataset_name, split):
    # Save the trajs to pkl file
    save_dir = os.path.join('data', 'saved_trajs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{dataset_name[0]}_{split}_trajs_filterv2.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(traj_dict, f)

def save_trajs(config, logger, dataset_name, dataloader, in_F, split='train', f_init_pose=None):
    num_trajs =  len(dataloader)
    dataiter = iter(dataloader)
    bar = Bar(f"Loading trajs", fill="#", max=num_trajs)

    traj_dict = {}
    traj_phase = np.array([0.0000, 0.0707, 0.1414, 0.2122, 0.2829, 0.3536, 0.4243, 0.4950, 0.5658, 0.6365, 0.7072, 0.7779, 0.8487]) * 100

    # import pdb; pdb.set_trace()
    for id in tqdm.tqdm(range(num_trajs)):
        try:
            joints, masks, padding_mask, idxs_list = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            joints, masks, padding_mask, idxs_list = next(dataiter)

        # in_joints, _, out_joints, _, _ = batch_process_coords(joints, masks, padding_mask, config, training=False)

        # import pdb; pdb.set_trace()

        primary_init_pose = f_init_pose(joints)
        nan_mask_pose = torch.isnan(primary_init_pose).any(dim=1).any(dim=1)

        traj = joints[0,0,in_F-1:,0,:3]
        nan_mask_traj = torch.isnan(traj).any()

        if not nan_mask_traj.any():
            # import pdb; pdb.set_trace()
            # interpolate the trajectory
            scene_data = traj.squeeze().numpy()[:13]
            natural = CubicSpline(traj_phase, scene_data, axis=0, bc_type='natural')
            interp_idx = np.arange(101)
            converted_scene = natural(interp_idx)

            # f = interp1d(traj_phase, scene_data, axis=0, kind="cubic", fill_value="extrapolate", bounds_error=False)
            # converted_scene = f(interp_idx)

            # visualize
            if len(traj_dict.keys()) < 5:
                plt.figure()
                plt.plot(scene_data[:,0], scene_data[:,1], 'o')
                # plot frame number next to the point
                plt.text(scene_data[0,0], scene_data[0,1], '0')
                plt.text(scene_data[-1,0], scene_data[-1,1], '12')
                plt.plot(converted_scene[:,0], converted_scene[:,1])
                plt.savefig(f"traj_{id}.png")
                plt.close()

            if nan_mask_pose.any():
                traj_dict[id] = {'pose': None, 'traj': converted_scene}
            else:
                traj_dict[id] = {'pose': primary_init_pose.squeeze(), 'traj': converted_scene}

    bar.finish()

    save_to_pkl(traj_dict, dataset_name, split)
    logger.info(f"Saved {len(traj_dict.keys())} trajectories!")
    logger.info("Done.")

def main(config, logger, experiment_name, dataset_name, f_init_pose):

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    dataset_train = ConcatDataset(get_datasets(config['DATA']['train_datasets'], config, logger))
    dataloader_train = dataloader_for_val(dataset_train, config, shuffle=False, pin_memory=True)
    logger.info(f"Created training dataset!")

    save_trajs(config, logger, dataset_name, dataloader_train, in_F, split='train', f_init_pose=f_init_pose)

    dataset_val = create_dataset(config['DATA']['train_datasets'][0], logger, split="val", track_size=(in_F+out_F), track_cutoff=in_F, preprocessed=config['DATA']['preprocessed'])
    dataloader_val = dataloader_for_val(dataset_val, config, shuffle=False, pin_memory=True)
    logger.info(f"Created validation dataset!")

    save_trajs(config, logger, dataset_name, dataloader_val, in_F, split='val', f_init_pose=f_init_pose)

    dataset_test = create_dataset(config['DATA']['train_datasets'][0], logger, split="test", track_size=(in_F+out_F), track_cutoff=in_F, preprocessed=config['DATA']['preprocessed'])
    dataloader_test = dataloader_for_val(dataset_test, config, shuffle=False, pin_memory=True)
    logger.info(f"Created test dataset!")

    save_trajs(config, logger, dataset_name, dataloader_test, in_F, split='test', f_init_pose=f_init_pose)

    logger.info("All Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/jrdb_all_visual_cues.yaml", help="Config name. Otherwise will use default config")
    args = parser.parse_args()

    if args.cfg != "":
        cfg = load_config(args.cfg, exp_name='save_trajs')
    else:
        cfg = load_default_config()

    if torch.cuda.is_available():
        cfg["DEVICE"] = "cuda"
    else:
        cfg["DEVICE"] = "cpu"

    random.seed(cfg['SEED'])
    torch.manual_seed(cfg['SEED'])
    np.random.seed(cfg['SEED'])

    dataset = cfg["DATA"]["train_datasets"]

    logger = create_logger(cfg["OUTPUT"]["log_dir"])
    logger.info("Initializing with config:")
    logger.info(cfg)

    main(cfg, logger, experiment_name='save_trajs', dataset_name=dataset, f_init_pose=get_primary_init_pose_jrdb)
