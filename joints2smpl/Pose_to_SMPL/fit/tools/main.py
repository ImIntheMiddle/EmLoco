"""SMPL fitting entry point for JTA / JRDB / etc.

Walks DATASET.PATH for `*part<N>.pkl` shards, fits SMPL parameters per
batch, and writes them to `fit/output/<dataset>/<file>/batch<i>_params.pkl`.
The downstream consolidation scripts
(`save_jta_smplpose.py`, `consolidate_jrdb_with_action_filter.py`) read
those outputs and emit the J=49 / J=26 shards consumed by Social-Transmotion.
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
import tqdm
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from dataset import PoseDataset
from meters import Meters
from save import save_params
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from train import train

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Fit SMPL")
    parser.add_argument(
        "--exp",
        dest="exp",
        default=time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time())),
        help="Experiment name (output goes to exp/<exp>/)",
        type=str,
    )
    parser.add_argument(
        "--dataset_name",
        "-n",
        dest="dataset_name",
        default="",
        help="Dataset name (JTA, JRDB, VRU, ...)",
        type=str,
    )
    parser.add_argument(
        "--dataset_path",
        dest="dataset_path",
        default=None,
        help="Override cfg.DATASET.PATH",
        type=str,
    )
    parser.add_argument(
        "--save_params",
        dest="save_params",
        action="store_true",
        help="Save per-batch fit parameters to disk",
    )
    parser.add_argument(
        "--split",
        dest="split",
        default="all",
        help="Which split substring to match (all / train / val / test)",
        type=str,
    )
    parser.add_argument(
        "--part_start",
        dest="part_start",
        default=0,
        help="First part index to process (inclusive)",
        type=int,
    )
    parser.add_argument(
        "--part_end",
        dest="part_end",
        default=-1,
        help="Last part index to process (-1 = no limit)",
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        default=-1,
        help="Override cfg.TRAIN.BATCH_SIZE (-1 = use config value)",
        type=int,
    )
    return parser.parse_args()


def get_config(args):
    config_path = f"fit/configs/{args.dataset_name}.json"
    with open(config_path, "r") as f:
        data = json.load(f)
    cfg = edict(data.copy())
    if args.dataset_path is not None:
        cfg.DATASET.PATH = args.dataset_path
    if args.batch_size != -1:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    return cfg


def set_device(USE_GPU):
    if USE_GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(cur_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(os.path.join(cur_path, "log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    writer = SummaryWriter(os.path.join(cur_path, "tb"))
    return logger, writer


if __name__ == "__main__":
    args = parse_args()

    cur_path = os.path.join(os.getcwd(), "exp", args.exp)
    os.makedirs(cur_path, exist_ok=True)

    cfg = get_config(args)
    json.dump(dict(cfg), open(os.path.join(cur_path, "config.json"), "w"))

    logger, writer = get_logger(cur_path)
    logger.info("Start print log")

    device = set_device(USE_GPU=cfg.USE_GPU)
    logger.info(f"using device: {device}")

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender=cfg.MODEL.GENDER,
        model_root="smplpytorch/native/models",
    )

    meters = Meters()
    file_num = 0
    for root, _, files in os.walk(cfg.DATASET.PATH):
        for file in sorted(files):
            if args.split != "all" and args.split not in file:
                continue
            part = int(file.split(".pkl")[0].split("part")[1])
            if part < args.part_start or (args.part_end != -1 and part > args.part_end):
                continue
            file_num += 1
            logger.info(f"Processing file: {file}    [{file_num} / {len(files)}]")
            dataset = PoseDataset(args.dataset_name, root, file)
            dataloader = DataLoader(
                dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                drop_last=False,
                pin_memory=True,
                collate_fn=dataset.collate_fn,
            )
            logger.info(f"dataset shape:{dataset.shape()}")

            bar = tqdm.tqdm(dataloader)
            for batch_idx, (idxs, key_list, target) in enumerate(bar):
                nans = torch.isnan(target).any(dim=1).any(dim=1)
                nan_idxs = torch.where(nans)
                assert torch.isnan(target[nan_idxs]).all(), "irregular nan detected"
                real_idxs = torch.where(~nans)
                input_target = target.clone()[real_idxs]
                pose_params, shape_params, verts, Jtr = train(
                    smpl_layer,
                    input_target,
                    logger,
                    writer,
                    device,
                    args,
                    cfg,
                    meters,
                )

                pose_params = pose_params.cpu().detach()
                shape_params = shape_params.cpu().detach()
                verts = verts.cpu().detach()
                Jtr = Jtr.cpu().detach()

                pose_params_full = torch.zeros(target.shape[0], 72)
                shape_params_full = torch.zeros(target.shape[0], 10)
                verts_full = torch.zeros(target.shape[0], 6890, 3)
                Jtr_full = torch.zeros(target.shape[0], 24, 3)

                pose_params_full[real_idxs] = pose_params
                shape_params_full[real_idxs] = shape_params
                verts_full[real_idxs] = verts
                Jtr_full[real_idxs] = Jtr

                nan_tensor = torch.tensor(float("nan"))
                pose_params_full[nan_idxs] = nan_tensor
                shape_params_full[nan_idxs] = nan_tensor
                verts_full[nan_idxs] = nan_tensor
                Jtr_full[nan_idxs] = nan_tensor

                res = (pose_params_full, shape_params_full, verts_full, Jtr_full)

                meters.update_avg(meters.min_loss, k=target.shape[0])
                bar.set_description(f"avg_loss:{meters.avg:.4f}")
                meters.reset_early_stop()

                if args.save_params:
                    save_params(
                        res,
                        file,
                        logger,
                        args.dataset_name,
                        key_list,
                        batch_idx,
                        args.exp,
                    )

                del target, res
                torch.cuda.empty_cache()

            logger.info(f"avg_loss:{meters.avg:.4f}")

    logger.info(f"Fitting finished! Average loss: {meters.avg:.9f}")
