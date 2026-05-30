import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())


def init(smpl_layer, target, device, cfg):
    params = {}
    params["pose_params"] = torch.zeros(target.shape[0], 72)
    # params["pose_params"] = torch.zeros(target.shape[0], target.shape[1], 72)
    params["shape_params"] = torch.zeros(target.shape[0], 10)
    # params["shape_params"] = torch.zeros(target.shape[0], target.shape[1], 10)
    params["scale"] = torch.ones([1])

    smpl_layer = smpl_layer.to(device)
    params["pose_params"] = params["pose_params"].to(device)
    params["shape_params"] = params["shape_params"].to(device)
    target = target.to(device)
    params["scale"] = params["scale"].to(device)

    params["pose_params"].requires_grad = True
    params["shape_params"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SHAPE)
    params["scale"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SCALE)

    optim_params = [
        {"params": params["pose_params"], "lr": cfg.TRAIN.LEARNING_RATE},
        {"params": params["shape_params"], "lr": cfg.TRAIN.LEARNING_RATE},
        {"params": params["scale"], "lr": cfg.TRAIN.LEARNING_RATE * 10},
    ]
    optimizer = optim.Adam(optim_params)

    index = {}
    smpl_index = []
    dataset_index = []
    for tp in cfg.DATASET.DATA_MAP:
        smpl_index.append(tp[0])
        dataset_index.append(tp[1])

    index["smpl_index"] = torch.tensor(smpl_index).to(device)
    index["dataset_index"] = torch.tensor(dataset_index).to(device)

    return smpl_layer, params, target, optimizer, index


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def train(smpl_layer, target, logger, writer, device, args, cfg, meters):
    res = []
    ftol = (float(1e-6),)
    smpl_layer, params, target, optimizer, index = init(smpl_layer, target, device, cfg)
    pose_params = params["pose_params"]
    shape_params = params["shape_params"]
    scale = params["scale"]

    with torch.no_grad():
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        params["scale"] *= torch.max(torch.abs(target)) / torch.max(torch.abs(Jtr))

    batch_bar = tqdm(range(cfg.TRAIN.MAX_EPOCH), leave=False, dynamic_ncols=True)
    prev_loss = None
    loss_rel_change = None
    for epoch, epo in enumerate(batch_bar):
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        loss = F.smooth_l1_loss(
            scale * Jtr.index_select(1, index["smpl_index"]),
            target.index_select(1, index["dataset_index"]),
        )
        # import pdb; pdb.set_trace()
        if epoch != 0 and prev_loss is not None:
            loss_rel_change = rel_change(prev_loss, loss.item())
            if loss_rel_change <= ftol[0]:
                break

        prev_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meters.update_early_stop(float(loss))
        if meters.update_res:
            res = [pose_params, shape_params, verts, Jtr]
        if meters.early_stop:
            logger.info("Early stop at epoch {} !".format(epoch))
            break

        if epoch % cfg.TRAIN.WRITE == 0:
            # logger.info("Epoch {}, lossPerBatch={:.6f}, scale={:.4f}".format(
            #         epoch, float(loss),float(scale)))
            # print("Epoch {}, lossPerBatch={:.6f}, scale={:.4f}".format(
            #  epoch, float(loss),float(scale)))
            if loss_rel_change is not None:
                batch_bar.set_description(
                    "loss={:.5f}, scale={:.3f}, rel_change={:.5f}".format(
                        float(loss), float(scale), float(loss_rel_change)
                    )
                )
            else:
                batch_bar.set_description(
                    "loss={:.5f}, scale={:.3f}".format(float(loss), float(scale))
                )
            writer.add_scalar("loss", float(loss), epoch)
            writer.add_scalar(
                "learning_rate",
                float(optimizer.state_dict()["param_groups"][0]["lr"]),
                epoch,
            )
            # save_single_pic(res,smpl_layer,epoch,logger,args.dataset_name,target)
    batch_bar.close()

    logger.info("Train ended, min_loss = {:.4f}".format(float(meters.min_loss)))
    return res
