"""SMPL fit parameter saver. Writes per-batch SMPL fit output to disk.

The CVPR EmLoco pipeline only needs `save_params` — the original
`save_pic` / `save_single_pic` from Pose_to_SMPL depended on
`display_utils.display_model`, which has been dropped from the public
release together with the rest of the visualization helpers.
"""

import os
import pickle
import re


from label import get_label


def create_dir_not_exist(path):
    os.makedirs(path, exist_ok=True)


def save_params(res, file, logger, dataset_name, key_list, batch_idx, exp_name):
    """Dump (pose_params, shape_params, Jtr) for one batch into fit/output/.

    Output layout:
      fit/output/<dataset_name>{_<exp>}/<file_basename>/batch<batch_idx>_params.pkl
    """
    pose_params, shape_params, verts, Jtr = res
    file_name = re.split("[/.]", file)[-2]
    if exp_name:
        fit_path = f"fit/output/{dataset_name}_{exp_name}/{file_name}"
    else:
        fit_path = f"fit/output/{dataset_name}/{file_name}"
    create_dir_not_exist(fit_path)
    logger.info(f"Saving params at {fit_path}")
    label = get_label(file_name, dataset_name)
    pose_params = pose_params.numpy().tolist()
    shape_params = shape_params.numpy().tolist()
    Jtr = Jtr.numpy()

    # Put the Y and Z axes back to the original (un-fit) orientation so that
    # downstream consolidation operates in the dataset's native coordinate system.
    if dataset_name == "JTA":
        Jtr_y = Jtr[:, :, 2].copy()
        Jtr_z = Jtr[:, :, 1].copy() * -1
    elif dataset_name == "JRDB":
        Jtr[:, :, 0] *= -1
        Jtr_y = Jtr[:, :, 2].copy()
        Jtr_z = Jtr[:, :, 1].copy()
    elif dataset_name == "VRU":
        Jtr_y = Jtr[:, :, 2].copy()
        Jtr_z = Jtr[:, :, 1].copy()
        Jtr[:, :, 0] *= -1
    else:
        Jtr_y = Jtr[:, :, 1].copy()
        Jtr_z = Jtr[:, :, 2].copy()
    Jtr[:, :, 1] = Jtr_y
    Jtr[:, :, 2] = Jtr_z
    Jtr = Jtr.tolist()

    params = {
        "key_list": key_list,
        "label": label,
        "pose_params": pose_params,
        "shape_params": shape_params,
        "Jtr": Jtr,
    }
    with open(os.path.join(fit_path, f"batch{batch_idx}_params.pkl"), "wb") as f:
        pickle.dump(params, f)
