"""Consolidate per-batch SMPL fit outputs into JTA preprocess_smpl_cvpr shards.

Pipeline:
  fit/output/JTA_cross_fixed/jtapose_<split>_part<N>/batch*_params.pkl
        (key format: "part<N>_scene<S>_person<I>_frame<F>", Jtr (N, 24, 3))
  social-transmotion/data/jta_all_visual_cues/preprocess/{split}/part_<N>.pkl
        (list[scene] of list[person] of (joints[T,25,4], mask[T,25]))
        |
        v
  social-transmotion/data/jta_all_visual_cues/preprocess_smpl_cvpr/{split}/part_<N>.pt
        (list[scene] of list[person] of (joints[T,27,4], mask[T,27]))
          joints[:, :3,  :]  = traj + 2dbb (3 tokens, copied from preprocess)
          joints[:, 3:27, :3]= SMPL Jtr 24 joints (AMP-reordered via SMPL_mapping)
          joints[:, 27:, :]  = remaining tokens (2dpose, etc.) from preprocess
"""

import argparse
import copy
import os
import pickle

import numpy as np
import torch
import tqdm

# Mapping from SMPL fit output joint indices (col 0) to AMP joint indices (col 1).
SMPL_mapping = torch.tensor(
    [
        [0, 0],
        [1, 1],
        [2, 5],
        [3, 9],
        [4, 2],
        [5, 6],
        [6, 10],
        [7, 3],
        [8, 7],
        [9, 11],
        [10, 4],
        [11, 8],
        [12, 12],
        [13, 14],
        [14, 19],
        [15, 13],
        [16, 15],
        [17, 20],
        [18, 16],
        [19, 21],
        [20, 17],
        [21, 22],
        [22, 18],
        [23, 23],
    ]
)


def concat_frames(batch_data, save_pose_shape=False, dataset="jta", split="train"):
    """Aggregate per-frame SMPL fit outputs into per-person tensors.

    `pose_params` / `shape_params` are only needed when --save_pose_shape is
    requested (downstream SMPL-X transfer). Older fits saved only `Jtr` and
    `key_list`, so we read them lazily and skip pose/shape aggregation when
    the keys are missing.
    """
    Jtr = batch_data["Jtr"]
    has_pose_shape = "pose_params" in batch_data and "shape_params" in batch_data
    if save_pose_shape and not has_pose_shape:
        raise KeyError(
            "--save_pose_shape requires pose_params/shape_params in batch data, "
            "but the input fits were saved without them."
        )
    pose_params = batch_data["pose_params"] if has_pose_shape else None
    shape_params = batch_data["shape_params"] if has_pose_shape else None

    joint_dict, pose_dict, shape_dict = {}, {}, {}
    for i, frame_key in enumerate(batch_data["key_list"]):
        person_key, frame_num = (
            frame_key.split("_frame")[0],
            int(frame_key.split("_frame")[1]),
        )
        joint_dict.setdefault(person_key, {})[frame_num] = Jtr[i]
        if has_pose_shape:
            pose_dict.setdefault(person_key, {})[frame_num] = pose_params[i]
            shape_dict.setdefault(person_key, {})[frame_num] = shape_params[i]

    frame_concat_dict_joint, frame_concat_dict_pose, frame_concat_dict_shape = (
        {},
        {},
        {},
    )
    for concat_dict, dict_to_concat in zip(
        [frame_concat_dict_joint, frame_concat_dict_pose, frame_concat_dict_shape],
        [joint_dict, pose_dict, shape_dict],
    ):
        for person_key in dict_to_concat.keys():
            sorted_person_info = sorted(
                dict_to_concat[person_key].items(), key=lambda x: x[0]
            )
            person_array = np.array([info for _, info in sorted_person_info])
            concat_dict[person_key] = torch.from_numpy(person_array).float()

    if save_pose_shape:
        save_pose_and_shape(
            frame_concat_dict_pose,
            frame_concat_dict_shape,
            dataset=dataset,
            split=split,
        )
    return frame_concat_dict_joint


def save_pose_and_shape(
    frame_concat_dict_pose, frame_concat_dict_shape, dataset="jta", split="train"
):
    """Save (poses, betas) per person for downstream SMPL-X transfer. Skips NaN frames."""
    nan_any = 0
    nan_all = 0
    for person_key in frame_concat_dict_pose.keys():
        # primary pedestrian only
        if "person0" not in person_key:
            continue
        pose_array = frame_concat_dict_pose[person_key]
        shape_array = frame_concat_dict_shape[person_key]
        motion_dict = {"poses": pose_array, "betas": shape_array}
        if torch.isnan(pose_array).all(dim=1).any():
            nan_any += torch.isnan(pose_array).any(dim=1).sum().item()
        if torch.isnan(pose_array).any(dim=1).any():
            nan_frames = torch.isnan(pose_array).all(dim=1).sum().item()
            nan_all += nan_frames
        else:
            nan_frames = 0
        assert nan_any == nan_all, "nan_any and nan_all are not equal!"
        if nan_frames == pose_array.shape[0]:
            continue
        save_dir = f"../../../smplx/transfer_data/motion_data/{dataset}/{split}"
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{person_key}.pkl"), "wb") as f:
            pickle.dump(motion_dict, f)
    print(f"nan_any: {nan_any}")


def load_preprocessed_part_data(part_name, preprocess_root):
    split = part_name.split("_")[1]
    part = part_name.split("part")[1]
    # Prefer .pt (dataset_jta.initialize() output); fall back to legacy .pkl.
    pt_path = f"{preprocess_root}/{split}/part_{part}.pt"
    pkl_path = f"{preprocess_root}/{split}/part_{part}.pkl"
    if os.path.exists(pt_path):
        return torch.load(pt_path, weights_only=False)
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def replace_batch_smpl_pose(jta_data, smpl_data, frame_concat_dict, jta_data_counter):
    for person_key in frame_concat_dict.keys():
        scene_id = int(person_key.split("_")[1].split("scene")[1])
        person_id = int(person_key.split("_")[2].split("person")[1])
        jta_person_info = jta_data[scene_id][person_id][0]
        smpl_pose_tensor_before = frame_concat_dict[person_key]
        smpl_pose_tensor_after = torch.zeros_like(smpl_pose_tensor_before)
        smpl_pose_tensor_after[:, SMPL_mapping[:, 1]] = smpl_pose_tensor_before[
            :, SMPL_mapping[:, 0]
        ]

        smpl_person_info = torch.zeros(
            (
                jta_person_info.shape[0],
                jta_person_info.shape[1] + 2,
                jta_person_info.shape[2],
            )
        )
        smpl_person_info[:, :3, :] = jta_person_info[:, :3, :]
        smpl_person_info[:, 3:27, :3] = smpl_pose_tensor_after
        smpl_person_info[:, 27:, :] = jta_person_info[:, 25:, :]
        smpl_person_mask = torch.ones(
            (jta_person_info.shape[0], jta_person_info.shape[1] + 2)
        )

        smpl_data[scene_id][person_id] = (smpl_person_info, smpl_person_mask)
        jta_data_counter -= 1
    return smpl_data, jta_data_counter


def save_part_smpl_data(smpl_data, part_name, output_dir):
    split = part_name.split("_")[1]
    part = part_name.split("part")[1]

    output_dir = os.path.join(output_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    out_path = f"{output_dir}/part_{part}.pt"
    torch.save(smpl_data, out_path)


def concat_and_save(args):
    part_bar = tqdm.tqdm(os.listdir(args.input_dir), dynamic_ncols=True, leave=True)
    for part_name in part_bar:
        if args.part != "" and args.part != part_name:
            continue
        part_bar.set_description(f"Processing part: {part_name}")
        jta_data = load_preprocessed_part_data(part_name, args.preprocess_root)
        jta_data_counter = sum(
            [len(jta_data[scene_id]) for scene_id in range(len(jta_data))]
        )
        smpl_data = copy.deepcopy(jta_data)

        batch_bar = tqdm.tqdm(
            os.listdir(os.path.join(args.input_dir, part_name)),
            leave=False,
            dynamic_ncols=True,
        )
        for batch_i, batch_file in enumerate(batch_bar):
            batch_bar.set_description(
                f"Batch: {batch_i}, Remaining JTA data: {jta_data_counter}"
            )
            with open(os.path.join(args.input_dir, part_name, batch_file), "rb") as f:
                batch_data = pickle.load(f)
            frame_concat_dict = concat_frames(
                batch_data,
                dataset="jta",
                split=part_name.split("_")[1],
                save_pose_shape=args.save_pose_shape,
            )
            smpl_data, jta_data_counter = replace_batch_smpl_pose(
                jta_data, smpl_data, frame_concat_dict, jta_data_counter
            )
        assert jta_data_counter == 0, "jta data counter is not zero!"

        save_part_smpl_data(smpl_data, part_name, args.output_dir)
    print("All parts finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate and save SMPL 3D pose data (JTA)"
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        default="fit/output/JTA_cross_fixed",
        help="Directory holding per-part SMPL fit outputs (jtapose_<split>_part<N>/batch*_params.pkl)",
        type=str,
    )
    parser.add_argument(
        "--preprocess_root",
        dest="preprocess_root",
        default="../../social-transmotion/data/jta_all_visual_cues/preprocess",
        help="Base preprocess providing the traj+2dbb+...+2dpose token layout",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default="../../social-transmotion/data/jta_all_visual_cues/preprocess_smpl_cvpr",
        help="Target directory for J=49 preprocess shards (.pt)",
        type=str,
    )
    parser.add_argument(
        "--part_name",
        dest="part",
        default="",
        help="Specify part name, if not specified, process all parts in input_dir",
        type=str,
    )
    parser.add_argument(
        "--save_pose_shape",
        dest="save_pose_shape",
        action="store_true",
        help="Also dump per-person (poses, betas) pickles for SMPL-X transfer",
    )
    args = parser.parse_args()
    concat_and_save(args)
