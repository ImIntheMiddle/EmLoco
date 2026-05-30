"""Consolidate per-part SMPL fit outputs into JRDB preprocess_smpl shards with
action-aware pose masking (CVPR 2025 EmLoco reproduction).

Pipeline:
  fit/output/JRDB_cross_fixed/jrdbpose_<split>_part<N>/batch*_params.pkl
        (key format: "part<N>_scene<S>_person<I>_frame<F>", Jtr (N, 24, 3))
  jrdb_2dbox/preprocess/{split}/part_<N>.pkl
        (list[scene] of list[person] of (joints[T,2,4], mask[T,2], (scene_name, ped_ids)))
  action_dict.json
        ({split: {base_scene: {frame_str: {ped_str: action}}}})
        ↓
  jrdb_all_visual_cues/preprocess_smpl_cvpr/{split}/part_<N>.pkl
     list[scene] of list[person] of (joints[T, 26, 4], mask[T, 26], (scene_name, ped_ids))
       joints[:, 0:2, :] = traj + 2dbb (copied from jrdb_2dbox)
       joints[:, 2:26, :3] = SMPL Jtr 24 joints (AMP-reordered via SMPL_MAPPING)
       mask[:, 0:2] = original traj+2dbb mask
       mask[:, 2:26] = 1 iff action_dict[split][base_scene][frame][ped] exists, else 0

This is the *action-aware mask* variant of save_jrdb_smplpose.py — same per-part /
per-scene / per-person / per-frame correspondence, but instead of writing all-1s
mask over pose tokens, we toggle the pose-token mask per-frame using action_dict.
"""

import argparse
import json
import os
import pickle
import re
import sys

import torch

# numpy 2.x → 1.x compatibility shim (some shards were saved under numpy 2.x).
import numpy as _np

if not hasattr(_np, "_core"):
    sys.modules.setdefault("numpy._core", _np.core)
    sys.modules.setdefault("numpy._core.multiarray", _np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", _np.core.numeric)

# Mapping from SMPL fit output joint indices (col 0) to AMP / dataset joint
# indices (col 1). Inherited verbatim from save_jrdb_smplpose.py.
SMPL_MAPPING = torch.tensor(
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

# Key format example: "part0_scene12_person3_frame5"
_KEY_RE = re.compile(r"part(\d+)_scene(\d+)_person(\d+)_frame(\d+)")


def parse_key(key):
    m = _KEY_RE.match(key)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def load_smpl_jtr_for_part(part_dir):
    """Return dict[(scene_idx, person_idx, frame_idx)] -> tensor[24, 3]."""
    out = {}
    for batch_file in sorted(os.listdir(part_dir)):
        if not batch_file.endswith(".pkl"):
            continue
        with open(os.path.join(part_dir, batch_file), "rb") as f:
            batch_data = pickle.load(f)
        key_list = batch_data["key_list"]
        jtr = batch_data["Jtr"]  # (N, 24, 3)
        for i, key in enumerate(key_list):
            parsed = parse_key(key)
            if parsed is None:
                continue
            _part_id, scene_idx, person_idx, frame_idx = parsed
            out[(scene_idx, person_idx, frame_idx)] = torch.as_tensor(
                jtr[i], dtype=torch.float32
            )
    return out


def build_preprocess_smpl_shard(scene_list, smpl_jtr, scene_action_dict):
    """Convert one shard's worth of jrdb_2dbox scenes into J=26 SMPL preprocess."""
    out_scenes = []
    n_pose_filled = 0
    n_action_filled = 0
    n_total_frames = 0

    for scene_idx, scene in enumerate(scene_list):
        out_persons = []
        for person_idx, person in enumerate(scene):
            traj_bbox_joints = torch.as_tensor(
                person[0], dtype=torch.float32
            )  # (T, 2, 4)
            traj_bbox_mask = torch.as_tensor(person[1], dtype=torch.float32)  # (T, 2)
            meta = person[2]
            scene_name = meta[0]
            ped_ids = meta[1]  # numpy (T, 2) → cols (frame_num, ped_id) in raw JRDB
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode()
            base_scene = scene_name.split("_shift_")[0]
            T = traj_bbox_joints.shape[0]
            n_total_frames += T

            # CVPR-era jrdb_3dpose convention:
            #   - mask is all-1s (loss skip uses NaN-detection on values, not the mask)
            #   - pose tokens (2:26) are NaN-filled by default; only frames with a
            #     JRDB-Act action label receive SMPL Jtr values. Frames without an
            #     action label remain NaN so EmLoco loss skips them automatically
            #     while the trajectory tokens (0:2) keep their normal values.
            joints = torch.full((T, 26, 4), float("nan"), dtype=torch.float32)
            mask = torch.ones((T, 26), dtype=torch.float32)
            joints[:, :2, :] = traj_bbox_joints
            mask[:, :2] = traj_bbox_mask  # let the 2dbb mask override if needed

            for t in range(T):
                jtr_24 = smpl_jtr.get((scene_idx, person_idx, t))
                if jtr_24 is None:
                    # SMPL fit unavailable — pose stays NaN.
                    continue
                # NaN ped_ids indicate padding in jrdb_2dbox: cannot look up an action.
                if _np.isnan(ped_ids[t, 0]) or _np.isnan(ped_ids[t, 1]):
                    continue
                frame_num = int(ped_ids[t, 0])
                ped_id = int(ped_ids[t, 1])
                action = (
                    scene_action_dict.get(base_scene, {})
                    .get(str(frame_num), {})
                    .get(str(ped_id))
                )
                if action is None:
                    # No JRDB-Act label → keep pose tokens NaN; EmLoco loss skips this frame.
                    continue
                # Action label exists → fill pose tokens with SMPL Jtr (AMP-reordered).
                jtr_mapped = torch.zeros_like(jtr_24)
                jtr_mapped[SMPL_MAPPING[:, 1]] = jtr_24[SMPL_MAPPING[:, 0]]
                joints[t, 2:, :3] = jtr_mapped
                joints[t, 2:, 3] = 0.0  # replace the NaN in the 4th channel
                n_pose_filled += 1
                n_action_filled += 1

            out_persons.append((joints, mask, meta))
        out_scenes.append(out_persons)
    return out_scenes, n_total_frames, n_pose_filled, n_action_filled


def consolidate_split(split, args, action_dict):
    print(f"\n=== Split: {split} ===")
    jrdb_2dbox_split_dir = os.path.join(args.jrdb_2dbox_dir, split)
    if not os.path.isdir(jrdb_2dbox_split_dir):
        print(f"  jrdb_2dbox split missing: {jrdb_2dbox_split_dir}")
        return
    out_dir = os.path.join(args.output_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    scene_action_dict = action_dict.get(split, {})

    parts = sorted(p for p in os.listdir(jrdb_2dbox_split_dir) if p.endswith(".pkl"))
    grand_total = grand_pose = grand_action = 0

    for part_file in parts:
        # part_file: "part_0.pkl" -> part_id = 0
        part_id = int(re.search(r"part_(\d+)", part_file).group(1))
        fit_dir = os.path.join(args.input_dir, f"jrdbpose_{split}_part{part_id}")
        if not os.path.isdir(fit_dir):
            print(f"  fit output missing for part {part_id}: {fit_dir}")
            continue

        print(f"  → part {part_id}: loading SMPL Jtr from {fit_dir}")
        smpl_jtr = load_smpl_jtr_for_part(fit_dir)
        print(f"     SMPL Jtr keys: {len(smpl_jtr)}")

        with open(os.path.join(jrdb_2dbox_split_dir, part_file), "rb") as f:
            scene_list = pickle.load(f)
        print(f"     jrdb_2dbox scenes: {len(scene_list)}")

        out_scenes, n_total, n_pose, n_action = build_preprocess_smpl_shard(
            scene_list, smpl_jtr, scene_action_dict
        )
        grand_total += n_total
        grand_pose += n_pose
        grand_action += n_action

        # Save as torch .pt (matches dataset_jta.py / dataset_jrdb.py loader convention).
        out_path = os.path.join(out_dir, part_file.replace(".pkl", ".pt"))
        torch.save(out_scenes, out_path)
        print(
            f"     wrote {out_path} ({len(out_scenes)} scenes, "
            f"pose-filled={n_pose}/{n_total}, action-labeled={n_action})"
        )

    if grand_total > 0:
        print(
            f"  Split summary: pose-filled {grand_pose}/{grand_total} "
            f"({grand_pose / grand_total:.1%}), "
            f"action-labeled {grand_action}/{grand_total} ({grand_action / grand_total:.1%})"
        )


def main():
    parser = argparse.ArgumentParser(description="Action-aware SMPL pose consolidation")
    parser.add_argument(
        "--input_dir",
        default="fit/output/JRDB_cross_fixed",
        help=(
            "Directory holding per-part SMPL fit outputs "
            "(jrdbpose_{split}_part{N}/batch*_params.pkl). Run from joints2smpl/Pose_to_SMPL/."
        ),
    )
    parser.add_argument(
        "--jrdb_2dbox_dir",
        default="../../social-transmotion/data/jrdb_2dbox/preprocess",
        help="Base preprocess providing traj+2dbb tokens and shard structure",
    )
    parser.add_argument(
        "--action_dict",
        default="action_dict.json",
        help="JRDB-Act action labels JSON",
    )
    parser.add_argument(
        "--output_dir",
        default="../../social-transmotion/data/jrdb_all_visual_cues/preprocess_smpl_cvpr",
        help="Target directory for J=26 action-masked preprocess shards (.pt)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process",
    )
    args = parser.parse_args()

    with open(args.action_dict) as f:
        action_dict = json.load(f)
    print(
        f"Loaded action_dict: splits={list(action_dict.keys())}, "
        f"scenes per split={ {k: len(v) for k, v in action_dict.items()} }"
    )

    for split in args.splits:
        consolidate_split(split, args, action_dict)

    print("\nAll done.")


if __name__ == "__main__":
    main()
