# 🛠️Data Preparation

> [!Note]
> Preprocessed `.pt` / `.pkl` shards are published on the Hugging Face Hub at [🤗 iminthemiddle/EmLoco](https://huggingface.co/iminthemiddle/EmLoco). Follow the *Data & Checkpoints* section of the [main README](../README.md) for the eval-only workflow.

## 0. Overview

```
=== JTA pipeline ===

JTA raw videos + annotations
  │  (Social-Transmotion preprocessing)
  ▼
social-transmotion/data/jta_all_visual_cues/preprocess/{train,val,test}/part_<N>.pt
  │
  ├─▶ load_jta_3dpose.py    ──▶ data/jta_all_visual_cues/original_pose/<split>/*.pkl   (required for SMPL fit)
  │     │
  │     ▼  joints2smpl/Pose_to_SMPL/fit/tools/main.py --dataset_name JTA --save_params
  │   fit/output/JTA_cross_fixed/jtapose_<split>_part<N>/batch*_params.pkl
  │     │
  │     ▼  joints2smpl/Pose_to_SMPL/fit/tools/save_jta_smplpose.py
  │   social-transmotion/data/jta_all_visual_cues/preprocess_smpl_cvpr/<split>/part_<N>.pt   ← final shards (J=49)
  │
  └─▶ load_jta_traj.py      ──▶ data/saved_trajs/jta_*_trajs.pkl   (only needed for PACER LocoVal training)


=== JRDB pipeline ===

JRDB raw + JRDB-Act labels
  │  (Social-Transmotion preprocessing)
  ▼
social-transmotion/data/jrdb_2dbox/preprocess/{train,val,test}/part_<N>.pt
  │
  ├─▶ load_jrdb_3dpose.py   ──▶ data/jrdb_all_visual_cues/original_pose/<split>/*.pkl   (required for SMPL fit)
  │     │
  │     ▼  joints2smpl/Pose_to_SMPL/fit/tools/main.py --dataset_name JRDB --save_params
  │   fit/output/JRDB_cross_fixed/jrdbpose_<split>_part<N>/batch*_params.pkl
  │     │
  │     ▼  joints2smpl/Pose_to_SMPL/fit/tools/consolidate_jrdb_with_action_filter.py
  │   social-transmotion/data/jrdb_all_visual_cues/preprocess_smpl_cvpr/<split>/part_<N>.pt   ← final shards (J=26, action-aware)
  │
  ├─▶ load_jrdb_traj.py     ──▶ data/saved_trajs/jrdb_*_trajs_filterv2.pkl   (only needed for PACER LocoVal training)
  │
  └─▶ create_action_dict.py ──▶ joints2smpl/Pose_to_SMPL/action_dict.json   (consumed by consolidate_jrdb_with_action_filter.py)
```

## 1. Prerequisites

| Item | Where to get it | Notes |
|---|---|---|
| Raw JTA videos + annotations | [JTA-Dataset](https://github.com/fabbrimatteo/JTA-Dataset) | Registration required |
| Raw JRDB sequences | [jrdb.erc.monash.edu](https://jrdb.erc.monash.edu/) | Registration required |
| JRDB-Act 3D labels | [JRDB Activity](https://jrdb.erc.monash.edu/dataset/activity) `train_dataset_with_activity/labels/labels_3d/` | Per-scene `<scene>.json` with a `labels` key (`label_id`, `box`, `social_activity`, ...) consumed by `load_jrdb_3dpose.py` |
| JRDB-Act action labels | [JRDB Activity](https://jrdb.erc.monash.edu/dataset/activity) | `labels_2d_stitched/` is the input we consume |
| SMPL body models v1.1.0 | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) | Place per the main [README §SMPL body models](../README.md#smpl-body-models) |
| Working `uv sync` env | This repo | See main [README §Installation](../README.md#installation) |

All commands below assume the repository root unless `cd` is shown.

## 2. JTA Pipeline

### 2.1 Build the Social-Transmotion `.pt` shards (J=49)

Upstream Social-Transmotion ships per-sequence `.ndjson` files in a [GitHub Release](https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data); EmLoco's `dataset_jta.py` then chunks them into `preprocess/*.pt` shards on first instantiation with `preprocessed=False`.

```bash
# 1. Stage upstream's per-sequence ndjsons under social-transmotion/data/jta_all_visual_cues/
#    (unzip releases.zip from the upstream tag above; place its `jta/data/{train,val,test}/`
#     subdir at social-transmotion/data/jta_all_visual_cues/{train,val,test}/)

# 2. Trigger one-time chunking (writes 5000-track .pt shards):
cd social-transmotion
python -c "
from dataset_jta import JtaAllVisualCuesDataset
for split in ['train', 'val', 'test']:
    JtaAllVisualCuesDataset(split=split, track_size=21, track_cutoff=9,
        segmented=True, add_flips=False, preprocessed=False)
"
cd ..
# Output: social-transmotion/data/jta_all_visual_cues/preprocess/{train,val,test}/part_<N>.pt
```

### 2.2 Extract per-pedestrian 3D pose (`original_pose/`)

Required input for the SMPL fit step (2.3).

```bash
cd social-transmotion
python load_jta_3dpose.py --split train
python load_jta_3dpose.py --split val
python load_jta_3dpose.py --split test
cd ..
# Output: social-transmotion/data/jta_all_visual_cues/original_pose/<split>/jtapose_<split>_part<N>.pkl
```

### 2.3 Per-pedestrian SMPL fitting

```bash
cd joints2smpl/Pose_to_SMPL
for split in train val test; do
  python fit/tools/main.py --dataset_name JTA --save_params --exp cross_fixed \
    --dataset_path ../../social-transmotion/data/jta_all_visual_cues/original_pose/$split/
done
cd ../..
# Output: joints2smpl/Pose_to_SMPL/fit/output/JTA_cross_fixed/jtapose_<split>_part<N>/batch*_params.pkl
```

`--dataset_path` overrides `DATASET.PATH` in `fit/configs/JTA.json`, which otherwise points at a single split.

> [!Tip]
> SMPL fitting is heavy (≈10 GPU-hours for JTA `train`). Sanity-check on a single part first with `--part_start 0 --part_end 0` before launching the full run.

### 2.4 Consolidate into final J=49 `.pt` shards

```bash
cd joints2smpl/Pose_to_SMPL
python fit/tools/save_jta_smplpose.py
cd ../..
# Output: social-transmotion/data/jta_all_visual_cues/preprocess_smpl_cvpr/<split>/part_<N>.pt
```

### 2.5 Verify

```bash
cd social-transmotion
python evaluate_jta.py --exp_name jta_ours --modality traj+all
# Expected: ADE ≈ 0.951, FDE ≈ 1.921
```

### 2.6 (Only for PACER training) PACER trajectory cache

Required only if you plan to (re)train the LocoVal value network in PACER (see [main README §C](../README.md#optional-re-train-the-locoval-value-function-in-isaac-gym)).

```bash
cd social-transmotion
python load_jta_traj.py --cfg configs/jta_all_visual_cues.yaml
cd ..
# Output: social-transmotion/data/saved_trajs/jta_all_visual_cues_{train,val,test}_trajs.pkl
```

## 3. JRDB Pipeline

### 3.1 Build the Social-Transmotion `.pt` shards

Stage upstream's per-scene `.ndjson` files (from the same [Social-Transmotion releases.zip](https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data); its `jrdb/data/{train,val,test}/` subdir) under `social-transmotion/data/jrdb_2dbox/`, then trigger `Jrdb2dboxDataset(preprocessed=False)` once to chunk them into `preprocess/{split}/part_<N>.pt` shards.

```bash
cd social-transmotion
python -c "
from dataset_jrdb import Jrdb2dboxDataset
for split in ['train', 'val', 'test']:
    Jrdb2dboxDataset(name='jrdb_all_visual_cues', split=split, track_size=21,
        track_cutoff=9, segmented=True, add_flips=False, preprocessed=False)
"
cd ..
# Output: social-transmotion/data/jrdb_2dbox/preprocess/{train,val,test}/part_<N>.pt
```

### 3.2 Extract per-pedestrian 3D pose (`original_pose/`)

Required input for the SMPL fit step (3.4). Point `--hst_dir` at JRDB-Act's `labels_3d/` directory.

```bash
cd social-transmotion
LABELS_3D=<path-to-jrdb-act>/train_dataset_with_activity/labels/labels_3d
python load_jrdb_3dpose.py --split train --hst_dir "$LABELS_3D"
python load_jrdb_3dpose.py --split val   --hst_dir "$LABELS_3D"
python load_jrdb_3dpose.py --split test  --hst_dir "$LABELS_3D"
cd ..
# Output: social-transmotion/data/jrdb_all_visual_cues/original_pose/<split>/jrdbpose_<split>_part<N>.pkl
```

### 3.3 Build the action-label dictionary

```bash
cd joints2smpl/Pose_to_SMPL
python fit/tools/create_action_dict.py \
    --action_dir <path-to-JRDB-Act>/train_dataset_with_activity/labels/labels_2d_stitched
cd ../..
# Output: joints2smpl/Pose_to_SMPL/action_dict.json   (written to the CWD)
```

> [!Note]
> If you do not have JRDB-Act locally, you can use the prebuilt `action_dict.json` shipped in the HF release (`.assets/action_dict.json`).

### 3.4 Per-pedestrian SMPL fitting

```bash
cd joints2smpl/Pose_to_SMPL
for split in train val test; do
  python fit/tools/main.py --dataset_name JRDB --save_params --exp cross_fixed \
    --dataset_path ../../social-transmotion/data/jrdb_all_visual_cues/original_pose/$split/
done
cd ../..
# Output: joints2smpl/Pose_to_SMPL/fit/output/JRDB_cross_fixed/jrdbpose_<split>_part<N>/batch*_params.pkl
```

### 3.5 Consolidate into final J=26 `.pt` shards (action-aware)

```bash
cd joints2smpl/Pose_to_SMPL
python fit/tools/consolidate_jrdb_with_action_filter.py
cd ../..
# Output: social-transmotion/data/jrdb_all_visual_cues/preprocess_smpl_cvpr/<split>/part_<N>.pt
```

### 3.6 Verify

```bash
cd social-transmotion
python evaluate_jrdb.py --exp_name jrdb_ours --modality traj+all
```

### 3.7 (Only for PACER training) PACER trajectory cache

```bash
cd social-transmotion
python load_jrdb_traj.py --cfg configs/jrdb_all_visual_cues.yaml
cd ..
# Output: social-transmotion/data/saved_trajs/jrdb_all_visual_cues_{train,val,test}_trajs_filterv2.pkl
```

## 4. Script Reference

| Script | Purpose | Input | Output |
|---|---|---|---|
| `social-transmotion/load_jta_3dpose.py` | Extract per-pedestrian 3D pose for SMPL fit | preprocess shards | `data/jta_all_visual_cues/original_pose/<split>/*.pkl` |
| `social-transmotion/load_jrdb_3dpose.py` | Extract per-pedestrian 3D pose for SMPL fit | preprocess shards | `data/jrdb_all_visual_cues/original_pose/<split>/*.pkl` |
| `social-transmotion/load_jta_traj.py` | PACER trajectory cache (JTA) | preprocess shards | `data/saved_trajs/jta_*_trajs.pkl` |
| `social-transmotion/load_jrdb_traj.py` | PACER trajectory cache (JRDB) | preprocess shards | `data/saved_trajs/jrdb_*_trajs_filterv2.pkl` |
| `joints2smpl/Pose_to_SMPL/fit/tools/main.py` | Per-pedestrian SMPL fitting | `original_pose/` + SMPL body model | `fit/output/<dataset>_cross_fixed/...` |
| `joints2smpl/Pose_to_SMPL/fit/tools/save_jta_smplpose.py` | Consolidate JTA SMPL fits → J=49 `.pt` shards | `fit/output/JTA_cross_fixed/` + JTA preprocess shards | `preprocess_smpl_cvpr/jta_*/...` |
| `joints2smpl/Pose_to_SMPL/fit/tools/consolidate_jrdb_with_action_filter.py` | Consolidate JRDB SMPL fits → J=26 `.pt` shards (action-aware) | `fit/output/JRDB_cross_fixed/` + JRDB preprocess shards + `action_dict.json` | `preprocess_smpl_cvpr/jrdb_*/...` |
| `joints2smpl/Pose_to_SMPL/fit/tools/create_action_dict.py` | JRDB-Act labels → `action_dict.json` | JRDB-Act `labels_2d_stitched/` | `joints2smpl/Pose_to_SMPL/action_dict.json` |
| `joints2smpl/Pose_to_SMPL/fit/tools/cross_handler.py` | Cross-validation split helper (used internally) | — | — |

## 5. Output Format Reference

### JTA — `preprocess_smpl_cvpr/jta_*/<split>/part_<N>.pt`

- Type: `list[scene] of list[person] of (joints[T, 49, 4], mask[T, 49])`
- Token layout (J=49): `traj (1) | 3dbb (1) | 2dbb (1) | SMPL Jtr (24) | 2dpose (22)`
- The 4 channels size the array to the widest token. Only the two bbox tokens use all 4; trajectory uses the first 2, SMPL joints the first 3, 2D pose the first 2. Unused channels are zero.

### JRDB — `preprocess_smpl_cvpr/jrdb_*/<split>/part_<N>.pt`

- Type: `list[scene] of list[person] of (joints[T, 26, 4], mask[T, 26], meta)`
- Token layout (J=26): `traj + 2dbb (2) | SMPL Jtr (24)`
- Pose tokens are **NaN-filled** on frames where the action label is missing (action-aware filter). `mask` is all-1s; the EmLoco loss skips frames by NaN-detection on values.
- `meta` carries the per-person scene / pedestrian-id metadata used by the loader.