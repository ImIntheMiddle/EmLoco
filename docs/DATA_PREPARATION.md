# 🛠️Data Preparation (Regenerating Preprocessed Shards from Raw)

> [!Note]
> **You usually do not need this doc.** Preprocessed `.pt` / `.pkl` shards are published on the Hugging Face Hub at [🤗 iminthemiddle/EmLoco](https://huggingface.co/iminthemiddle/EmLoco). Follow the *Data & Checkpoints* section of the [main README](../README.md) for the eval-only workflow.
>
> This doc is for users who already hold the **raw** JTA / JRDB datasets and want to rebuild the shards themselves (e.g., to extend the pipeline to a new split, add modalities, or audit preprocessing).

## 0. Overview

```
=== JTA pipeline ===

JTA raw videos + annotations
  │
  ▼  social-transmotion/jta_preprocess.py  (joint normalization, J=49)
social-transmotion/data/jta_all_visual_cues/preprocess/{train,val,test}/part_<N>.pkl
  │
  ├─▶ load_jta_traj.py       ──▶ data/saved_trajs/jta_*_trajs.pkl        (PACER LocoVal trajectory cache)
  ├─▶ load_jta_3dpose.py     ──▶ logs / stats                            (auxiliary 3D-pose stats)
  │
  ▼  joints2smpl/Pose_to_SMPL/fit/tools/main.py --dataset_name JTA --save_params
joints2smpl/Pose_to_SMPL/fit/output/JTA_cross_fixed/jtapose_<split>_part<N>/batch*_params.pkl
  │
  ▼  joints2smpl/Pose_to_SMPL/fit/tools/save_jta_smplpose.py
social-transmotion/data/jta_all_visual_cues/preprocess_smpl_cvpr/{train,val,test}/part_<N>.pt   ← final shards (J=49)


=== JRDB pipeline ===

JRDB raw + JRDB-Act labels
  │
  ▼  (Social-Transmotion preprocessing, see upstream repo)
social-transmotion/data/jrdb_2dbox/preprocess/{train,val,test}/part_<N>.pkl
  │
  ├─▶ load_jrdb_traj.py      ──▶ data/saved_trajs/jrdb_*_trajs_filterv2.pkl   (PACER trajectory cache)
  ├─▶ load_jrdb_3dpose.py    ──▶ logs / stats                                 (auxiliary)
  │
  ▼  joints2smpl/Pose_to_SMPL/fit/tools/create_action_dict.py
joints2smpl/Pose_to_SMPL/action_dict.json
  │
  ▼  joints2smpl/Pose_to_SMPL/fit/tools/main.py --dataset_name JRDB --save_params
joints2smpl/Pose_to_SMPL/fit/output/JRDB_cross_fixed/...
  │
  ▼  joints2smpl/Pose_to_SMPL/fit/tools/consolidate_jrdb_with_action_filter.py
social-transmotion/data/jrdb_all_visual_cues/preprocess_smpl_cvpr/{train,val,test}/part_<N>.pt  ← final shards (J=26)
```

## 1. Prerequisites

| Item | Where to get it | Notes |
|---|---|---|
| Raw JTA videos + annotations | [JTA-Dataset](https://github.com/fabbrimatteo/JTA-Dataset) | Registration required |
| Raw JRDB sequences | [jrdb.erc.monash.edu](https://jrdb.erc.monash.edu/) | Registration required |
| JRDB-Act action labels | [JRDB Activity](https://jrdb.erc.monash.edu/dataset/activity) | `labels_2d_stitched/` is the input we consume |
| SMPL body models v1.1.0 | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) | Place at `pacer/data/smpl/{SMPL_NEUTRAL,SMPL_MALE,SMPL_FEMALE}.pkl` |
| Working `uv sync` env | This repo | See main [README §Installation](../README.md#installation) |

All commands below assume the repository root as CWD unless `cd` is shown.

## 2. JTA Pipeline

### 2.1 Build the Social-Transmotion `.pkl` shards (J=49)

Convert raw JTA into the joint-normalized intermediate consumed by every downstream step. Use the preprocessing script shipped with the Social-Transmotion submodule:

```bash
cd social-transmotion
python jta_preprocess.py   # see upstream Social-Transmotion repo for flags
cd ..
# Output: social-transmotion/data/jta_all_visual_cues/preprocess/{train,val,test}/part_<N>.pkl
```

> [!Note]
> Refer to the [upstream Social-Transmotion repo](https://github.com/vita-epfl/social-transmotion) for the JTA preprocessing recipe; the script in this repo is unchanged from upstream.

### 2.2 PACER trajectory cache

```bash
cd social-transmotion
python load_jta_traj.py --cfg configs/jta_all_visual_cues.yaml
cd ..
# Output: social-transmotion/data/saved_trajs/jta_all_visual_cues_{train,val,test}_trajs.pkl
```

Required only if you plan to (re)train the LocoVal value network in PACER.

### 2.3 (Optional) 3D-pose statistics

```bash
cd social-transmotion
python load_jta_3dpose.py --cfg configs/jta_all_visual_cues.yaml
cd ..
```

Logs per-joint statistics; not consumed downstream but useful for sanity checks.

### 2.4 Per-pedestrian SMPL fitting

```bash
cd joints2smpl/Pose_to_SMPL
python fit/tools/main.py --dataset_name JTA --save_params
# Output: fit/output/JTA_cross_fixed/jtapose_<split>_part<N>/batch*_params.pkl
```

> [!Tip]
> SMPL fitting is heavy (≈10 GPU-hours for JTA `train`). Sanity-check on a single part first with `--part_start 0 --part_end 0` before launching the full run.

### 2.5 Consolidate into final J=49 `.pt` shards

```bash
cd joints2smpl/Pose_to_SMPL
python fit/tools/save_jta_smplpose.py
cd ../..
# Output: social-transmotion/data/jta_all_visual_cues/preprocess_smpl_cvpr/{train,val,test}/part_<N>.pt
```

### 2.6 Verify

```bash
cd social-transmotion
python evaluate_jta.py --exp_name jta_ours --modality traj+all
# Expected: ADE ≈ 0.951, FDE ≈ 1.921
```

## 3. JRDB Pipeline

### 3.1 Build the Social-Transmotion `.pkl` shards

```bash
cd social-transmotion
# See upstream Social-Transmotion / JRDB-Traj repos for the recipe
cd ..
# Output: social-transmotion/data/jrdb_2dbox/preprocess/{train,val,test}/part_<N>.pkl
```

### 3.2 PACER trajectory cache

```bash
cd social-transmotion
python load_jrdb_traj.py --cfg configs/jrdb_all_visual_cues.yaml
cd ..
# Output: social-transmotion/data/saved_trajs/jrdb_all_visual_cues_{train,val,test}_trajs_filterv2.pkl
```

### 3.3 (Optional) 3D-pose statistics

```bash
cd social-transmotion
python load_jrdb_3dpose.py --cfg configs/jrdb_all_visual_cues.yaml
cd ..
```

### 3.4 Build the action-label dictionary

JRDB consolidation is action-aware: pose tokens for frames without an action label are NaN-filled. Build `action_dict.json` from the raw JRDB-Act labels:

```bash
cd joints2smpl/Pose_to_SMPL/fit/tools
python create_action_dict.py
cd ../../../..
# Input : JRDB-Act labels_2d_stitched/ (path configured inside the script)
# Output: joints2smpl/Pose_to_SMPL/action_dict.json
```

> [!Note]
> If you do not have JRDB-Act locally, you can use the prebuilt `action_dict.json` shipped in the HF release (`.assets/action_dict.json`) — symlink it into place as shown in the main README.

### 3.5 Per-pedestrian SMPL fitting

```bash
cd joints2smpl/Pose_to_SMPL
python fit/tools/main.py --dataset_name JRDB --save_params
# Output: fit/output/JRDB_cross_fixed/...
```

### 3.6 Consolidate into final J=26 `.pt` shards (action-aware)

```bash
cd joints2smpl/Pose_to_SMPL
python fit/tools/consolidate_jrdb_with_action_filter.py
cd ../..
# Output: social-transmotion/data/jrdb_all_visual_cues/preprocess_smpl_cvpr/{train,val,test}/part_<N>.pt
```

### 3.7 Verify

```bash
cd social-transmotion
python evaluate_jrdb.py --exp_name jrdb_ours --modality traj+all
# Expected: ADE ≈ 0.369, FDE ≈ 0.724
```

## 4. Script Reference

| Script | Purpose | Input | Output |
|---|---|---|---|
| `social-transmotion/load_jta_traj.py` | PACER trajectory cache (JTA) | `data/jta_all_visual_cues/preprocess/*.pkl` | `data/saved_trajs/jta_*_trajs.pkl` |
| `social-transmotion/load_jrdb_traj.py` | PACER trajectory cache (JRDB) | `data/jrdb_2dbox/preprocess/*.pkl` | `data/saved_trajs/jrdb_*_trajs_filterv2.pkl` |
| `social-transmotion/load_jta_3dpose.py` | 3D-pose statistics (JTA, auxiliary) | preprocess shards | logs / stats |
| `social-transmotion/load_jrdb_3dpose.py` | 3D-pose statistics (JRDB, auxiliary) | preprocess shards | logs / stats |
| `joints2smpl/Pose_to_SMPL/fit/tools/main.py` | Per-pedestrian SMPL fitting | preprocess shards + SMPL body model | `fit/output/<dataset>_cross_fixed/...` |
| `joints2smpl/Pose_to_SMPL/fit/tools/save_jta_smplpose.py` | JTA SMPL fit → J=49 `.pt` shards | `fit/output/JTA_cross_fixed/` + JTA preprocess shards | `preprocess_smpl_cvpr/jta_*/...` |
| `joints2smpl/Pose_to_SMPL/fit/tools/consolidate_jrdb_with_action_filter.py` | JRDB SMPL fit + action filter → J=26 `.pt` shards | `fit/output/JRDB_cross_fixed/` + JRDB preprocess shards + `action_dict.json` | `preprocess_smpl_cvpr/jrdb_*/...` |
| `joints2smpl/Pose_to_SMPL/fit/tools/create_action_dict.py` | JRDB-Act labels → `action_dict.json` | JRDB-Act `labels_2d_stitched/` | `joints2smpl/Pose_to_SMPL/action_dict.json` |
| `joints2smpl/Pose_to_SMPL/fit/tools/cross_handler.py` | Cross-validation split helper (used internally) | — | — |

## 5. Output Format Reference

### JTA — `preprocess_smpl_cvpr/jta_*/{train,val,test}/part_<N>.pt`

- Type: `list[scene] of list[person] of (joints[T, 49, 4], mask[T, 49])`
- Token layout (J=49): `traj + 2dbb (3) | SMPL Jtr (24) | remaining 2dpose (22)`
- Last channel of `joints[..., 3]` carries the per-joint validity flag mirrored in `mask`.

### JRDB — `preprocess_smpl_cvpr/jrdb_*/{train,val,test}/part_<N>.pt`

- Type: `list[scene] of list[person] of (joints[T, 26, 4], mask[T, 26], meta)`
- Token layout (J=26): `traj + 2dbb (2) | SMPL Jtr (24)`
- Pose tokens are **NaN-filled** on frames where the action label is missing (action-aware filter).
- `meta` carries the per-person scene / action metadata used by the loader.

## 6. Troubleshooting & Notes

- **Pickle JRDB shards from HF release.** The HF release ships JRDB shards as `.pkl` while this pipeline writes `.pt`. They are interchangeable for evaluation — `social-transmotion/dataset_jrdb.py` calls `torch.load(..., weights_only=False)`, which transparently loads both.
- **SMPL fitting cost.** Per-pedestrian SMPL fitting in step 2.4 / 3.5 dominates total wall-clock (≈10 GPU-hours for JTA `train`). Always validate on a tiny subset (`--part_start 0 --part_end 0`) before kicking off the full run.
- **Missing JRDB-Act labels.** If you only need to reproduce shard generation but lack JRDB-Act locally, download `.assets/action_dict.json` from the HF release and skip §3.4.
- **PACER caches are optional for shard generation.** Steps 2.2 and 3.2 are only required for LocoVal training (PACER pipeline); skip them if you are only rebuilding Social-Transmotion shards.
- **`HYDRA_FULL_ERROR=1`.** Set this when debugging config-driven scripts to see full tracebacks.
