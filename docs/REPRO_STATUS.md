# 📊 Reproducibility Verification Status

Tracks which parts of the public release pipeline have been **actually run** on a fresh clone vs. **only smoke-tested** vs. **never executed**. Updated alongside repo changes; use this as the canonical "what is and isn't proven" list before publishing or citing reproducibility claims.

Legend: ✅ verified end-to-end · 🟡 smoke only (startup / partial range) · 🔴 deferred (never run)

---

## Eval-only path (HF release consumers)

| # | Step | Status | Notes |
|---|---|---|---|
| 1 | `hf download iminthemiddle/EmLoco` | ✅ | 28 GB; symlink setup verified |
| 2 | `evaluate_jta.py --exp_name jta_ours` | ✅ | ADE 0.94994 / FDE 1.91892 (matches paper 0.951 / 1.921) |
| 3 | `evaluate_jrdb.py --exp_name jrdb_ours` | ✅ | ADE 0.36921 / FDE 0.72383 (matches paper 0.369 / 0.724) |
| 4 | `visualize_pred.py` (populated `paths`) | 🟡 | Empty-`paths` SystemExit fix verified; with a populated `paths` dict the script correctly resolves the expected `experiments/<exp>/visualization/3d_plot/test/<modality>/vis_dict_<N>frame.pkl` path. Generating those `vis_dict` files requires a prior eval-with-visualization dump that is not part of the default eval — actual rendered figure never produced end-to-end |

## Raw → preprocessed shards (regeneration path)

| # | Step | Status | Notes |
|---|---|---|---|
| 5 | JTA upstream preprocess (`raw → preprocess/*.pkl`) | 🟡 | Upstream Social-Transmotion does **not** ship a preprocess script — they release per-sequence `.ndjson` files in [`ckpt_data`](https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data). EmLoco's `dataset_jta.py initialize()` chunks them into our `.pkl` shards. Verified by running `JtaAllVisualCuesDataset(preprocessed=False)` on upstream `releases.zip/val/`: 3565 tracks produced. Content differs from the plausibl reference (different per-scene people counts), so byte-equivalence is **not** guaranteed — the pipeline is functional but not byte-locked to the released checkpoint's training set. |
| 6 | JRDB upstream preprocess (`raw + JRDB-Traj → jrdb_2dbox/preprocess/*.pkl`) | 🟡 | Same recipe as #5; upstream `releases.zip/jrdb/data/` ndjsons → `Jrdb2dboxDataset(preprocessed=False)` writes `preprocess/<split>/part_<N>.pkl`. Smoke-tested on the same upstream archive; not byte-locked to the released checkpoint's training set for the same reason as JTA. |
| 7 | `load_jta_3dpose.py` (val) → `original_pose/` | ✅ | Output `(40319, 21, 22, 3)`, identical to plausibl-side ground truth (NaN-aware) |
| 8 | `load_jta_3dpose.py` (train + test) | ✅ | Replayed on the full plausibl preprocess: train 18 shards, test 2 shards |
| 9 | `load_jrdb_3dpose.py` (val, JRDB labels_3d) | ✅ | Output `(4122, 21, 33, 3)`, byte-equivalent to plausibl-side (max diff 0) |
| 10 | `load_jrdb_3dpose.py` (train + test) | ✅ | Replayed on plausibl `labels_3d/`: train 4 shards (89819 seqs), test 2 shards (26594 seqs) |
| 11 | `create_action_dict.py` | ✅ | 13800 frame entries, byte-equivalent to HF `action_dict.json` |
| 12 | `main.py --dataset_name JTA --save_params` | 🟡 | Verified on `val/part_0` for batches 0–9 (loss curve healthy, params written); train (~10 GPU-hours) deferred |
| 13 | `main.py --dataset_name JRDB --save_params` | 🟡 | Verified on `test/part_0` batch 0 partial; same code path as JTA |
| 14 | `save_jta_smplpose.py` (test/part_0) | ✅ | 1.1 GB `.pt`, loader confirms `(scene_count=5000, joints[T,49,4])` |
| 15 | `save_jta_smplpose.py` (train + val) | ✅ | 18 train parts (~22 GB) + 1 val part (812 MB); regenerated `test/part_0.pt` is byte-equivalent to the HF release (`max diff = 0.000000`), confirming the full JTA SMPL consolidation pipeline is correct |
| 16 | `consolidate_jrdb_with_action_filter.py` (test) | ✅ | `.pt` generated; running `evaluate_jrdb.py` against it reproduces ADE 0.36921 / FDE 0.72383 exactly (proves the consolidation is byte-equivalent in expectation) |
| 17 | `consolidate_jrdb_with_action_filter.py` (train + val) | ✅ | 4 train parts (~3 GB, 82.1% pose-filled) + 1 val part (47 MB, 56.2% pose-filled); loader confirms `[T, 26, 4]` shape across 19843 + 496 scenes |
| 18 | `load_jta_traj.py` (all splits) | ✅ | 3 `.pkl` outputs (~18 GB), used downstream by PACER policy_pretrain |
| 19 | `load_jrdb_traj.py` (all splits) | ✅ | 3 `_filterv2.pkl` outputs (~815 MB) |
| 20 | `pacer/download_data.sh` | ✅ | Actual `bash`-run produced the 3 expected `sample_data/*.pkl` files |

## Training (predictor + LocoVal)

| # | Step | Status | Notes |
|---|---|---|---|
| 21 | `train_jta.py --dry-run --valueloss_w 1.0` | 🟡 | 1 of 30 epochs; valuenet loaded, EmLoco loss active, checkpoint saved |
| 22 | `train_jta.py --multi_modal --dry-run` | 🟡 | 1 epoch; `num_modes=20` path exercised, ckpt saved |
| 23 | `train_jrdb.py --dry-run --valueloss_w 1.0` | 🟡 | 1 of 150 epochs; same coverage as JTA |
| 24 | `train_jta.py` full convergence | 🔴 | 30 epochs × ~30 min/epoch ≈ 15 GPU-hours |
| 25 | `train_jrdb.py` full convergence | 🔴 | 150 epochs |
| 26 | `pacer/run.py policy_pretrain` | 🟡 | Env init + first iteration reached; full 150k iter is days of GPU time |
| 27 | `pacer/run.py valuenet_train` | 🟡 | Env init reached; full 25k iter not run |
| 28 | "Self-trained valuenet → used in `train_*.py --valueloss_w 1.0` → eval matches paper" | 🔴 | Full loop never closed; we use the released valuenet for the EmLoco-loss path |

## EqMotion ETH/UCY (alternative backbone)

| # | Step | Status | Notes |
|---|---|---|---|
| 29 | Raw ETH/UCY → `eth_ucy/data/<dataset>/<seq>.txt` | 🔴 | External AgentFormer dump, doc'd in `EqMotion/README.md` |
| 30 | `process_eth_data_diverse.py --subset eth` | ✅ | 4 `.npy` outputs in `eth_ucy/processed_data_diverse/` |
| 31 | `process_eth_data_diverse.py` other subsets | ✅ | All 5 subsets (eth, hotel, univ, zara1, zara2) produced their 4 `.npy` outputs |
| 32 | `main_eth_diverse.py --subset eth --test` | 🟡 | Loads released valuenet ckpt successfully; full training/eval not run |
| 33 | EqMotion full train (60 epochs × 5 subsets) | 🔴 | Hours per subset |

## Optional / experimental scripts

| # | Step | Status | Notes |
|---|---|---|---|
| 34 | `render_mesh.py` | 🟡 | `--help` works after `uv pip install pyrender trimesh` (deps deliberately not in `pyproject.toml`). Never actually rendered an end-to-end mesh (no `.obj` asset shipped) |

---

## How to update this file

- When a `🔴` becomes a `🟡` or `✅`, edit the row in place and add a one-line note describing the verification.
- Add new rows when new scripts join the public surface area.
- Keep the legend keys stable so CI scripts can grep on the status icons.
