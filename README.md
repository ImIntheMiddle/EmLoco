<div align="center">
    <img src="overview.png" width="960">
</div>

# 🚶‍➡️[EmLoco](https://iminthemiddle.github.io/EmLoco-Page/#)🏃‍➡️

> [!Note]
> Official implementation of **Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment** (CVPR 2025).
> - Authors: [Hiromu Taketsugu](https://iminthemiddle.github.io/), [Takeru Oba](https://obat2343.wixsite.com/my-site), [Takahiro Maeda](https://meaten.github.io/), [Shohei Nobuhara](https://shohei.nobuhara.org/index.en.html), [Norimichi Ukita](https://www.toyota-ti.ac.jp/Lab/Denshi/iim/ukita/index.html)
> - [Project page](https://iminthemiddle.github.io/EmLoco-Page/#) · [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Taketsugu_Physical_Plausibility-aware_Trajectory_Prediction_via_Locomotion_Embodiment_CVPR_2025_paper.html) · [arXiv](https://arxiv.org/abs/2503.17267) · [YouTube](https://youtu.be/OYLDYinc9DU?si=ztusWEL1M7qB21UI) · [🤗 HF assets](https://huggingface.co/iminthemiddle/EmLoco)

## 📑Abstract
*Humans can predict future human trajectories even from momentary observations by using human pose-related cues. However, previous **Human Trajectory Prediction (HTP)** methods leverage the pose cues implicitly, resulting in implausible predictions. To address this, we propose **Locomotion Embodiment**, a framework that explicitly evaluates the physical plausibility of the predicted trajectory by locomotion generation under the laws of physics. While the plausibility of locomotion is learned with an indifferentiable physics simulator, it is replaced by our differentiable **Locomotion Value function** to train an HTP network in a data-driven manner. In particular, our proposed **Embodied Locomotion loss** is beneficial for efficiently training a stochastic HTP network using multiple heads. Furthermore, the **Locomotion Value filter** is proposed to filter out implausible trajectories at inference. Experiments demonstrate that our method further enhances even the state-of-the-art HTP methods across diverse datasets and problem settings.*

## 🗂️Layout

- `social-transmotion/` — trajectory + pose backbone with EmLoco loss / filter
- `pacer/` — pedestrian animation controller + LocoVal value-network training (Isaac Gym)
- `joints2smpl/` — 3D keypoint → SMPL pose fitting
- `EqMotion/` — alternative backbone (ETH/UCY benchmark)
- `isaacgym/` — NVIDIA Isaac Gym

## ⬇️Installation

Tested on Python 3.8.20 + CUDA 12.1. Requires [`uv`](https://docs.astral.sh/uv/) ≥ 0.4 (and [`pyenv`](https://github.com/pyenv/pyenv) for the Python toolchain).

```bash
git clone https://github.com/ImIntheMiddle/EmLoco
cd EmLoco
uv sync
source .venv/bin/activate    # required: PACER's gymtorch JIT needs `ninja` on PATH
```

### Isaac Gym binaries (only needed for PACER training)

```bash
# Get IsaacGym_Preview_4 from https://developer.nvidia.com/isaac-gym
tar -xf IsaacGym_Preview_4_Package.tar.gz
cp -r IsaacGym_Preview_4_Package/isaacgym/python/isaacgym/_bindings \
      ./isaacgym/python/isaacgym/_bindings
```

### SMPL body models

Register at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) (v1.1.0). Both `pacer/` and `joints2smpl/` load SMPL:

```bash
# Place official SMPL files at:
pacer/data/smpl/{SMPL_NEUTRAL,SMPL_MALE,SMPL_FEMALE}.pkl

# Mirror into joints2smpl's loader path:
for g in NEUTRAL MALE FEMALE; do
  ln -s "$PWD/pacer/data/smpl/SMPL_${g}.pkl" \
        "joints2smpl/Pose_to_SMPL/smplpytorch/native/models/SMPL_${g}.pkl"
done
```

## 🌐Data & Checkpoints

Preprocessed data and `Ours` checkpoints live on 🤗 **[iminthemiddle/EmLoco](https://huggingface.co/iminthemiddle/EmLoco)** (`CC BY-NC 4.0`, research-only):

```bash
pip install -U "huggingface_hub[cli]"
hf download iminthemiddle/EmLoco --local-dir .assets --repo-type model
```

```bash
# Preprocessed shards (~28 GB: JTA J=49 .pt + JRDB J=26 .pkl)
mkdir -p social-transmotion/data/jta_all_visual_cues social-transmotion/data/jrdb_all_visual_cues
ln -s "$PWD/.assets/preprocess_smpl_cvpr/jta_all_visual_cues"  social-transmotion/data/jta_all_visual_cues/preprocess_smpl_cvpr
ln -s "$PWD/.assets/preprocess_smpl_cvpr/jrdb_all_visual_cues" social-transmotion/data/jrdb_all_visual_cues/preprocess_smpl_cvpr

# Ours checkpoints (~76 MB)
mkdir -p social-transmotion/experiments/JTA/jta_ours social-transmotion/experiments/JRDB/jrdb_ours
ln -s "$PWD/.assets/checkpoints/jta_ours"  social-transmotion/experiments/JTA/jta_ours/checkpoints
ln -s "$PWD/.assets/checkpoints/jrdb_ours" social-transmotion/experiments/JRDB/jrdb_ours/checkpoints

# LocoVal value-network checkpoints (~34 KB, used by evaluate_*.py and train_*.py with --valueloss_w)
mkdir -p pacer/output/exp/pacer
ln -s "$PWD/.assets/checkpoints/valuenets/valuenet_realpath_JTA+JRDB_valuenet_00025000.pth"        pacer/output/exp/pacer/
ln -s "$PWD/.assets/checkpoints/valuenets/valuenet_realpath_JTA+JRDB_nopose_valuenet_00025000.pth" pacer/output/exp/pacer/

# Per-action ADE/FDE breakdown (5.5 MB, optional)
ln -s "$PWD/.assets/action_dict.json" joints2smpl/Pose_to_SMPL/action_dict.json
```

> See **[docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md)** for the full data pipeline.

## 🚀Quick Start

> [!Tip]
> Each subproject's scripts run from inside that subproject's directory.

### Evaluate the released Ours checkpoint

```bash
cd social-transmotion

# JTA
python evaluate_jta.py  --exp_name jta_ours  --modality traj+all

# JRDB
python evaluate_jrdb.py --exp_name jrdb_ours --modality traj+all
```

### Train your own model with EmLoco loss

```bash
cd social-transmotion
python train_jta.py  --exp_name jta_my_emloco  --valueloss_w 100
python train_jrdb.py --exp_name jrdb_my_emloco --valueloss_w 150
```

### Visualization

`visualize_pred.py` overlays several experiments in one figure, so it reads the `paths` dict defined near the top of its `__main__` (fill that in first) and the cache written by `evaluate_*.py --vis`:

```bash
cd social-transmotion
python evaluate_jta.py --exp_name jta_ours --modality traj+all --vis
python visualize_pred.py --save_name jta_ours_vis
```

### (Optional) Re-train the LocoVal value function in Isaac Gym

Requires Isaac Gym binaries + SMPL (above).

```bash
# Pre-step 1: PACER sample data (AMASS shapes, standing-upright pose, occlusions) — populates
# pacer/sample_data/{amass_isaac_gender_betas_unique.pkl, amass_isaac_standing_upright_slim.pkl,
#                    amass_copycat_occlusion_v2.pkl} via gdown
cd pacer && bash download_data.sh && cd ..

# Pre-step 2: trajectory caches (PACER uses the same JTA/JRDB trajectories as Social-Transmotion)
cd social-transmotion
python load_jta_traj.py  --cfg configs/jta_all_visual_cues.yaml
python load_jrdb_traj.py --cfg configs/jrdb_all_visual_cues.yaml
cd ..

# Step 1: pretrain locomotion policy
cd pacer
python pacer/run.py --pipeline=gpu --random_heading --init_heading --adjust_root_vel \
    --num_envs 1600 --real_path JTA+JRDB \
    --experiment policy_pretrain --max_iterations 150000

# Step 2: train LocoVal function on top of the pretrained policy
python pacer/run.py --pipeline=gpu --random_heading --num_envs 160 \
    --load_path output/exp/pacer/policy_pretrain_00150000.pth \
    --real_path JTA+JRDB --input_init_pose --input_init_vel \
    --experiment valuenet_train --max_iterations 25000
# Final ckpt: pacer/output/exp/pacer/valuenet_train_valuenet_00025000.pth
```

## 📜License

The code original to this work is released under the MIT License (see [`LICENSE`](LICENSE)).

## 🔍Citation

```bibtex
@InProceedings{EmLoco_CVPR25,
  author    = {Taketsugu, Hiromu and Oba, Takeru and Maeda, Takahiro and Nobuhara, Shohei and Ukita, Norimichi},
  title     = {Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment},
  booktitle = {IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year      = {2025}
}
```

## 🤗Acknowledgements

This work is built on [PACER](https://github.com/nv-tlabs/pacer), [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs), [Social-Transmotion](https://github.com/vita-epfl/social-transmotion), [EqMotion](https://github.com/MediaBrain-SJTU/EqMotion), [JTA-Dataset](https://github.com/fabbrimatteo/JTA-Dataset), [JRDB-Traj](https://github.com/vita-epfl/JRDB-Traj), [Pose to SMPL](https://github.com/Dou-Yiming/Pose_to_SMPL), and [human-scene-transformer](https://github.com/google-research/human-scene-transformer). Huge thanks!
