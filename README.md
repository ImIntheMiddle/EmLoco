# 🚶‍➡️[EmLoco](https://iminthemiddle.github.io/EmLoco-Page/#)🏃‍➡️

> This is an implementation of the paper: **Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment (CVPR2025 main)**
>   - Author: [Hiromu Taketsugu](https://iminthemiddle.github.io/), [Takeru Oba](https://obat2343.wixsite.com/my-site), [Takahiro Maeda](https://meaten.github.io/), [Shohei Nobuhara](https://shohei.nobuhara.org/index.en.html), [Norimichi Ukita](https://www.toyota-ti.ac.jp/Lab/Denshi/iim/ukita/index.html)
>   - [Project page](https://iminthemiddle.github.io/EmLoco-Page/#)
>   - [arXiv](https://arxiv.org/abs/2503.17267)

<div align="center">
    <img src=".github/overview.png", width="960">
</div>

## 📑Abstract
*Humans can predict future human trajectories even from momentary observations by using human pose-related cues. However, previous **Human Trajectory Prediction (HTP)** methods leverage the pose cues implicitly, resulting in implausible predictions. To address this, we propose **Locomotion Embodiment**, a framework that explicitly evaluates the physical plausibility of the predicted trajectory by locomotion generation under the laws of physics. While the plausibility of locomotion is learned with an indifferentiable physics simulator, it is replaced by our differentiable **Locomotion Value function** to train an HTP network in a data-driven manner. In particular, our proposed **Embodied Locomotion loss** is beneficial for efficiently training a stochastic HTP network using multiple heads. Furthermore, the **Locomotion Value filter** is proposed to filter out implausible trajectories at inference. Experiments demonstrate that our method further enhances even the state-of-the-art HTP methods across diverse datasets and problem settings.*

## ⬇️Installation
> **Warning**
> Environment: Python 3.10.7, CUDA 12.1
> 
> Other versions have not been tested.
- Create and activate a virtual environment for this repository.
- Following the command below, please install the required packages:
    ```
    pip install -r requirement.txt
    ```
- Then, you can set up your environment by following:
    ```
    python setup.py build develop --user
    ```
    
## 🌐Downloads
- Please download PoseTrack21 and JRDB-Pose, and place them under the `./data` directory.
    - PoseTrack21: https://github.com/anDoer/PoseTrack21
    - JRDB-Pose: https://jrdb.erc.monash.edu/dataset/pose
- After downloading, you can prepare annotation files as follows (please specify the mode in each script):

<details><summary>PoseTrack21</summary>

```
python ./data/PoseTrack21/make_new_annotation.py
python ./data/PoseTrack21/integrate_new_annotation.py
```
</details>

<details><summary>JRDB-Pose</summary>
    
```
python ./data/jrdb-pose/make_new_annotation.py
python ./data/jrdb-pose/integrate_new_annotation.py
```
</details>

- You can download pretrained models of Human Pose Estimator (HRNet, FastPose and SimpleBaseline) and our Wholebody Auto-Encoder from Releases ``Pretrained models''.
    - Unzip and place the ``pretrained_models'' directory under the root directory of the repository.

## 🚀Quick Start
- Make sure you are in the root directory.
- You can execute **VATL (Video-specific Active Transfer Learning)** by following commands.

<details><summary><bold>VATL on PoseTrack21 using SimpleBaseline</bold></summary>
    
1. **(Optional) Train an initial pose estimator from scratch**
    ```
    python ./scripts/posetrack_train.py --cfg ./configs/posetrack21/{CONFIG_FILE} --exp-id {EXP_ID}
    ```
2. **(Optional) Evaluate the performance of the pre-trained model on train/val/test split**
    ```
    python ./scripts/poseestimatoreval.py --cfg ./configs/posetrack21/{CONFIG_FILE} --exp-id {EXP_ID}
    ```
3. **(Optional) Pre-train the AutoEncoder for WPU (Whole-body Pose Unnaturalness)**
    ```
    python ./scripts/wholebodyAE_train --dataset_type Posetrack21
    ```
4. **Execute Video-specific Active Transfer Learning on test videos**

    > **Warning**
    > Please specify the detailed settings in the shell script if you like.
    ```
    bash ./scripts/run_active_learning.sh ${GPU_ID}
    ```
5. **Evaluate the results of video-specific ATL**

    > **Warning**
    > Please specify the results to summarize in the Python script.
    ```
    python ./scripts/detailed_result.py
    ```
6. **(Optional) Visualize the estimated poses on each ATL cycle**

    > **Warning**
    > Please specify the results to summarize in the Python script.
    ```
    python ./scripts/visualize_result.py
    ```
</details>

## 🔍Citation
**If you found this code useful, please consider citing our work ;D**

```
@InProceedings{EmLoco_CVPR25,
  author       = {Taketsugu, Hiromu and Oba, Takeru and Maeda, Takahiro and Nobuhara, Shohei and Ukita, Norimichi},
  title        = {Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment},
  booktitle    = {IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year         = {2025}}
```

## 🤗Acknowledgement
This project repository builds upon the shoulders of giants.
Huge thanks to these awesome works!
- [JTA-Dataset](https://github.com/fabbrimatteo/JTA-Dataset) and [JRDB-Traj](https://github.com/vita-epfl/JRDB-Traj) for datasets.
- [PACER](https://github.com/nv-tlabs/pacer) and [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) for locomotion generation in physics-simulator.
- [Social-Transmotion](https://github.com/vita-epfl/social-transmotion) and [EqMotion](https://github.com/MediaBrain-SJTU/EqMotion) for trajectory prediction.
- [Pose to SMPL](https://github.com/Dou-Yiming/Pose_to_SMPL) for SMPL pose conversion.
