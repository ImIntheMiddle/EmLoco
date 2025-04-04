# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch

SIM_TIMESTEP = 1.0 / 60.0

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def load_cfg(args):
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    cfg["name"] = args.task
    cfg["headless"] = args.headless

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    logdir = args.logdir
    # Set deterministic mode
    if args.torch_deterministic:
        cfg_train["params"]["torch_deterministic"] = True

    exp_name = cfg_train["params"]["config"]['name']

    if args.experiment != 'Base':
        if args.metadata:
            exp_name = "{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])

            if cfg["task"]["randomize"]:
                exp_name += "_DR"
        else:
            exp_name = args.experiment

    # Override config name
    cfg_train["params"]["config"]['name'] = exp_name

    if args.epoch > 0:
        cfg_train["params"]["load_checkpoint"] = True
        cfg_train["params"]["load_path"] = osp.join(args.network_path,   exp_name + "_" + str(args.epoch).zfill(8) + '.pth')
        args.checkpoint = cfg_train["params"]["load_path"]
    elif args.epoch == -1:
        cfg_train["params"]["load_checkpoint"] = True
        cfg_train["params"]["load_path"] = osp.join(args.network_path,   exp_name  + '.pth')
        args.checkpoint = cfg_train["params"]["load_path"]

    # if args.checkpoint != "Base":
    # cfg_train["params"]["load_path"] = osp.join(args.network_path,   exp_name + "_" + str(args.epoch).zfill(8) + '.pth')

    if args.llc_checkpoint != "":
        cfg_train["params"]["config"]["llc_checkpoint"] = args.llc_checkpoint

    # Set maximum number of training iterations (epochs)
    if args.max_iterations > 0:
        cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

    cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]

    seed = cfg_train["params"].get("seed", -1)
    if args.seed is not None:
        seed = args.seed
    cfg["seed"] = seed
    cfg_train["params"]["seed"] = seed

    cfg["args"] = args

    return cfg, cfg_train, logdir


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu
    

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False):
    custom_parameters = [
        {
            "name": "--test",
            "action": "store_true",
            "default": False,
            "help": "Run trained policy, no training"
        },
        {
            "name": "--debug",
            "action": "store_true",
            "default": False,
            "help": "Debugging, no training and no logging"
        },
        {
            "name":
            "--play",
            "action":
            "store_true",
            "default":
            False,
            "help":
            "Run trained policy, the same as test, can be used only by rl_games RL library"
        },
        {
            "name": "--epoch",
            "type": int,
            "default": 0,
            "help": "Resume training or start testing from a checkpoint"
        },
        {
            "name": "--checkpoint",
            "type": str,
            "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"
        },
        {
            "name": "--headless",
            "type": bool,
            "default": True,
            "help": "Force display off at all times"
        },
        {
            "name":
            "--horovod",
            "action":
            "store_true",
            "default":
            False,
            "help":
            "Use horovod for multi-gpu training, have effect only with rl_games RL library"
        },
        {
            "name":
            "--task",
            "type":
            str,
            "default":
            "HumanoidPedestrianTerrain",
            "help":
            "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"
        },
        {
            "name": "--task_type",
            "type": str,
            "default": "Python",
            "help": "Choose Python or C++"
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"
        },
        {
            "name": "--logdir",
            "type": str,
            "default": "logs/"
        },
        {
            "name":
            "--experiment",
            "type":
            str,
            "default":
            "Base",
            "help":
            "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"
        },
        {
            "name":
            "--metadata",
            "action":
            "store_true",
            "default":
            False,
            "help":
            "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"
        },
        {
            "name": "--cfg_env",
            "type": str,
            "default":
                "pacer/data/cfg/pacer.yaml",
            "help": "Environment configuration file (.yaml)"
        },
        {
            "name": "--cfg_train",
            "type": str,
            "default":
                "pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml",
            "help": "Training configuration file (.yaml)"
        },
        {
            "name": "--motion_file",
            "type": str,
            "default": "data/amass/pkls/amass_run_isaac.pkl",
            "help": "Specify reference motion file"
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 1,
            "help": "Number of environments to create - override config file"
        },
        {
            "name": "--episode_length",
            "type": int,
            "default": 0,
            "help": "Episode length, by default is read from yaml config"
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed"
        },
        {
            "name": "--max_iterations",
            "type": int,
            "default": 0,
            "help": "Set a maximum number of training iterations"
        },
        {
            "name":
            "--horizon_length",
            "type":
            int,
            "default":
            -1,
            "help":
            "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."
        },
        {
            "name":
            "--minibatch_size",
            "type":
            int,
            "default":
            -1,
            "help":
            "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."
        },
        {
            "name": "--randomize",
            "action": "store_true",
            "default": False,
            "help": "Apply physics domain randomization"
        },
        {
            "name":
            "--torch_deterministic",
            "action":
            "store_true",
            "default":
            False,
            "help":
            "Apply additional PyTorch settings for more deterministic behaviour"
        },
        {
            "name": "--network_path",
            "type": str,
            "default": "output/exp/pacer",
            "help": "Specify network output directory"
        },
        {
            "name": "--log_path",
            "type": str,
            "default": "log/",
            "help": "Specify log directory"
        },
        {
            "name":
            "--llc_checkpoint",
            "type":
            str,
            "default":
            "",
            "help":
            "Path to the saved weights for the low-level controller of an HRL agent."
        },
        {
            "name": "--no_log",
            "action": "store_true",
            "default": False,
            "help": "No wandb logging"
        },
        {
            "name": "--resume_str",
            "type": str,
            "default": None,
            "help": "Resuming training from a specific logging instance"
        },
        {
            "name": "--follow",
            "action": "store_true",
            "default": False,
            "help": "Follow Humanoid"
        },
        {
            "name": "--real_mesh",
            "action": "store_true",
            "default": False,
            "help": "load real data mesh"
        },
        {
            "name": "--show_sensors",
            "action": "store_true",
            "default": False,
            "help": "load real data mesh"
        },
        {
            "name": "--real_path",
            "type": str,
            "default": "",
            "help": "use real path from dataset. Specify from ['JTA', 'JRDB', 'JTA+JRDB']"
        },
        {
            "name": "--pred_path",
            "action": "store_true",
            "default": False,
            "help": "use predicted traj from traj_pred_data.pkl"
        },
        {
            "name": "--small_terrain",
            "type": bool,
            "default": True,
            "help": "load real data mesh"
        },
        {
            "name": "--server_mode",
            "action": "store_true",
            "default": False,
            "help": "load real data mesh"
        },
        {
            "name": "--add_proj",
            "action": "store_true",
            "default": False,
            "help": "adding small projectiiles or not"
        },
        {
            "name": "--random_heading",
            "action": "store_true",
            "default": False,
            "help": "sample random heading"
        },
        {
            "name": "--init_heading",
            "action": "store_true",
            "default": False,
            "help": "align initial heading"
        },
        {
            "name": "--heading_inversion",
            "action": "store_true",
            "default": False,
            "help": "invert root heading randomly"
        },
        {
            "name": "--adjust_root_vel",
            "action": "store_true",
            "default": False,
            "help": "adjust trajectory speed to root velocity"
        },
        {
            "name": "--input_init_pose",
            "action": "store_true",
            "default": False,
            "help": "input initial humanoid pose to the value function"
        },
        {
            "name": "--input_init_vel",
            "action": "store_true",
            "default": False,
            "help": "input initial humanoid velocity to the value function"
        },
        {
            "name": "--no_virtual_display",
            "type": bool,
            "default": True,
            "help": "Disable virtual display"
        },
        {
            "name": "--load_path",
            "type": str,
            "default": "",
            "help": "Specify pretrained network path"
        },
        {
            "name": "--valuenet_path",
            "type": str,
            "default": "",
            "help": "Specify value network path"
        },
        {
            "name": "--add_noise",
            "action": "store_true",
            "help": "Add noise to the observation"
        },
        {
            "name": "--vru",
            "action": "store_true",
            "help": "Use VRU dataset-style trajectory"
        },
    ]


    if benchmark:
        custom_parameters += [{"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
                              {"name": "--random_actions", "action": "store_true",
                                  "help": "Run benchmark with random actions instead of inferencing"},
                              {"name": "--bench_len", "type": int, "default": 10,
                                  "help": "Number of timing reports"},
                              {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # allignment with examples
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args
