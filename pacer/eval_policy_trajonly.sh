#!/bin/bash
# compare the performance of various policy network
# set -x # print the command that is going to be executed
export CUDA_VISIBLE_DEVICES=$1 # set GPU device
POLICY="policy_v4_realpath_JTA+JRDB_00005000" # policy network name
# POLICY_NAME_LIST="policy_v4_realpath_JTA+JRDB_00005000 policy_v4_random_align_00005000 policy_v4_hybrid_align_00005000 policy_v4_random_no_align_00005000 policy_v4_realpath_JTA+JRDB_noalign_00005000" # list of policy network names

VALUE=$2

echo "Real traj w/o alignment"
python pacer/run.py --test --num_envs 1 --epoch -1 --pipeline=cpu --random_heading --experiment ${POLICY} --real_path JTA+JRDB --valuenet_path "output/exp/pacer/valuenet_${VALUE}_valuenet_00025000.pth"

echo "Real traj w/ alignment"
python pacer/run.py --test --num_envs 1 --epoch -1 --pipeline=cpu --init_heading --random_heading --adjust_root_vel --experiment ${POLICY} --real_path JTA+JRDB --valuenet_path "output/exp/pacer/valuenet_${VALUE}_valuenet_00025000.pth"

echo "Random traj w/o alignment"
python pacer/run.py --test --num_envs 1 --epoch -1 --pipeline=cpu --random_heading --experiment ${POLICY} --valuenet_path "output/exp/pacer/valuenet_${VALUE}_valuenet_00025000.pth"

echo "Random traj w/ alignment"
python pacer/run.py --test --num_envs 1 --epoch -1 --pipeline=cpu --init_heading --random_heading --adjust_root_vel --experiment ${POLICY} --valuenet_path "output/exp/pacer/valuenet_${VALUE}_valuenet_00025000.pth"