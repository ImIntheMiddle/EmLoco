"""draw a histogram of the data regarding differences in ADE/FDE against the baseline."""

import os
import argparse
import torch
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle

def main(args, data, save_dict):
    # create visualizer
    # import pdb; pdb.set_trace()
    # compare the data
    bar = tqdm.tqdm(data[0][1]['data'], desc="Comparing data", dynamic_ncols=True)
    # import pdb; pdb.set_trace()
    ade_diffs_base_traj = []
    ade_diffs_base_all = []
    fde_diffs_base_traj = []
    fde_diffs_base_all = []
    for id, sample in enumerate(bar):
        ade_base = data[0][1]['data'][id][6]
        ade_base_all = data[1][1]['data'][id][6]
        ade_ours = data[2][1]['data'][id][6]
        fde_base = data[0][1]['data'][id][9]
        fde_base_all = data[1][1]['data'][id][9]
        fde_ours = data[2][1]['data'][id][9]
        ade_diffs_base_traj.append(ade_base - ade_ours)
        ade_diffs_base_all.append(ade_base_all - ade_ours)
        fde_diffs_base_traj.append(fde_base - fde_ours)
        fde_diffs_base_all.append(fde_base_all - fde_ours)
    # import pdb; pdb.set_trace()

    type =['ADE', 'ADE', 'FDE', 'FDE']
    names = ['traj', 'all', 'traj', 'all']
    for i, diffs in enumerate([ade_diffs_base_traj, ade_diffs_base_all, fde_diffs_base_traj, fde_diffs_base_all]):
        fig = plt.figure()
        plt.hist(diffs, bins=10, alpha=0.8, edgecolor='black')

        mean_value = np.mean(diffs) # 平均値を計算
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
        # plt.xlabel(f"Difference in {type[i]}", fontsize=16)
        # plt.ylabel("Frequency", fontsize=16)
        plt.legend(fontsize=16)
        # 軸のフォントサイズを設定
        plt.tick_params(labelsize=16)
        # plt.text(x=plt.xlim()[0], y=-350, 
        #  s='Worse', fontsize=16, color='black')
        # plt.text(x=plt.xlim()[1]*0.8, y=-350, 
        #  s='Better', fontsize=16, color='black')
        plt.savefig(os.path.join(save_dict, f"{type[i]}histogram_{names[i]}_{args.frame_num}frame.png"))
        plt.savefig(os.path.join(save_dict, f"{type[i]}histogram_{names[i]}_{args.frame_num}frame.pdf"), bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, default="histogram")
    parser.add_argument("--base_dict", type=str, default="./visualization/compare_vis/")
    parser.add_argument("--frame_num", type=int, default=9)
    args = parser.parse_args()

    paths = {
        '(i) Baseline (Traj.)': ['JTA/jta_st_standard', 'traj'],
        '(i) Baseline (All)': ['JTA/jta_st_standard', 'traj+all'],
        '(ii) Ours (All)': ['JTA/jta_valuenet_100', 'traj+all'],
    }

    data = []
    for name in paths.keys():
        exp_name = paths[name][0]
        modality_selection = paths[name][1]
        path = os.path.join(f'./experiments/{exp_name}/visualization/3d_plot/test/{modality_selection}', f'vis_dict_{args.frame_num}frame.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")
        print(f"Loading data for {name}...")
        with open(path, 'rb') as f:
            data.append([name, pickle.load(f)])

    save_dict = os.path.join(args.base_dict, args.save_name)
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)
    print(f"Will save the histogram to {save_dict}")

    main(args, data, save_dict)
