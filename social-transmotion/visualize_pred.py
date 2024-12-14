import sys
sys.path.append("/misc/dl00/halo/plausibl/pacer/pacer")
import os
import argparse
import torch
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from progress.bar import Bar
from torch.utils.data import DataLoader

from utils.utils import create_logger, load_default_config, load_config
from evaluate_jta import Visualizer_3D

class CompareVis_3D(Visualizer_3D):
    def save_plot(self, gt_xy, preds, past_xy, init_pose, id_b, id_k, ades, values=[[None, None]], names=['Pred'], past_len=9):
        fig = plt.figure()
        fig, ax = self.plot_3d(fig, gt_xy, preds, past_xy, init_pose, id_b, id_k, ades, values, names, past_len=past_len)
        # save plot
        # 余白を入れる
        plt.savefig(os.path.join(self.save_dir, f"batch{id_b}_person{id_k}.png"), bbox_inches="tight")
        # plt.savefig(os.path.join(self.save_dir, f"batch{id_b}_person{id_k}.pdf"), bbox_inches="tight")
        plt.close(fig)

def main(data, save_dir, frame_num):
    # create visualizer
    # import pdb; pdb.set_trace()
    vis = CompareVis_3D(save_dir=save_dir)
    names = [d[0] for d in data]
    # compare the data
    bar = tqdm.tqdm(data[0][1]['data'], desc="Comparing data", dynamic_ncols=True)
    # bar = tqdm.tqdm(data[2][1]['data'], desc="Comparing data", dynamic_ncols=True)
    # import pdb; pdb.set_trace()
    vis_dict = {}
    for id, sample in enumerate(bar):
        gt_xy = sample[0]
        # if trajectory is too short, skip
        # import pdb; pdb.set_trace()
        if (abs(gt_xy[-1])>1).sum() == 0:
            continue
        # add gaussian noise to the ground truth
        # gt_xy = np.array(gt_xy) + (torch.randn_like(torch.Tensor(gt_xy)) * 0.4).numpy()
        past_xy = sample[2]
        # past_xy = np.array(past_xy) + (torch.randn_like(torch.Tensor(past_xy)) * 0.4).numpy()
        init_pose = sample[3].cpu()
        id_b = sample[4]
        id_k = sample[5]
        preds = []
        ades = []
        values = []
        for i in range(len(data)):
            d_i = data[i][1]['data'][id]
            preds.append(d_i[1])
            ades.append(d_i[6])
            values.append([d_i[7], d_i[8]])
        # if (abs(preds[2][-1])>1).sum() == 0: # if our prediction is too short
            # continue

        # if (ades[2]+1 < ades[0]) and (ades[2] < ades[1]) and (ades[2] < 3): # if the ades are in ascending order
        # if (ades[1]+0.5 < ades[0]) and (ades[1] < 3): # if the ades are in ascending order
        # if id_b==6 and id_k==90:
        vis_dict[id] = {}
        vis_dict[id]['gt'] = gt_xy
        vis_dict[id]['past'] = past_xy
        vis_dict[id]['init_pose'] = init_pose
        vis_dict[id]['id_b'] = id_b
        vis_dict[id]['id_k'] = id_k
        vis_dict[id]['preds'] = preds
        vis_dict[id]['ades'] = ades
        vis_dict[id]['values'] =values
        # vis.save_plot(gt_xy, preds, past_xy, init_pose, id_b, id_k, ades, values, names, past_len=frame_num)
        # if (ades[0] < ades[2]) and (ades[1] < ades[2]):
    # save the visualization dictionary
    with open(os.path.join(save_dir, 'vis_dict.pkl'), 'wb') as f:
        pickle.dump(vis_dict, f)
    # sort by ades
    # import pdb; pdb.set_trace()

def use_vis_dict(data, save_dir, frame_num, picked_sample=[]):
    vis = CompareVis_3D(save_dir=save_dir)
    names = [d[0] for d in data]
    initial_iter = True if len(picked_sample)==0 else False
    # open pre-saved vis_dict
    # import pdb; pdb.set_trace()
    # save_dir = os.path.join(args.base_dict, 'JTA_CoRL', f"{frame_num}frame")
    vis_dict = pickle.load(open(os.path.join(save_dir, 'vis_dict.pkl'), 'rb'))
    bar = tqdm.tqdm(vis_dict.keys(), desc="Comparing data", dynamic_ncols=True)
    for id in bar:
        gt_xy = vis_dict[id]['gt']
        past_xy = vis_dict[id]['past']
        init_pose = vis_dict[id]['init_pose']
        init_pose[:,0] = -init_pose[:,0]
        # init_pose[:,1] = -init_pose[:,1]
        id_b = vis_dict[id]['id_b']
        id_k = vis_dict[id]['id_k']
        preds = vis_dict[id]['preds']
        ades = vis_dict[id]['ades']
        values = vis_dict[id]['values']
        # if ((frame_num==1) and (ades[2]+1 < ades[0]) and (ades[2]+0.5 < ades[1]) and (ades[1]+1 < ades[0]) and (ades[2] < 1.2)) or ((frame_num==9) and (ades[2] < 0.5) and (ades[1]<ades[0]) and (ades[2]<ades[0]) and (ades[2]<ades[1])):
        # if ades[1] < ades[0]:
        # if id_b == 4 and id_k == 18:
        # if id_b == 6 and id_k == 11:
            # if initial_iter:
            #     picked_sample.append((id_b, id_k))
            # elif (id_b, id_k) not in picked_sample:
            #     continue
        # import pdb; pdb.set_trace()
        # if (ades[1]+0.5 < ades[0]) and (ades[1] < 3):
        # if (ades[0] < 2) and (max(values[0][0])>0.7) and (min(values[0][0])<0.5):
        # if id_b==3 and id_k==51:
            # print(values[0])
        # if id_b==15 and id_k==99:
        if (ades[0]>ades[1]):
            vis.save_plot(gt_xy, preds, past_xy, init_pose, id_b, id_k, ades, values, names, past_len=frame_num)
    return picked_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--base_dict", type=str, default="./visualization/compare_vis/")
    parser.add_argument("--use_vis_dict", action='store_true')
    parser.add_argument("--frame_num", type=int, default=9)
    args = parser.parse_args()

    # paths = { # PRMUv1
        # '(i) Baseline (Traj)': ['jta_smpl', 'train', 'traj'],
        # '(ii) Baseline (Traj+Pose)': ['jta_smpl', 'train', 'traj+all'],
        # '(iii) Ours (Traj+Pose)': ['valuenet_finetune', 'train', 'traj+all'],
    # }
    # paths = { # CoRL
    #     '(i) Baseline (Traj.)': ['JTA/jta_st_standard', 'traj'],
    #     '(ii) Baseline (All)': ['JTA/jta_st_standard', 'traj+all'],
    #     '(iii) Ours (All)': ['JTA/jta_valuenet_100', 'traj+all'],
    # }
    # paths = { # PRMU
        # '(i) Baseline (Traj.)': ['JRDB/standard_st_v1filter', 'traj'],
        # '(ii) Baseline (Traj. + 3D Pose)': ['JRDB/standard_st_v1filter', 'traj+3dpose'],
        # '(iii) Ours (Traj. + 3D Pose)': ['JRDB/valuenet120_v1filter', 'traj+3dpose'],
    # }
    paths = { # CoRL
        '(i) Social-Trans (Traj.)': ['JRDB/jrdb_standard_v4', 'traj'],
        '(ii) Social-Trans': ['JRDB/jrdb_standard_v4', 'traj+all'],
        '(iii) Ours': ['JRDB/jrdb_discount_1101_value100', 'traj+all'],
    }
    # paths = { # CoRL
        # 'Social-Trans': ['JRDB/jrdb_discount_nonnorm_standard_5heads_1112', 'traj+all'],
        # 'Ours': ['JRDB/jrdb_multimodal_5heads_value100', 'traj+all'],
    # }

    # paths = {
        # '(i) Baseline (All)': ['JTA/jta_multimodal_5heads_standard', 'traj+all/valuenet_realpath_JTA+JRDB_valuenet_00025000'],
        # '(ii) Ours (All)': ['JTA/jta_multimodal_5heads_value100', 'traj+all'],
    # }

    # paths = { # CVPR
    #     '(i) Social-Trans (Traj.)': ['JTA/jta_discount_3dnonnorm_1104_standard', 'traj'],
    #     '(ii) Social-Trans': ['JTA/jta_discount_3dnonnorm_1104_standard', 'traj+all'],
    #     # '(iii) Ours': ['JTA/jta_discount_3dnonnorm_1104_value100', 'traj+all/valuenet_1106_discount_hybrid_full_valuenet_00025000'],
    #     '(iii) Ours': ['JTA/jta_discount_3dnonnorm_1104_value100', 'traj+all'],
    # }

    # paths = { # CVPR
        # 'Social-Trans': ['JTA/jta_discount_nonnorm_standard_5heads_1110', 'traj+all/valuenet_1106_discount_hybrid_full_valuenet_00025000'],
    #     # '(ii) Social-Trans': ['JTA/jta_discount_3dnonnorm_1104_standard', 'traj+all'],
    #     # '(iii) Ours': ['JTA/jta_discount_3dnonnorm_1104_value100', 'traj+all/valuenet_1106_discount_hybrid_full_valuenet_00025000'],
        # 'Ours': ['JTA/jta_discount_nonnorm_value100_5heads_1110', 'traj+all/valuenet_1106_discount_hybrid_full_valuenet_00025000'],
    # }

    picked_sample = [] # initialize picked sample
    for frame_num in [2]:
    # for frame_num in [args.frame_num]:
        print(f"Frame number: {frame_num}")
        data = []
        # load visualization data
        for name in paths.keys():
            if args.use_vis_dict:
                data.append([name, None])
            else:
                exp_name = paths[name][0]
                # split = paths[name][1]
                # modality_selection = paths[name][2]
                modality_selection = paths[name][1]
                # path = os.path.join(f'./experiments/{exp_name}/visualization/3d_plot/{split}/{modality_selection}', 'vis_dict.pkl')
                path = os.path.join(f'./experiments/{exp_name}/visualization/3d_plot/test/{modality_selection}', f'vis_dict_{frame_num}frame.pkl')
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Path {path} does not exist")
                print(f"Loading data for {name}...")
                with open(path, 'rb') as f:
                    data.append([name, pickle.load(f)])

        # compare the data length
        if not args.use_vis_dict:
            for i in range(1, len(data)):
                assert len(data[i][1]) == len(data[i-1][1]), f"Data length mismatch: {len(data[i][1])} != {len(data[i-1][1])}"

        save_dir = os.path.join(args.base_dict, args.save_name, f"{frame_num}frame")
        # import pdb; pdb.set_trace()
        if args.use_vis_dict:
            picked_sample = use_vis_dict(data, save_dir, frame_num, picked_sample=picked_sample)
        else:
            main(data, save_dir, frame_num)
            print(f"Saved to {save_dir}")
            picked_sample = use_vis_dict(data, save_dir, frame_num, picked_sample=picked_sample)