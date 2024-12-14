"""load 3d pose from JRDB dataset, and save them as pkl files"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

from utils.utils import path_to_data
from utils.trajnetplusplustools import Reader_jta_all_visual_cues, Reader_jrdb_2dbox
from utils.data import load_data_jta_all_visual_cues, prepare_data

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_jrdb_data(opt):
    datalist = []
    dataset_name = "jrdb_2dbox"
    load_bar = tqdm.tqdm(os.listdir(f"data/{dataset_name}/preprocess/{opt.split}"))
    for part, file in enumerate(load_bar):
        with open(f"data/{dataset_name}/preprocess/{opt.split}/{file}", 'rb') as f:
            datalist.append(pickle.load(f))
            # print(f"Loaded {len(self.datalist)} tracks")
            load_bar.set_description(f"Loaded {len(datalist)} parts")
        # if part == 0:
        #     break
    return datalist

def load_jrdb_3dpose(hst_dir, sequence):
    posedict = {}
    for scene_name in sequence:
        pose_data_path = f"{hst_dir}/{scene_name}.json"
        scene_data = read_json(pose_data_path)
        posedict[scene_name] = scene_data['labels']
    return posedict

def plot_pose(pose, plot_count):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    bone_list = [[0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16], [15, 17], [16, 18], [17, 19], [18, 20], [15, 19], [16, 20], [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32]]
    for i in range(0, 33):
        x = pose[i*3]
        y = pose[i*3+1]
        z = pose[i*3+2]
        ax.scatter(x, y, z, c='r', marker='o', s=10)
        ax.text(x, y, z, f'{i}', color='black')
    for bone in bone_list:
        x = [pose[bone[0]*3], pose[bone[1]*3]]
        y = [pose[bone[0]*3+1], pose[bone[1]*3+1]]
        z = [pose[bone[0]*3+2], pose[bone[1]*3+2]]
        ax.plot(x, y, z, c='b')
    # save the plot in the real scale (adjust ticks)
    ax.set_xticks(np.arange(-0.6, 0.6, 0.2))
    ax.set_yticks(np.arange(-0.6, 0.6, 0.2))
    ax.set_zticks(np.arange(-0.8, 0.8, 0.2))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(f'data/jrdb_3dpose/plot/pose_{plot_count}.png')
    print("Plot saved to data/jrdb_3dpose/plot/pose.png")
    plt.close()

def main(opt):
    datalist = load_jrdb_data(opt)

    sequence = {'train': ['bytes-cafe-2019-02-07_0', 'gates-basement-elevators-2019-01-17_1', 'hewlett-packard-intersection-2019-01-24_0', 'huang-lane-2019-02-12_0', 'jordan-hall-2019-04-22_0', 'packard-poster-session-2019-03-20_2', 'stlc-111-2019-04-19_0', 'svl-meeting-gates-2-2019-04-08_0', 'svl-meeting-gates-2-2019-04-08_1', 'tressider-2019-03-16_1'], 'val': ['gates-ai-lab-2019-02-08_0'], 'test': ['packard-poster-session-2019-03-20_1', 'tressider-2019-03-16_0']}
    posedict = load_jrdb_3dpose(opt.hst_dir, sequence[opt.split])

    save_dir = f"data/jrdb_3dpose/original_pose/{opt.split}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # import pdb; pdb.set_trace()
    for part, data in enumerate(datalist):
        poselist = []
        keylist = []
        plot_count = 0
        part_loader = tqdm.tqdm(data)
        part_loader.set_description(f"Part {part}")
        for scene_id, scene in enumerate(part_loader):
            # trajbbox = scene[0][:, 0:3]
            for person_id, person in enumerate(scene):
                scene_name = person[2][0].split('_shift')[0]
                assert scene_name in posedict, f"{scene_name} is not in {posedict.keys()}"
                ped_eigen_id = int(person[2][1][0, 1])
                pose_3d = []
                for frame in person[2][1]:
                    got = False
                    if not np.isnan(frame[0]):
                        pose_key = f"000{int(frame[0])}.jpg"
                        if pose_key in posedict[scene_name]:
                            for pedestrian in posedict[scene_name][pose_key]:
                                if pedestrian['label_id'] == f'pedestrian:{int(frame[1])}':
                                    # import pdb; pdb.set_trace()
                                    pose = pedestrian['keypoints']
                                    if plot_count <= 10:
                                        # plot the first pose with joint number
                                        plot_pose(pose, plot_count)
                                        plot_count += 1
                                    pose= np.array(pose).reshape(-1, 3)
                                    pose_3d.append(pose)
                                    got = True
                                    break
                    if not got:
                        pose = np.zeros((33, 3))
                        pose[:] = np.nan
                        pose_3d.append(pose)

                keylist.append(f"part{part}_scene{scene_id}_person{person_id}")
                poselist.append(np.array(pose_3d))
        # import pdb; pdb.set_trace()
        posearray = np.array(poselist) # (n, 21, 22, 3)
        tosavedict = {'keylist': keylist, 'posearray': posearray}

        with open(f"{save_dir}/jrdbpose_{opt.split}_part{part}.pkl", 'wb') as f:
            pickle.dump(tosavedict, f)
        print(f"Processed {len(keylist)} seqs")
        print(f"Saved to {save_dir}/jrdbpose_{opt.split}_part{part}.pkl")
        print(f"Part {part} finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", help="Split to use. one of [train, test, val]")
    parser.add_argument("--hst_dir", type=str, default="/home/halo/dataset_dl001/halo/HumanSceneTransformer/processed/labels/labels_3d_keypoints_train", help="Directory of the HST dataset.")
    opt = parser.parse_args()
    main(opt)