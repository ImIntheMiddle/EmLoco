"""load 3d pose from JTA dataset, and save them as pkl files"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import tqdm

from utils.utils import path_to_data
from utils.trajnetplusplustools import Reader_jta_all_visual_cues, Reader_jrdb_2dbox
from utils.data import load_data_jta_all_visual_cues, prepare_data


def load_jta_3dpose(opt):
    datalist = []
    dataset_name = "jta_all_visual_cues"
    load_bar = tqdm.tqdm(os.listdir(f"data/{dataset_name}/preprocess/{opt.split}"))
    for part, file in enumerate(load_bar):
        with open(f"data/{dataset_name}/preprocess/{opt.split}/{file}", 'rb') as f:
            datalist.append(pickle.load(f))
            # print(f"Loaded {len(self.datalist)} tracks")
            load_bar.set_description(f"Loaded {len(datalist)} tracks")
        # if part == 0:
        #     break
    return datalist

def main(opt):
    datalist = load_jta_3dpose(opt)
    save_dir = f"data/jta_all_visual_cues/original_pose/{opt.split}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # import pdb; pdb.set_trace()
    for part, data in enumerate(datalist):
        poselist = []
        keylist = []
        part_loader = tqdm.tqdm(data)
        part_loader.set_description(f"Part {part}")
        for scene_id, scene in enumerate(part_loader):
            # trajbbox = scene[0][:, 0:3]
            for person_id, person in enumerate(scene):
                pose_3d = person[0][:, 3:25, 0:3]
                # pose_2d = scene[0][:, 25:47]
                keylist.append(f"part{part}_scene{scene_id}_person{person_id}")
                poselist.append(pose_3d.numpy())
        # import pdb; pdb.set_trace()
        posearray = np.array(poselist) # (n, 21, 22, 3)
        posedict = {'keylist': keylist, 'posearray': posearray}

        with open(f"{save_dir}/jtapose_{opt.split}_part{part}.pkl", 'wb') as f:
            pickle.dump(posedict, f)
        print(f"Processed {len(keylist)} seqs")
        print(f"Saved to {save_dir}/jtapose_{opt.split}_part{part}.pkl")
        print(f"Part {part} finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", help="Split to use. one of [train, test, val]")
    opt = parser.parse_args()
    main(opt)