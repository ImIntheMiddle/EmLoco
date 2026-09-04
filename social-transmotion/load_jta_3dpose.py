"""load 3d pose from JTA dataset, and save them as pkl files"""

import os
import pickle
import argparse
import numpy as np
import torch
import tqdm


def load_jta_3dpose(opt):
    datalist = []
    dataset_name = "jta_all_visual_cues"
    preprocess_dir = f"data/{dataset_name}/preprocess/{opt.split}"
    # Dispatch by extension: dataset_jta.initialize() now writes .pt, but older runs
    # may have left .pkl shards behind. Prefer .pt when both exist for
    # the same part to avoid double-loading.
    files_pt = {f for f in os.listdir(preprocess_dir) if f.endswith(".pt")}
    files_pkl = {
        f
        for f in os.listdir(preprocess_dir)
        if f.endswith(".pkl") and f[:-4] + ".pt" not in files_pt
    }
    files = sorted(files_pt | files_pkl, key=lambda f: int(f.rsplit("part_", 1)[1].split(".")[0]))
    load_bar = tqdm.tqdm(files)
    for file in load_bar:
        path = f"{preprocess_dir}/{file}"
        # Carry the shard's own part number: downstream pairs
        # <prefix>pose_<split>_part<N> back with preprocess/<split>/part_<N>.
        part = int(file.rsplit("part_", 1)[1].split(".")[0])
        if file.endswith(".pt"):
            datalist.append((part, torch.load(path, weights_only=False)))
        else:
            with open(path, "rb") as f:
                datalist.append((part, pickle.load(f)))
        load_bar.set_description(f"Loaded {len(datalist)} tracks")
    return datalist


def main(opt):
    datalist = load_jta_3dpose(opt)
    save_dir = f"data/jta_all_visual_cues/original_pose/{opt.split}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for part, data in datalist:
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
        posearray = np.array(poselist)  # (n, 21, 22, 3)
        posedict = {"keylist": keylist, "posearray": posearray}

        with open(f"{save_dir}/jtapose_{opt.split}_part{part}.pkl", "wb") as f:
            pickle.dump(posedict, f)
        print(f"Processed {len(keylist)} seqs")
        print(f"Saved to {save_dir}/jtapose_{opt.split}_part{part}.pkl")
        print(f"Part {part} finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to use. one of [train, test, val]",
    )
    opt = parser.parse_args()
    main(opt)
