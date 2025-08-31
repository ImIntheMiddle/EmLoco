import pickle
import numpy as np
import os
import json
import argparse
import copy
import tqdm

def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_Jtr(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    # import pdb; pdb.set_trace()
    Jtr = np.array(data["Jtr"])
    key_list = data["key_list"]
    return key_list, Jtr

def has_leg_cross(joints: np.ndarray):
    return ((joints[1][0]-joints[2][0]) * (joints[4][0]-joints[5][0]) < 0) and ((joints[1][1]-joints[2][1]) * (joints[4][1]-joints[5][1]) < 0)

def has_shoulder_cross(joints: np.ndarray):
    return ((joints[13][0]-joints[14][0]) * (joints[16][0]-joints[17][0]) < 0) and ((joints[13][1]-joints[14][1]) * (joints[16][1]-joints[17][1]) < 0)

def cross_frames(Jtr: np.ndarray):
    crossed_frames_leg = []
    crossed_frames_shoulder = []
    for frame in range(Jtr.shape[0]): # check crossed joints for each frame
        if has_leg_cross(Jtr[frame]):
            crossed_frames_leg.append(frame)
        if has_shoulder_cross(Jtr[frame]):
            crossed_frames_shoulder.append(frame)
    return crossed_frames_leg, crossed_frames_shoulder

def fix_cross_leg(Jtr: np.ndarray, crossed_frames: list):
    """fix crossed joints (flip the coordinates of the left_hip 1 and right hip 2)"""
    Jtr[crossed_frames, 1], Jtr[crossed_frames, 2] = Jtr[crossed_frames, 2], Jtr[crossed_frames, 1]
    return Jtr

def fix_cross_shoulder(Jtr: np.ndarray, crossed_frames: list):
    """fix crossed joints (flip the coordinates of the left_shoulder 13 and right_shoulder 14)"""
    Jtr[crossed_frames, 13], Jtr[crossed_frames, 14] = Jtr[crossed_frames, 14], Jtr[crossed_frames, 13]
    return Jtr

def cross_detector(args):
    input_dir = os.path.join(args.input_dir, args.dataset_name)
    crossed_counter_leg = 0
    crossed_counter_shoulder = 0
    total_data_num = 0
    for root, dirs, files in os.walk(input_dir):
        print("Processing: ", root)
        if 'picture' in root:
            continue
        output_dir = copy.deepcopy(root).replace(args.dataset_name, args.dataset_name + '_cross_fixed') # save to output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        files_bar = tqdm.tqdm(files, leave=False, dynamic_ncols=True)
        for file in files_bar:
            data = dict()
            # import pdb; pdb.set_trace()
            input_path = os.path.join(root, file)
            files_bar.set_description(f"Processing file: {input_path}, crossed leg: {crossed_counter_leg}/{total_data_num}, crossed shoulder: {crossed_counter_shoulder}/{total_data_num}")
            key_list, Jtr = load_Jtr(input_path) # load pose array of the file
            crossed_frames_leg, crossed_frames_shoulder = cross_frames(Jtr) # detect crossed joints and return indices
            crossed_counter_leg += len(crossed_frames_leg)
            crossed_counter_shoulder += len(crossed_frames_shoulder)
            total_data_num += len(Jtr)
            files_bar.set_description(f"Processing file: {input_path}, crossed leg: {crossed_counter_leg}/{total_data_num}, crossed shoulder: {crossed_counter_shoulder}/{total_data_num}")
            if args.fix:
                Jtr_fixed = fix_cross_leg(Jtr, crossed_frames_leg) # fix crossed joints following the detected indices
                Jtr_fixed = fix_cross_shoulder(Jtr_fixed, crossed_frames_shoulder)
                output_path = os.path.join(output_dir, file)
                data["key_list"] = key_list
                data["Jtr"] = Jtr_fixed
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)
    print("Finished!")
    print(f"Total crossed leg: {crossed_counter_leg}, crossed shoulder: {crossed_counter_shoulder}")
    print(f"cross rate leg: {crossed_counter_leg/total_data_num}, shoulder: {crossed_counter_shoulder/total_data_num}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect cross joints')
    parser.add_argument('--dataset_name', dest='dataset_name',
                        help='select dataset',
                        default='JTA', type=str)
    parser.add_argument('--input_dir', dest='input_dir',
                        help='path of input',
                        default='fit/output', type=str)
    parser.add_argument('--fix', dest='fix', action='store_true',
                        help='fix detected cross joints')
    args = parser.parse_args()
    cross_detector(args)