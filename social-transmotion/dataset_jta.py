import os

import torch
from torch.nn.utils.rnn import pad_sequence

import pickle
import tqdm

from utils.data import load_data_jta_all_visual_cues

def collate_batch(batch):
    joints_list = []
    masks_list = []
    num_people_list = []
    for joints, masks in batch:

        joints_list.append(joints)
        masks_list.append(masks)
        num_people_list.append(torch.zeros(joints.shape[0]))

    joints = pad_sequence(joints_list, batch_first=True)
    masks = pad_sequence(masks_list, batch_first=True)
    padding_mask = pad_sequence(num_people_list, batch_first=True, padding_value=1).bool()

    return joints, masks, padding_mask

def batch_process_coords(coords, masks, padding_mask, config, modality_selection='traj+all', training=False, multiperson=True):
    # import pdb; pdb.set_trace()
    joints = coords.to(config["DEVICE"])
    masks = masks.to(config["DEVICE"])
    # needed_joints = [0] + list(range(3, 27))
    # needed_joints = [0] + list(range(3, 7)) + list(range(8, 11)) + list(range(15, 27))
    # joints = joints[:, :, :, needed_joints, :]
    # masks = masks[:, :, :, needed_joints]

    in_F = config["TRAIN"]["input_track_size"]

    in_joints_pelvis = joints[:,:, (in_F-1):in_F, 0:1, :].clone()
    in_joints_pelvis_last = joints[:,:, (in_F-2):(in_F-1), 0:1, :].clone()

    joints[:,:,:,0] = joints[:,:,:,0] - joints[:,0:1, (in_F-1):in_F, 0] # normalize pelvis

    joints[:,:,:,1:3] = joints[:,:,:,1:3] - joints[:,:,(in_F-1):in_F,1:3] # normalize other joints
    joints[:,:,:,3:27] = joints[:,:,:,3:27] - joints[:,:,(in_F-1):in_F,3:27] # normalize 3d poses #nonnorm or not
    joints[:,:,:,27:] = joints[:,:,:,27:] - joints[:,:,(in_F-1):in_F,27:] # normalize 2d poses

    # select traj and 3d pose (0, 3:27) only, drop other modalities (1,2,27:)

    B, N, F, J, K = joints.shape

    if not training:
        if modality_selection == 'traj+all':
            pass
        elif modality_selection=='traj':
            joints[:,:,:,1:]=0 # discard all other modalities
        elif modality_selection=='traj+2dbox':
            joints[:,:,:,1]=0
            joints[:,:,:,3:]=0
        elif modality_selection == 'traj+3dpose':
            joints[:,:,:,1:3]=0
            joints[:,:,:,27:]=0 # default: 25:
        elif modality_selection == 'traj+2dpose':
            joints[:,:,:,1:27]=0 # default: 1:25
        elif modality_selection == 'traj+3dpose+3dbox':
            joints[:,:,:,2]=0
            joints[:,:,:,27:]=0 # default: 25
        elif modality_selection == 'traj+2dpose+3dpose':
            joints[:,:,:,1:3]=0
        else:
            print('modality error')
            exit()
    joints = joints.transpose(1, 2).reshape(B, F, N*J, K)
    in_joints_pelvis = in_joints_pelvis.reshape(B, 1, N, K)
    in_joints_pelvis_last = in_joints_pelvis_last.reshape(B, 1, N, K)
    masks = masks.transpose(1, 2).reshape(B, F, N*J)

    in_F, out_F = config["TRAIN"]["input_track_size"], config["TRAIN"]["output_track_size"]
    in_joints = joints[:,:in_F].float()
    out_joints = joints[:,in_F:in_F+out_F].float()
    in_masks = masks[:,:in_F].float()
    out_masks = masks[:,in_F:in_F+out_F].float()


    return in_joints, in_masks, out_joints, out_masks, padding_mask.float()

class MultiPersonTrajPoseDataset(torch.utils.data.Dataset):

    def __init__(self, name, split="train", track_size=21, track_cutoff=9, segmented=True,
                 add_flips=False, frequency=1, preprocessed=False):

        self.name = name
        self.split = split
        self.track_size = track_size
        self.track_cutoff = track_cutoff
        self.frequency = frequency

        if preprocessed:
            self.datalist = []
            load_bar = tqdm.tqdm(os.listdir(f"data/{self.name}/preprocess_smpl/{self.split}"))
            for part, file in enumerate(load_bar):
                with open(f"data/{self.name}/preprocess_smpl/{self.split}/{file}", 'rb') as f:
                    self.datalist += pickle.load(f)
                    load_bar.set_description(f"Loaded {len(self.datalist)} tracks")
                # if part == 0:
                #     break
        else:
            self.initialize()

    def load_data(self):
        raise NotImplementedError("Dataset load_data() method is not implemented.")

    def initialize(self):
        self.load_data()
        # make dir for preprocessed data
        if not os.path.exists(f"data/{self.name}/preprocess/{self.split}"):
            os.makedirs(f"data/{self.name}/preprocess/{self.split}")
        all_tracks = []
        part_tracks = []
        part = 0
        for scene in self.datalist:
            for seg, j in enumerate(range(0, len(scene[0][0]) - self.track_size * self.frequency + 1, self.track_size)):
                people = []
                for person in scene:
                    start_idx = j
                    end_idx = start_idx + self.track_size * self.frequency
                    J_3D_real, J_3D_mask = person[0][start_idx:end_idx:self.frequency], person[1][start_idx:end_idx:self.frequency]
                    people.append((J_3D_real, J_3D_mask))
                all_tracks.append(people)
                part_tracks.append(people)
                if (len(part_tracks) >= 5000):
                    with open(f"data/{self.name}/preprocess/{self.split}/part_{part}.pkl", 'wb') as f:
                        pickle.dump(part_tracks, f)
                    print(f"Processed {len(all_tracks)} tracks")
                    part_tracks = []
                    part += 1

        self.datalist = all_tracks
        with open(f"data/{self.name}/preprocess/{self.split}/part_{part}.pkl", 'wb') as f:
            pickle.dump(part_tracks, f)
        print(f"Processed {len(all_tracks)} tracks")


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        scene = self.datalist[idx]

        J_3D_real = torch.stack([s[0] for s in scene])
        J_3D_mask = torch.stack([s[1] for s in scene])

        return J_3D_real, J_3D_mask


class JtaAllVisualCuesDataset(MultiPersonTrajPoseDataset):
    def __init__(self, **args):
        super(JtaAllVisualCuesDataset, self).__init__("jta_all_visual_cues", frequency=1, **args)

    def load_data(self):

        self.data = load_data_jta_all_visual_cues(split=self.split)
        self.datalist = []
        for scene in self.data:
            joints, mask = scene
            people=[]
            for n in range(len(joints)):
                people.append((torch.from_numpy(joints[n]),torch.from_numpy(mask[n])))

            self.datalist.append(people)


def create_dataset(dataset_name, logger, **args):
    if logger is not None:
        logger.info("Loading dataset " + dataset_name)

    if dataset_name == 'jta_all_visual_cues':
        dataset = JtaAllVisualCuesDataset(**args)
    else:
        raise ValueError(f"Dataset with name '{dataset_name}' not found.")

    return dataset


def get_datasets(datasets_list, config, logger):

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    preprocessed = config['DATA']['preprocessed']
    datasets = []
    for dataset_name in datasets_list:
        datasets.append(create_dataset(dataset_name, logger, split="train", track_size=(in_F+out_F), track_cutoff=in_F, preprocessed=preprocessed))
    return datasets