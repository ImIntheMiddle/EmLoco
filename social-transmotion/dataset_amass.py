import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

import tqdm
import pickle
from matplotlib import pyplot as plt

from utils.data import load_data_jta_all_visual_cues, load_data_jrdb_2dbox, load_data_amass
from dataset_jta import JtaAllVisualCuesDataset

def collate_batch(batch):
    joints_list = []
    masks_list = []
    num_people_list = []
    for joints, masks, in batch:
        joints_list.append(joints)
        masks_list.append(masks)
        num_people_list.append(torch.zeros(joints.shape[0]))

    joints = pad_sequence(joints_list, batch_first=True)
    masks = pad_sequence(masks_list, batch_first=True)
    padding_mask = pad_sequence(num_people_list, batch_first=True, padding_value=1).bool()

    return joints, masks, padding_mask

def batch_process_coords(coords, masks, padding_mask, config, modality_selection='traj+all', training=False, multiperson=True):
    joints = coords.to(config["DEVICE"])
    masks = masks.to(config["DEVICE"])
    # import pdb; pdb.set_trace()

    in_F = config["TRAIN"]["input_track_size"]

    in_joints_pelvis = joints[:,:, (in_F-1):in_F, 0:1, :].clone()
    in_joints_pelvis_last = joints[:,:, (in_F-2):(in_F-1), 0:1, :].clone()
    joints[:,:,:,0] = joints[:,:,:,0] - joints[:,0:1,(in_F-1):in_F,0] # normalize traj to origin for primary person at first frame
    joints[:,:,:,1:,0] *= -1 # flip x axis for pose

    B, N, F, J, K = joints.shape
    if not training:
        if modality_selection=='traj':
            joints[:,:,:,1:]=0
        elif modality_selection == 'traj+all':
            pass
        else:
            print('modality error')
            exit()
    elif 'amass_all_visual_cues' in config['DATA']['train_datasets']:
        # visualize_pose(joints[0,0].clone().cpu(), label='before')
        angles = torch.deg2rad(torch.rand(len(joints))*360)
        joints[:,:,:,0,:3] = getRandomRotatePoseTransform(config, angles)(joints[:,:,:,0,:3].unsqueeze(3)).reshape(B, N, F, 3)
        joints[:,:,:,1:,:3] = getRandomRotatePoseTransform(config, angles)(joints[:,:,:,1:,:3]) # TODO: check for 3d pose case
        # visualize_pose(joints[0,0].clone().cpu(), label='after')
        pass
    else:
        print('dataset error')
        exit()

    # import pdb; pdb.set_trace()
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

def getRandomRotatePoseTransform(config, angles):
    """
    Performs a random rotation about the origin (0, 0, 0)
    """
    def do_rotate(pose_seq):
        # import pdb; pdb.set_trace()
        B, N, F, J, K = pose_seq.shape

        ## rotate around z axis (vertical axis)
        rotation_matrix = torch.zeros(B, 3, 3).to(pose_seq.device)
        rotation_matrix[:,0,0] = torch.cos(angles)
        rotation_matrix[:,0,1] = -torch.sin(angles)
        rotation_matrix[:,1,0] = torch.sin(angles)
        rotation_matrix[:,1,1] = torch.cos(angles)
        rotation_matrix[:,2,2] = 1

        rot_pose = torch.bmm(pose_seq.reshape(B, -1, 3).float(), rotation_matrix)
        rot_pose = rot_pose.reshape(pose_seq.shape)
        return rot_pose

    return transforms.Lambda(lambda x: do_rotate(x))

def visualize_pose(joints, label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    trajectory = joints[:,0,:3]
    init_pose = joints[0,2:,:3]
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label='trajectory')
    ax.scatter(init_pose[:,0], init_pose[:,1], init_pose[:,2], label='init_pose')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 0.5)
    # save
    plt.savefig(f'pose_{label}.png')
    plt.close()

class MultiPersonTrajPoseDataset(torch.utils.data.Dataset):

    def __init__(self, name, split="train", track_size=42, track_cutoff=12, segmented=True,
                 add_flips=False, frequency=1, preprocessed=False):
        self.name = name
        self.split = split
        self.track_size = track_size
        self.track_cutoff = track_cutoff
        self.frequency = frequency

        if preprocessed:
            self.datalist = []
            saved_dir = f"data/{self.name}/preprocess/{self.split}"
            load_bar = tqdm.tqdm(os.listdir(saved_dir), dynamic_ncols=True, leave=False)
            for part, file in enumerate(load_bar):
                with open(os.path.join(saved_dir, file), 'rb') as f:
                    self.datalist += pickle.load(f)
                    # print(f"Loaded {len(self.datalist)} tracks")
                    load_bar.set_description(f"Loaded {len(self.datalist)} tracks")
                # if part == 0:
                    # break
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
        # import pdb; pdb.set_trace()
        for scene in self.datalist:
            for seg, j in enumerate(range(0, len(scene[0][0]) - self.track_size * self.frequency + 1, self.track_size)):
                people = []
                for person in scene:
                    start_idx = j
                    end_idx = start_idx + self.track_size * self.frequency
                    J_3D_real, J_3D_mask = person[0][start_idx:end_idx:self.frequency], person[1][start_idx:end_idx:self.frequency]
                    # import pdb; pdb.set_trace()
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
    def show_meta_info(self, idx):
        info = [s[2] for s in self.datalist[idx]]
        return info
    def __getitem__(self, idx):
        scene = self.datalist[idx]

        J_3D_real = torch.stack([s[0] for s in scene])
        J_3D_mask = torch.stack([s[1] for s in scene])

        return J_3D_real, J_3D_mask

class AmassAllVisualCuesDataset(MultiPersonTrajPoseDataset):
    def __init__(self, name, **args):
        super(AmassAllVisualCuesDataset, self).__init__(name=name, frequency=1, **args)

    def load_data(self):
        self.data = load_data_amass(split=self.split)
        self.datalist = []
        for scene in self.data:
            joints, mask = scene
            people=[]
            people.append((joints, mask))
            self.datalist.append(people)
        # import pdb; pdb.set_trace()

def create_dataset(dataset_name, logger, **args):
    if logger is not None:
        logger.info("Loading dataset " + dataset_name)

    if dataset_name == 'amass_all_visual_cues':
        dataset = AmassAllVisualCuesDataset('amass_all_visual_cues', **args)
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
