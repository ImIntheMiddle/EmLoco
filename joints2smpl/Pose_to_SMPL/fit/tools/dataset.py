"""pose dataset"""
import os
import torch

from transform import transform
from load import load

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, name, root, file):
        self.name = name
        data = load(name, os.path.join(root, file))
        self.data = torch.from_numpy(transform(name, data['posearray'])).float()
        self.keylist = data['keylist']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        person = self.data[idx] # shape (21, joints, 3)
        return idx, person

    def collate_fn(self, batch):
        idxs, data = zip(*batch)
        keylist = [f"{self.keylist[idx]}_frame{i}" for idx in idxs for i in range(len(data[0]))]
        target = torch.cat(data, dim=0)
        assert target.shape[0] == len(keylist), "keylist and target shape mismatch"
        # make the connected data batch * (21, joints, 3) -> (21*batch, joints, 3)
        return idxs, keylist, target

    def shape(self):
        return self.data.shape