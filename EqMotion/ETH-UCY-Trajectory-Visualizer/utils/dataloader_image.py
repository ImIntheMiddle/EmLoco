import cv2
import os
import math
import numpy as np


class TrajectoryDataset:
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1, delim='\t'):
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.cur_frame_no = 0

        all_files = sorted(os.listdir(self.data_dir + '/test/'))
        all_files = [os.path.join(self.data_dir + '/test/', _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        frame_list = []

        for path in all_files:
            data = self.read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))

                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    num_peds_considered += 1
                if num_peds_considered > min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    seq_list.append(curr_seq[:num_peds_considered])
                    frame_list.append(frames[idx])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        self.obs_traj = seq_list[:, :, :self.obs_len]
        self.pred_traj = seq_list[:, :, self.obs_len:]
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.frame_list = np.array(frame_list, dtype=np.int32)

        with open(self.data_dir + '/H.txt', 'r') as h_mat_file:
            h_mat = h_mat_file.read()
        h_mat = [x.split() for x in h_mat.split('\n')][:3][:3]
        self.h_mat = np.array(h_mat).astype(np.float32)
        # self.h_mat = np.linalg.inv(self.h_mat)

        self.cap = cv2.VideoCapture(self.data_dir + '/video.avi')
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_shape = [width, height, 3]

    def read_file(self, _path, delim):
        delim = delim if delim else self.delim
        data = []
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)

    def get_image_from_frame(self, frame_no):
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame % self.total_frames)
        # Faster method
        while True:
            if self.cur_frame_no == frame_no:
                break
            self.cur_frame_no += 1
            if self.cur_frame_no > frame_no:  # or self.cur_frame_no >= self.total_frames:
                print("Frame resetted")
                self.cur_frame_no = 0
                self.cap = cv2.VideoCapture(self.data_dir + '/video.avi')

            ret, frame = self.cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else np.ones(self.video_shape, dtype=np.uint8)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        frame = self.frame_list[index] + 70 - 5
        out = [self.obs_traj[start:end, :], self.pred_traj[start:end, :],
               self.frame_list[index], self.get_image_from_frame(frame)]
        return out
    