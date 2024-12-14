import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.dataloader_image import TrajectoryDataset
from utils.homography import world2image


set_names = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
output_dir = './output/'
dataset_dir = './datasets_visualize/'
figsize = (10, 8)
linew = 3

# import pickle
# pickle_names = ['./pred_traj_dump/MC-SGCN-{}.pkl',
#                 './pred_traj_dump/QMC-SGCN-{}.pkl',
#                 './pred_traj_dump/NPSN-SGCN-{}.pkl']
# pickle_color = ['deepskyblue', 'darkorange', 'lawngreen']

# import seaborn as sns
# color_tab10 = np.array(sns.color_palette('tab10'))
# color_deep = np.array(sns.color_palette('deep'))


if __name__ == '__main__':
    for data_name in set_names:
        os.makedirs(output_dir + data_name, exist_ok=True)
        dataset = TrajectoryDataset(dataset_dir + data_name)

        # pickled_data_all = []
        # for name in pickle_names:
        #     with open(name.format(data_name), 'rb') as f:
        #         pickled_data_all.append(pickle.load(f))

        plt.figure(figsize=figsize, dpi=100)

        for idx, (obs_traj, pred_traj, frame, image) in enumerate(tqdm(dataset)):
            all_traj = np.concatenate([obs_traj, pred_traj], axis=2).transpose([0, 2, 1])  # NTC

            plt.gca().imshow(image, origin='lower')

            for n in range(len(all_traj)):
                gt_traj = world2image(all_traj[n], H=dataset.h_mat)
                plt.plot(gt_traj[:8, 0], gt_traj[:8, 1], linestyle='-', color='darkviolet', linewidth=linew)
                plt.plot(gt_traj[7:, 0], gt_traj[7:, 1], linestyle='-', color='crimson', linewidth=linew)
                
                # for i, data in enumerate(pickled_data_all):
                #     traj = dataset.image_to_world(data[idx][n]) + 0.5
                #     traj = np.concatenate([gt_traj[:8], traj], axis=0)
                #     plt.plot(traj[7:, 0], traj[7:, 1], linestyle='-', color=pickle_color[i], linewidth=linew, zorder=10)

            plt.axis('off')
            plt.xlim(0, image.shape[1])
            plt.ylim(0, image.shape[0])
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('{}/traj_{}_{}.png'.format(output_dir + data_name, idx, frame), bbox_inches="tight", pad_inches=0, dpi=200)
            plt.savefig('{}/traj_{}_{}.svg'.format(output_dir + data_name, idx, frame), bbox_inches="tight", pad_inches=0, dpi=200)
            plt.cla()
            plt.clf()
