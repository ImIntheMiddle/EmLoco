import numpy as np
import torch

def MSE_LOSS(output, target, mask=None):
    # import pdb; pdb.set_trace()
    pred_xy = output[:,:,0,:2]
    gt_xy = target[:,:,0,:2]

    norm = torch.norm(pred_xy - gt_xy, p=2, dim=-1)

    mean_K = torch.mean(norm, dim=-1) # mean of joints (each trajectory)
    mean_B = torch.mean(mean_K) # mean of batch

    return mean_B * 100

def MSE_LOSS_MULTI(output, target, mask=None):
    # import pdb; pdb.set_trace()
    pred_xys = output[:,:,:,:2]
    gt_xy = target[:,:,0,:2]
    gt_xys = gt_xy.unsqueeze(2).repeat(1,1,pred_xys.size(2),1)

    norm = torch.norm(pred_xys - gt_xys, p=2, dim=-1)
    mean_K = torch.mean(norm, dim=1) # mean of joints (each trajectory)
    min_K = torch.min(mean_K, dim=1)[0] # min-loss prediction of each person
    mean_B = torch.mean(min_K) # mean of batch
    return mean_B*100

def MSE_LOSS_LSTM(output, target, batch_size, mask=None):
    pred_len, num_preds, _ = output.size()

    pred_xy = output.reshape(batch_size, pred_len, -1, 2)[:,:,0,:2]
    gt_xy = target.reshape(batch_size, pred_len, -1, 2)[:,:,0,:2]

    norm = torch.norm(pred_xy - gt_xy, p=2, dim=-1)

    mean_K = torch.mean(norm, dim=-1)
    mean_B = torch.mean(mean_K)

    return mean_B*100

def calc_hist_distance(hist1, hist2, bin_edges):
    from pyemd import emd
    bins = np.array(bin_edges)
    bins_dist = np.abs(bins[:, None] - bins[None, :])
    hist_dist = emd(hist1, hist2, bins_dist)
    return hist_dist

def calculate_initial_yaw_error(group_A, group_B):
    """
    Calculate angles between corresponding vectors in two groups using batch matrix multiplication,
    handling zero vectors and vectors with negative components.
    """
    # ベクトルを正規化（ゼロベクトルを除く）
    norm_A = torch.norm(group_A, dim=1, keepdim=True)
    norm_B = torch.norm(group_B, dim=1, keepdim=True)
    normalized_A = torch.where(norm_A > 0, group_A / norm_A, group_A)
    normalized_B = torch.where(norm_B > 0, group_B / norm_B, group_B)

    # 正規化されたベクトルの内積を計算
    dot_products = torch.bmm(normalized_A.view(len(norm_A), 1, 2), normalized_B.view(len(norm_B), 2, 1)).squeeze()

    # ゼロベクトルを含む場合の処理
    angles_radians = torch.acos(dot_products.clamp(-1, 1))  # 数値誤差を考慮してクランプ

    return angles_radians

def calculate_velocity(positions, delta_t=0.4): # 2.5fps
    """
    Calculate velocity vectors from displacement vectors and time interval.
    """
    velocities = np.zeros(len(positions)-1)
    for i in range(len(positions)-1):
        # norm
        velocities[i] = np.linalg.norm((positions[i+1] - positions[i]) / delta_t)
    return velocities

def calculate_acceleration(velocities, delta_t=0.4): # 2.5fps
    """
    Calculate acceleration vectors from velocity vectors and time interval.
    """
    accelerations = np.zeros(len(velocities)-1)
    for i in range(len(velocities)-1):
        accelerations[i] = (velocities[i+1] - velocities[i]) / delta_t
        accelerations[i] = np.abs(accelerations[i])
    return accelerations

def calculate_ang_velocity(positions, delta_t=0.4): # 2.5fps
    """
    Calculate angular velocity from positions and time interval.
    """
    ang_velocities = np.zeros(len(positions)-1)
    for i in range(len(positions)-1):
        ang_velocities[i] = np.arctan2(positions[i+1, 1] - positions[i, 1], positions[i+1, 0] - positions[i, 0]) / delta_t
        ang_velocities[i] = np.abs(ang_velocities[i])
    return ang_velocities

def calculate_ang_acceleration(ang_velocities, delta_t=0.4): # 2.5fps
    """
    Calculate angular acceleration from angular velocity and time interval.
    """
    ang_accelerations = np.zeros(len(ang_velocities)-1)
    for i in range(len(ang_velocities)-1):
        ang_accelerations[i] = (ang_velocities[i+1] - ang_velocities[i]) / delta_t
        ang_accelerations[i] = np.abs(ang_accelerations[i])
    return ang_accelerations

def calculate_chi_distance(gt_primitive, pred_primitive, num_bins=20):
    """
    Calculate Chi distance between two primitives, binning the values to 20 bins.
    """
    chi_square_dict = {}
    for primitive in list(gt_primitive.keys()):
        assert primitive in pred_primitive, f"Primitive {primitive} not found in predicted primitives."
        gt_values = gt_primitive[primitive]
        pred_values = pred_primitive[primitive]

        # get max and min values
        min_value = min(min(gt_values), min(pred_values))
        max_value = max(max(gt_values), max(pred_values))

        # binning the values to 20 bins
        bins = np.linspace(min_value, max_value, num_bins+1)

        # convert to probability density
        gt_hist, _ = np.histogram(gt_values, bins=bins, density=True)
        pred_hist, _ = np.histogram(pred_values, bins=bins, density=True)
        gt_dens = gt_hist * np.diff(bins)
        pred_dens = pred_hist * np.diff(bins)

        chi_square = 0
        for bin in range(num_bins): # culculate chi square distance for each bin and sum them
            if gt_dens[bin] == 0 and pred_dens[bin] == 0:
                continue
            else:
                chi_square += (gt_dens[bin] - pred_dens[bin])**2 / (gt_dens[bin] + pred_dens[bin])
        chi_square_dict[primitive] = chi_square

    return chi_square_dict