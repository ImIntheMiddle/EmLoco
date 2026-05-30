import numpy as np

rotate = {
    "HumanAct12": [1.0, -1.0, -1.0],
    "CMU_Mocap": [0.05, 0.05, 0.05],
    "UTD_MHAD": [-1.0, 1.0, -1.0],
    "Human3.6M": [-0.001, -0.001, 0.001],
    "NTU": [1.0, 1.0, -1.0],
    "HAA4D": [1.0, -1.0, -1.0],
    "JTA": [1.0, 1.0, -1.0],
    "JRDB": [-1.0, 1.0, 1.0],
    "VRU": [1.0, 1.0, 1.0],
}


def transform(name, arr):
    """Center each frame on its root joint, then re-orient axes for the SMPL fit.

    JTA  root = spine4 (idx 15); axes Y/Z swapped, Z flipped.
    JRDB root = mid-hip (idx 23,24 mean); X flipped, axes Y/Z swapped.
    VRU  root = mid-hip (idx 7,8 mean);   X flipped, axes Y/Z swapped.
    Others: center on joint 0 and apply per-axis scale from `rotate[name]`.
    """
    if name == "JTA":
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                origin = arr[i][j][15].copy()
                for k in range(arr.shape[2]):
                    arr[i][j][k] -= origin
        z_coords = arr[:, :, :, 2].copy() * -1
        y_coords = arr[:, :, :, 1].copy()
        arr[:, :, :, 1] = z_coords
        arr[:, :, :, 2] = y_coords

    elif name == "JRDB":
        if len(arr.shape) == 1:
            # no data
            return arr
        if len(arr.shape) == 3:
            arr = np.expand_dims(arr, axis=1)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                origin = (arr[i][j][23].copy() + arr[i][j][24].copy()) / 2
                for k in range(arr.shape[2]):
                    arr[i][j][k] -= origin
        z_coords = arr[:, :, :, 2].copy()
        y_coords = arr[:, :, :, 1].copy()
        arr[:, :, :, 0] *= -1
        arr[:, :, :, 1] = z_coords
        arr[:, :, :, 2] = y_coords

    elif name == "VRU":
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                origin = (arr[i][j][7].copy() + arr[i][j][8].copy()) / 2
                for k in range(arr.shape[2]):
                    arr[i][j][k] -= origin
        z_coords = arr[:, :, :, 2].copy()
        y_coords = arr[:, :, :, 1].copy()
        arr[:, :, :, 0] *= -1
        arr[:, :, :, 1] = z_coords
        arr[:, :, :, 2] = y_coords
    else:
        for i in range(arr.shape[0]):
            origin = arr[i][0].copy()
            for j in range(arr.shape[1]):
                arr[i][j] -= origin
                for k in range(3):
                    arr[i][j][k] *= rotate[name][k]
    return arr
