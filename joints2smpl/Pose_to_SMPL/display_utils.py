from xml.parsers.expat import model
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# plt.switch_backend('agg')


def display_model(
        model_info,
        model_faces=None,
        dataset_name=None,
        with_joints=False,
        kintree_table=None,
        ax=None,
        batch_idx=0,
        show=True,
        savepath=None,
        only_joint=False):
    """
    Displays mesh batch_idx in batch of model_info, model_info as returned by
    generate_random_model
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts = model_info['verts'][batch_idx] if 'verts' in model_info else None
    joints = model_info['joints'][batch_idx]
    target = model_info['target'][batch_idx]
    if model_faces is None and verts is not None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.2)
    elif not only_joint:
        model_faces = model_faces.cpu()
        mesh = Poly3DCollection(verts[model_faces], alpha=0.2)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    if with_joints:
        draw_skeleton(joints, kintree_table=kintree_table, ax=ax, dataset_name=dataset_name)
        draw_skeleton(target, kintree_table=kintree_table, ax=ax, use_target=True, dataset_name=dataset_name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    ax.view_init(azim=-90, elev=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if savepath:
        # print('Saving figure at {}.'.format(savepath))
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()
    return ax


def draw_skeleton(joints3D, kintree_table, ax=None, with_numbers=True, use_target=False, dataset_name=None):
    if dataset_name == 'JTA':
        tree = JTA_TREE
        tree_colors = [2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0]
    elif dataset_name == 'JRDB':
        tree = BlazePose_TREE
        tree_colors = [0, 1, 0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
    elif dataset_name == 'VRU':
        tree = VRU_TREE
        tree_colors = [0, 0, 2, 1, 1, 0, 1, 2, 0, 1, 0, 1]
    else:
        raise ValueError(f'Unknown dataset_name: {dataset_name}! Please use JTA or JRDB.')

    if ax is None:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax

    colors = []
    if use_target:
        left_right_mid = ['m', 'y', 'k']
        for c in tree_colors:
            colors += left_right_mid[c]
        for i in range(len(tree)):
            j1 = tree[i][0]
            j2 = tree[i][1]
            ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
                    [joints3D[j1, 1], joints3D[j2, 1]],
                    [joints3D[j1, 2], joints3D[j2, 2]],
                    color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
    else:
        left_right_mid = ['r', 'g', 'b']
        kintree_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]
        for c in kintree_colors:
            colors += left_right_mid[c]
        # For each 24 joint
        for i in range(1, kintree_table.shape[1]):
            j1 = kintree_table[0][i]
            j2 = kintree_table[1][i]
            ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
                    [joints3D[j1, 1], joints3D[j2, 1]],
                    [joints3D[j1, 2], joints3D[j2, 2]],
                    color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
            if with_numbers:
                ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)
    return ax

JTA_TREE = [
        (0, 1),  # head_top -> head_center
        (1, 2),  # head_center -> neck
        (2, 3),  # neck -> right_clavicle
        (3, 4),  # right_clavicle -> right_shoulder
        (4, 5),  # right_shoulder -> right_elbow
        (5, 6),  # right_elbow -> right_wrist
        (2, 7),  # neck -> left_clavicle
        (7, 8),  # left_clavicle -> left_shoulder
        (8, 9),  # left_shoulder -> left_elbow
        (9, 10),  # left_elbow -> left_wrist
        (2, 11),  # neck -> spine0
        (11, 12),  # spine0 -> spine1
        (12, 13),  # spine1 -> spine2
        (13, 14),  # spine2 -> spine3
        (14, 15),  # spine3 -> spine4
        (15, 16),  # spine4 -> right_hip
        (16, 17),  # right_hip -> right_knee
        (17, 18),  # right_knee -> right_ankle
        (15, 19),  # spine4 -> left_hip
        (19, 20),  # left_hip -> left_knee
        (20, 21)  # left_knee -> left_ankle
    ]

BlazePose_TREE = [
        (0, 1), # nose -> left_eye_inner
        (0, 4), # nose -> right_eye_inner
        (1, 2), # left_eye_inner -> left_eye
        (2, 3), # left_eye -> left_eye_outer
        (3, 7), # left_eye_outer -> left_ear
        (4, 5), # right_eye_inner -> right_eye
        (5, 6), # right_eye -> right_eye_outer
        (6, 8), # right_eye_outer -> right_ear
        (9, 10), # left_mouth -> right_mouth
        (11, 12), # left_shoulder -> right_shoulder
        (11, 13), # left_shoulder -> left_elbow
        (11, 23), # left_shoulder -> left_hip
        (12, 14), # right_shoulder -> right_elbow
        (12, 24), # right_shoulder -> right_hip
        (13, 15), # left_elbow -> left_wrist
        (14, 16), # right_elbow -> right_wrist
        (15, 17), # left_wrist -> left_pinky
        (15, 19), # left_wrist -> left_index
        (15, 21), # left_wrist -> left_thumb
        (16, 18), # right_wrist -> right_pinky
        (16, 20), # right_wrist -> right_index
        (16, 22), # right_wrist -> right_thumb
        (17, 19), # left_pinky -> left_index
        (18, 20), # right_pinky -> right_index
        (23, 24), # left_hip -> right_hip
        (23, 25), # left_hip -> left_knee
        (24, 26), # right_hip -> right_knee
        (25, 27), # left_knee -> left_ankle
        (26, 28), # right_knee -> right_ankle
        (27, 29), # left_ankle -> left_heel
        (27, 31), # left_ankle -> left_foot_index
        (28, 30), # right_ankle -> right_heel
        (28, 32), # right_ankle -> right_foot_index
        (29, 31), # left_heel -> left_foot_index
        (30, 32), # right_heel -> right_foot_index
    ]

VRU_TREE = [
        (1, 2), # left_wrist -> left_elbow
        (2, 3), # left_elbow -> left_shoulder
        (3, 4), # left_shoulder -> right_shoulder
        (4, 5), # right_shoulder -> right_elbow
        (5, 6), # right_elbow -> right_wrist
        (3, 7), # left_shoulder -> left_hip
        (4, 8), # right_shoulder -> right_hip
        (7, 8), # left_hip -> right_hip
        (7, 9), # left_hip -> left_knee
        (8, 10), # right_hip -> right_knee
        (9, 11), # left_knee -> left_ankle
        (10, 12), # right_knee -> right_ankle
    ]