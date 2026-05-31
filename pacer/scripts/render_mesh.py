"""Render a scene mesh to depth + walkable-map and dump it as a PACER
mesh asset (`data/mesh/<name>.pkl`).

Originally a one-off internal script with hardcoded paths to author-local
.obj files; reworked to be argparse-driven so a third party can point it
at any source mesh.

Note: not part of the default pipeline, so its `pyrender` / `trimesh`
dependencies are not in pyproject.toml. Install them manually before use:
    pip install pyrender trimesh
Run headless with the EGL backend:
    PYOPENGL_PLATFORM=egl python scripts/render_mesh.py --mesh ...
"""

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh
from tqdm import tqdm

# switch to "osmesa" or "egl" before loading pyrender (off-screen rendering)
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Source .obj mesh path")
    parser.add_argument(
        "--name",
        default=None,
        help="Output basename; defaults to mesh file stem (e.g. with_less_car)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/mesh",
        help="Directory under which to write <name>.pkl / .png",
    )
    parser.add_argument(
        "--size_mul",
        type=int,
        default=20,
        help="Pixel-per-unit multiplier for the rendered depth grid",
    )
    parser.add_argument(
        "--cam_height", type=float, default=100, help="Orthographic camera height"
    )
    parser.add_argument(
        "--threshold_big", type=float, default=1.5, help="Large step-change threshold"
    )
    parser.add_argument(
        "--threshold_small", type=float, default=0.5, help="Small step-change threshold"
    )
    parser.add_argument(
        "--no_preview",
        action="store_true",
        help="Skip the matplotlib preview windows (headless)",
    )
    return parser.parse_args()


def scan_columns(depth, threshold):
    """Sweep depth columnwise, flagging step changes above `threshold`."""
    curr_map_acc = []
    col_slice_prev = depth[:, 0]
    curr_map = np.zeros(depth.shape[0]).astype(bool)
    prev_grads = np.zeros(depth.shape[0])
    for i in tqdm(range(1, depth.shape[-1])):
        col_slice = depth[:, i]
        grad = col_slice - col_slice_prev
        curr_map[grad > threshold] = True
        curr_map[grad < -threshold] = False
        prev_grads[grad > threshold] = grad[grad > threshold]
        curr_map_acc.append(curr_map.copy())
        col_slice_prev = col_slice.copy()
    curr_map_acc.append(curr_map.copy())
    return ~np.stack(curr_map_acc, axis=1)


def main():
    args = parse_args()
    mesh_name = args.name or os.path.splitext(os.path.basename(args.mesh))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    mesh_data = trimesh.load(args.mesh)
    vertices = np.array(mesh_data.vertices).astype(np.float32)
    faces = np.array(mesh_data.faces)

    max_vals = vertices.max(axis=0) - vertices.min(axis=0)
    vertices[:, 1] -= vertices[:, 1].max() - (max_vals / 2)[1]
    vertices[:, 0] -= vertices[:, 0].max() - (max_vals / 2)[0]

    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)

    extent = (vertices.max(axis=0) - vertices.min(axis=0)).astype(int)
    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.3], bg_color=[0, 0, 0])
    camera = pyrender.OrthographicCamera(
        xmag=extent[0] / 2, ymag=extent[1] / 2, znear=0.1, zfar=1000
    )
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

    scene.add(mesh, pose=np.eye(4))
    scene.add(light, pose=np.eye(4))

    cam_pos = np.array([0, 0, args.cam_height])
    scene.add(
        camera,
        pose=[
            [1, 0, 0, cam_pos[0]],
            [0, 1, 0, cam_pos[1]],
            [0, 0, 1, cam_pos[2]],
            [0, 0, 0, 1],
        ],
    )

    print("size: ", extent[0] * args.size_mul, extent[1] * args.size_mul)
    renderer = pyrender.OffscreenRenderer(
        extent[0] * args.size_mul, extent[1] * args.size_mul
    )
    _color, depth = renderer.render(scene)

    depth = -depth + cam_pos[-1]
    depth[depth == cam_pos[2]] = 0
    print("depth range:", depth.max(), depth.min())

    print("Scanning")
    curr_map_acc_big = scan_columns(depth, args.threshold_big)
    curr_map_acc_small = scan_columns(depth, args.threshold_small)
    curr_map_acc = np.logical_and(curr_map_acc_small, curr_map_acc_big)

    # Crop edges (these constants matched the author's original mesh; tune for
    # your asset before relying on the walkable map).
    curr_map_acc[-2000:, :] = False
    curr_map_acc[:1500, :] = False
    curr_map_acc[:, -1500:] = False
    curr_map_acc[:, :1500] = False
    curr_map_acc[2300:5890, 6600:9800] = False
    curr_map_acc[2700:5500, 2700:5500] = False

    out_pkl = os.path.join(args.output_dir, f"{mesh_name}.pkl")
    joblib.dump(
        {
            "vertices": vertices,
            "faces": faces,
            "x_scale": args.size_mul,
            "y_scale": args.size_mul,
            "cam_pos": cam_pos,
            "walkable_map": curr_map_acc.T[:, ::-1].copy(),
            "heigthmap": depth.T[:, ::-1].copy(),
        },
        out_pkl,
    )
    print(f"Wrote {out_pkl}")

    plt.imsave(os.path.join(args.output_dir, f"{mesh_name}.png"), depth)
    plt.imsave(os.path.join(args.output_dir, f"{mesh_name}_walkable.png"), curr_map_acc)
    print(cam_pos)

    if not args.no_preview:
        plt.figure(dpi=100)
        plt.imshow(curr_map_acc, cmap="gray")
        plt.show()
        plt.figure(dpi=100)
        plt.imshow(depth, cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
