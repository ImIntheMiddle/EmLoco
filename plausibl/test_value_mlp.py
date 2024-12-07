import sys
import os

sys.path.append("/misc/dl00/halo/plausibl/pacer/pacer")
import env.util.traj_generator as traj_generator
import torch
import torch.nn as nn
import torch.optim as optim

# sys.path.append('../pacer/pacer/learning/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from pacer.pacer.learning.amp_network_sept_builder import AMPSeptBuilder
from pacer.pacer.utils.flags import flags
from utils import torch_utils

MLP_PATH = "pacer/output/exp/pacer/value_mlp.pth"
SIM_TIMESTEP = 1.0 / 60.0
NUM_ENVS = 101
CRLFRQINV = 2.0


class MLP():
    def __init__(self, **kwargs):
        self._build_value_mlp()

    def _build_value_mlp(self):
        mlp_input_shape = 24  # Traj, heightmap
        mlp = nn.Sequential()
        mlp_args = {
            "input_size": mlp_input_shape,
            "units": [12, 6], # [20, 10]
            "activation": "relu",
            "dense_func": torch.nn.Linear,
        }
        mlp_initializer = "default"
        self._value_mlp = self._build_mlp(**mlp_args)
        mlp_out_size = mlp_args["units"][-1]
        self._value_logits = torch.nn.Linear(mlp_out_size, 1)
        mlp_init = nn.Identity()

        for m in self._value_mlp.modules():
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        torch.nn.init.uniform_(self._value_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._value_logits.bias)
        return

    def _build_mlp(
        self,
        input_size,
        units,
        activation,
        dense_func,
        norm_only_first_layer=False,
        norm_func_name=None,
    ):
        return self._build_sequential_mlp(
            input_size,
            units,
            activation,
            dense_func,
            norm_func_name=None,
        )

    def _build_sequential_mlp(
        self,
        input_size,
        units,
        activation,
        dense_func,
        norm_only_first_layer=False,
        norm_func_name=None,
    ):
        # print('build mlp:', input_size)
        in_size = input_size
        layers = []
        need_norm = True
        for unit in units:
            layers.append(dense_func(in_size, unit))
            layers.append(nn.ReLU())

            if not need_norm:
                continue
            if norm_only_first_layer and norm_func_name is not None:
                need_norm = False
            if norm_func_name == "layer_norm":
                layers.append(torch.nn.LayerNorm(unit))
            elif norm_func_name == "batch_norm":
                layers.append(torch.nn.BatchNorm1d(unit))
            in_size = unit

        return nn.Sequential(*layers)

    def load_weights(self, path):
        self._value_mlp.load_state_dict(torch.load(path), strict=False)
        self._value_logits.load_state_dict(torch.load(path), strict=False)
        # show the value network
        print("Value Network:")
        print(self._value_mlp)
        print(self._value_logits)
        print("Value Network Parameters:")
        print(self._value_mlp.parameters())
        print(self._value_logits.parameters())
        print("load weights successfully from: ", path)

    def forward(self, trajs):
        value_mlp_out = self._value_mlp(trajs)
        values = self._value_logits(value_mlp_out)
        return values


class Traj():
    def __init__(self):
        self.traj_gen = self.build_traj_generator()
        self.progress_buf = torch.zeros(NUM_ENVS, device="cpu", dtype=torch.float)

    def build_traj_generator(self):
        traj_gen = traj_generator.TrajGenerator(
            NUM_ENVS,
            episode_dur=300 * CRLFRQINV * SIM_TIMESTEP,
            num_verts=101,
            device="cpu",
            dtheta_max=2.0,
            speed_min=0.0,
            speed_max=3.0,
            accel_max=2.0,
            sharp_turn_prob=0.02,
        )
        env_ids = torch.arange(NUM_ENVS, device="cpu", dtype=torch.long)
        self.root_pos = torch.zeros((NUM_ENVS, 3), device="cpu")
        traj_gen.reset(env_ids, self.root_pos)
        return traj_gen

    def fetch_traj_samples(self, num_traj_samples=12, env_ids=None):
        # 5 seconds with 0.4 second intervals, 20 samples.
        if env_ids is None:
            env_ids = torch.arange(NUM_ENVS, device="cpu", dtype=torch.long)

        timestep_beg = self.progress_buf[env_ids] * CRLFRQINV * SIM_TIMESTEP
        timesteps = torch.arange(num_traj_samples, device="cpu", dtype=torch.float)
        timesteps = timesteps * 0.4
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self.traj_gen.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps.flatten()
        )
        traj_samples = torch.reshape(
            traj_samples_flat,
            shape=(env_ids.shape[0], num_traj_samples, traj_samples_flat.shape[-1]),
        )
        return traj_samples

    def compute_location_observations(self, traj_samples):
        # type: (Tensor, Tensor) -> Tensor
        root_pos = self.root_pos
        root_rot = torch.zeros((NUM_ENVS, 4), device="cpu")
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

        heading_rot_exp = torch.broadcast_to(
            heading_rot.unsqueeze(-2),
            (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]),
        )
        heading_rot_exp = torch.reshape(
            heading_rot_exp,
            (
                heading_rot_exp.shape[0] * heading_rot_exp.shape[1],
                heading_rot_exp.shape[2],
            ),
        )
        traj_samples_delta = traj_samples - root_pos.unsqueeze(-2)
        traj_samples_delta_flat = torch.reshape(
            traj_samples_delta,
            (
                traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
                traj_samples_delta.shape[2],
            ),
        )

        local_traj_pos = torch_utils.my_quat_rotate(
            heading_rot_exp, traj_samples_delta_flat
        )
        local_traj_pos = local_traj_pos[..., 0:2]

        obs = torch.reshape(
            local_traj_pos,
            (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]),
        )
        return obs

    def select_trajs(self, trajs, values, freq=10):
        values = values.squeeze()
        values_sorted, indices = torch.sort(values, descending=True)
        print("values: ", values_sorted)
        # print('indices: ', indices)

        # extract 10 representative trajs
        idxs = indices[::freq]
        # print('idxs: ', idxs)
        trajs_selected = trajs[idxs]
        values_selected = values[idxs]
        # print('trajs_selected: ', trajs_selected)
        return trajs_selected, values_selected

    def plot_trajs(self, trajs, values):
        trajs = trajs.detach().numpy()
        # make it 2D
        trajs = trajs.reshape(trajs.shape[0], -1, 2)
        # print("trajs: ", trajs.shape)

        # change color according to values
        values = values.detach().numpy()
        # print("values: ", values.shape)
        color_map = plt.cm.get_cmap("jet")
        values_norm = (values - values.min()) / (values.max() - values.min())
        c_values = color_map(values_norm)

        # plot
        for i in range(trajs.shape[0]):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # ax.scatter(trajs[i, :, 0], trajs[i, :, 1], c=values[i])
            ax.plot(trajs[i, :, 0], trajs[i, :, 1], c=c_values[i])
            # add timestep
            for j in range(trajs.shape[1]):
                ax.text(trajs[i, j, 0] + 0.1, trajs[i, j, 1] + 0.1, str(j + 1))
            ax.scatter(trajs[i, :, 0], trajs[i, :, 1], c=c_values[i], s=30)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Value: {values[i]}")
            # save
            fig.savefig(f"plausibl/output/traj_{i}.png")

class Opt():
    def __init__(self, trajs, value_mlp):
        self.trajs = trajs.flatten()
        self.optimizer = self._build_optimizer(self.trajs)
        self.value_mlp = value_mlp

    def _build_optimizer(self, trajs):
        trajs.requires_grad = True
        optimizer = optim.Adam([trajs], lr=1e-4)
        return optimizer

    def loss_motion(self, values):
        loss = torch.exp(-values)
        return loss

    def optimize(self):
        history_loss = []
        history_value = []
        history_traj = []
        for i in range(750):
            self.optimizer.zero_grad()
            trajs = self.trajs.clone().reshape(-1, 24)
            values = self.value_mlp.forward(trajs)
            loss = self.loss_motion(values)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            history_loss.append(loss.item())
            history_value.append(values.mean().item())
            if i % 50 == 0:
                print(f"loss at iter {i}: ", loss.item())
                print(f"value at iter {i}: ", values.mean().item())
                # print(f"trajs at iter {i}: ", trajs.clone().detach().numpy())
                trajs = self.optimizer.param_groups[0]['params'][0]
                history_traj.append(trajs.clone().detach().numpy())
        return history_loss, history_traj, history_value

def vis_opt_traj(history_traj):
    # visualize the trajectory optimization process
    # animation
    traj_num = len(history_traj[0]) // 24
    for i in range(traj_num): # for each trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Trajectory {i}")
        ims = []
        for j, trajs in enumerate(history_traj): # for each iteration of optimization
            traj_i = trajs[i*24:(i+1)*24]
            traj_i = traj_i.reshape(12, 2)
            im = ax.plot(traj_i[:, 0], traj_i[:, 1], c="r")
            ims.append(im)
            if j == 0:
                fig.savefig(f"plausibl/output/traj_{i}_init.png")
        ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000)
        ani.save(f"plausibl/output/traj_{i}.gif", writer="pillow")
        fig.savefig(f"plausibl/output/traj_{i}_final.png")
        plt.close()

def plot_loss(history_loss):
    # plot loss
    plt.figure()
    plt.semilogy(history_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("plausibl/output/loss.png")
    plt.close()

def plot_value(history_value):
    # plot value
    plt.figure()
    plt.semilogy(history_value)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.savefig("plausibl/output/value.png")
    plt.close()

def main():
    (flags.fixed_path, flags.real_path, flags.slow) = (False, False, False)
    # print("flags: ", flags)
    # construct the network
    mlp = MLP()
    mlp.load_weights(MLP_PATH)

    # build traj generater
    traj_gen = Traj()
    env_ids = torch.arange(NUM_ENVS, device="cpu", dtype=torch.long)
    trajs = traj_gen.fetch_traj_samples(env_ids=env_ids)
    trajs = traj_gen.compute_location_observations(trajs)
    print("trajs: ", trajs.shape)
    print("trajs: ", trajs[0])

    # forward
    values = mlp.forward(trajs)
    trajs, values = traj_gen.select_trajs(trajs, values, freq=5)

    # plot traj and save
    traj_gen.plot_trajs(trajs, values)

    # test time optimization
    # print("trajs: ", trajs.shape)
    lbfgs = Opt(trajs, mlp)
    history_loss, history_traj, history_value = lbfgs.optimize()
    # plot loss
    plot_loss(history_loss)
    # plot value
    plot_value(history_value)
    # visualize the trajectory optimization process
    vis_opt_traj(history_traj)

if __name__ == "__main__":
    if not os.path.exists("plausibl/output"):
        os.makedirs("plausibl/output")
    main()
