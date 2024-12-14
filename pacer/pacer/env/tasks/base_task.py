# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import enum
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import operator
from copy import deepcopy
import random

from isaacgym import gymapi
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples
from isaacgym import gymtorch
from gym.wrappers import RecordVideo

import numpy as np
import torch

import imageio
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from pacer.pacer.utils.flags import flags
from pacer.utils.flags import flags
from collections import defaultdict
import aiohttp, cv2, asyncio
import json
from collections import deque
import shutil
PORT = 8081

SERVER = "localhost"

# Base class for RL tasks
class BaseTask():
    def __init__(self, cfg, enable_camera_sensors=False):
        self.headless = cfg["headless"]
        if self.headless == False and not flags.no_virtual_display:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=(1920, 1000),
            # self.virtual_display = SmartDisplay(size=(3840, 1000),
                                                visible=True)
            # visible=not flags.server_mode)
            self.virtual_display.start()
            print("Virtual display started")


        self.gym = gymapi.acquire_gym()
        self.paused = False
        self.color_image = None
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.state_record = defaultdict(list)

        self.device = "cpu"
        print("Device type: ", self.device_type)
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        # double check!
        # import pdb; pdb.set_trace()
        self.graphics_device_id = self.device_id + 2 # dl41
        # self.graphics_device_id = self.device_id # dl32

        if not flags.render:
            self.graphics_device_id = -1
        # enable_camera_sensors = True
        # if enable_camera_sensors == False and self.headless == True:
            # self.graphics_device_id = self.device_id+2 if flags.render else -1
        # if flags.server_mode:
            # self.graphics_device_id = self.device_id

        self.num_envs = cfg["env"]["numEnvs"]
        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]

        self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros((self.num_envs, self.num_states),
                                      device=self.device,
                                      dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs,
                                   device=self.device,
                                   dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs,
                                    device=self.device,
                                    dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs,
                                        device=self.device,
                                        dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs,
                                         device=self.device,
                                         dtype=torch.long)
        self.extras = {}

        self.original_props = {}
        self.dr_randomizations = {}
        self.first_randomization = True
        self.actor_params_generator = None
        self.extern_actor_params = {}
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.last_step = -1
        self.last_rand_step = -1

        # create envs, sim and viewer
        # import pdb; pdb.set_trace()
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.video_dir = str(osp.join('output', 'renderings', f"{cfg['name']}/"))

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        self.create_viewer()

        if flags.server_mode:
            import threading
            bgsk = threading.Thread(target=self.setup_client, daemon=True).start()
            bgsk = threading.Thread(target=self.setup_talk_client, daemon=True).start()

    def create_viewer(self):
        camera_props = gymapi.CameraProperties()
        # camera_props.width = 3840
        camera_props.width = 1920
        camera_props.height = 1000
        camera_props.enable_tensors = False # no means to change?

        # set the camera position based on up axis
        sim_params = self.gym.get_sim_params(self.sim)
        # print("Up axis: ", sim_params.up_axis==gymapi.UP_AXIS_Z)
        if sim_params.up_axis == gymapi.UP_AXIS_Z:
            # cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            cam_pos = gymapi.Vec3(40.0, 50, 9.0)
            cam_target = gymapi.Vec3(50.0, 50.0, 0.0)
            # cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        else:
            cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

        if self.headless == False:
            # headless server mode will use the smart display
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            # subscribe to keyboard shortcuts
            self.gym.subscribe_viewer_keyboard_event(self.viewer,gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V,"toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L,"toggle_video_record")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SEMICOLON,"cancel_video_record")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R,"reset")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F,"follow")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_G,"fixed")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_H,"divide_group")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C,"print_cam")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_M,"disable_collision_reset")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B,"fixed_path")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_N,"real_path")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K,"show_traj")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J,"apply_force")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT,"prev_env")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT,"next_env")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T,"resample_motion")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Y,"slow_traj")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_U,"height_debug")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I,"random_heading")

            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_SPACE, "PAUSE")

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        ###### Custom Camera Sensors ######
        self.recorder_camera_handles = []
        self.max_num_camera = 2 # 10 default
        self.viewing_env_idx = 0
        for idx, env in enumerate(self.envs):
            h = self.gym.create_camera_sensor(env, camera_props)
            # print(h)
            if flags.render:
                self.gym.set_camera_location(h, env, cam_pos, cam_target)
            self.recorder_camera_handles.append(h)
            if idx > self.max_num_camera: break

        self.recorder_camera_handle = self.recorder_camera_handles[0]
        self.recording, self.recording_state_change = True, True
        self.max_video_queue_size = 6000
        self._video_queue = deque(maxlen=self.max_video_queue_size)
        rendering_out = osp.join("output", "renderings")
        os.makedirs(rendering_out, exist_ok=True)
        self.cfg_name = self.cfg['args'].cfg_env.split("/")[-1].split(".")[0]
        self._video_path = osp.join(rendering_out, f"{self.cfg_name}_%s.mp4")
        print("video will be saved to: ", self._video_path)
        self._states_path = osp.join(rendering_out, f"{self.cfg_name}_%s.pkl")
        # self.gym.draw_env_rigid_contacts(self.viewer, self.envs[1], gymapi.Vec3(0.9, 0.3, 0.3), 1.0, True)

        # # recording
        # self.envs = RecordVideo(self.envs, self.video_dir)

        ##### Custom light source #####
        # import pdb; pdb.set_trace()
        self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.7, 0.7, 0.7), gymapi.Vec3(0.7, 0.7, 0.7), gymapi.Vec3(0, 0, 1))
        self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0))

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        print("Creating sim...")
        print("Compute device: ", self.device_id)
        print("Graphics device: ", self.graphics_device_id)
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def step(self, actions):

        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](
                actions)
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        # if self.device == 'cpu':
        self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations'][
                'noise_lambda'](self.obs_buf)

    def get_states(self):
        return self.states_buf


    def _clear_recorded_states(self):
        pass

    def _record_states(self):
        pass

    def _write_states_to_file(self, file_name):
        pass

    def setup_client(self):
        image = cv2.imread("data/mesh/mesh_downtown_san_jose_clean_bottom.png")
        loop = asyncio.new_event_loop()   # <-- create new loop in this thread here
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.video_stream())
        loop.run_forever()


    def setup_talk_client(self):
        loop = asyncio.new_event_loop()   # <-- create new loop in this thread here
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.talk())
        loop.run_forever()

    #print(URL)
    async def talk(self):
        URL = f'http://{SERVER}:{PORT}/ws'
        print("Starting websocket client")
        session = aiohttp.ClientSession()
        async with session.ws_connect(URL) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == 'close cmd':
                        await ws.close()
                        break
                    else:
                        print(msg.data)
                        try:
                            msg = json.loads(msg.data)
                            if msg['action'] == 'reset':
                                self.reset()
                            elif msg['action'] == 'start_record':
                                if self.recording:
                                    print("Already recording")
                                else:
                                    self.recording = True
                                    self.recording_state_change = True
                            elif msg['action'] == 'end_record':
                                if not self.recording:
                                    print("Not recording")
                                else:
                                    self.recording = False
                                    self.recording_state_change = True
                            elif msg['action'] == 'set_env':
                                query = msg['query']
                                env_id = query['env']
                                self.viewing_env_idx = int(env_id)
                                print("view env idx: ", self.viewing_env_idx)
                        except:
                            import ipdb
                            ipdb.set_trace()
                            print("error parsing server message")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

    #print(URL)
    async def video_stream(self):
        URL = f'http://{SERVER}:{PORT}/ws'
        print("Starting websocket client")
        session = aiohttp.ClientSession()
        async with session.ws_connect(URL) as ws:
            await ws.send_str("Start")
            while True:
                if "color_image" in self.__dict__ and not self.color_image is None and len(
                        self.color_image.shape) == 3:
                    image = cv2.resize(self.color_image, (800, 450), interpolation=cv2.INTER_AREA)
                    await ws.send_bytes(image.tobytes())


    def render(self, sync_frame_time=False):
        # import pdb; pdb.set_trace()
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):

                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                if evt.action == "PAUSE" and evt.value > 0:
                    self.paused = not self.paused

                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "toggle_video_record" and evt.value > 0:
                    self.recording = not self.recording
                    self.recording_state_change = True
                elif evt.action == "cancel_video_record" and evt.value > 0:
                    self.recording = False
                    self.recording_state_change = False
                    self._video_queue = deque(maxlen=self.max_video_queue_size)
                    self._clear_recorded_states()

                elif evt.action == "reset" and evt.value > 0:
                    self.reset()
                elif evt.action == "follow" and evt.value > 0:
                    flags.follow = not flags.follow
                elif evt.action == "fixed" and evt.value > 0:
                    flags.fixed = not flags.fixed
                elif evt.action == "divide_group" and evt.value > 0:
                    flags.divide_group = not flags.divide_group
                elif evt.action == "print_cam" and evt.value > 0:
                    cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
                    cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
                    print("Print camera", cam_pos)
                elif evt.action == "disable_collision_reset" and evt.value > 0:
                    flags.no_collision_check = not flags.no_collision_check
                    print("collision_reset: ", flags.no_collision_check)
                elif evt.action == "fixed_path" and evt.value > 0:
                    flags.fixed_path = not flags.fixed_path
                    print("fixed_path: ", flags.fixed_path)
                elif evt.action == "real_path" and evt.value > 0:
                    flags.real_path = not flags.real_path
                    print("real_path: ", flags.real_path)
                elif evt.action == "show_traj" and evt.value > 0:
                    flags.show_traj = not flags.show_traj
                    print("show_traj: ", flags.show_traj)
                elif evt.action == "apply_force" and evt.value > 0:
                    forces = torch.zeros((self.num_envs, self.num_bodies + self._num_traj_samples, 3), device=self.device, dtype=torch.float)
                    torques = torch.zeros((self.num_envs, self.num_bodies + self._num_traj_samples, 3), device=self.device, dtype=torch.float)
                    # forces[:, 8, :] = -800
                    forces[self.viewing_env_idx, 3, :] = -3500
                    forces[self.viewing_env_idx, 7, :] = -3500
                    # torques[:, 1, :] = 500
                    self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
                    if "_recovery_counter" in self.__dict__:
                        self._recovery_counter[self.viewing_env_idx] = self._recovery_steps

                elif evt.action == "prev_env" and evt.value > 0:
                    self.viewing_env_idx = (self.viewing_env_idx - 1) % self.num_envs
                    # self.recorder_camera_handle = self.recorder_camera_handles[self.viewing_env_idx]
                    print("Showing env: ", self.viewing_env_idx)
                elif evt.action == "next_env" and evt.value > 0:
                    self.viewing_env_idx = (self.viewing_env_idx + 1) % self.num_envs
                    # self.recorder_camera_handle = self.recorder_camera_handles[self.viewing_env_idx]
                    print("Showing env: ", self.viewing_env_idx)
                elif evt.action == "resample_motion" and evt.value > 0:
                    self.resample_motions()
                elif evt.action == "slow_traj" and evt.value > 0:
                    flags.slow = not flags.slow
                    print("slow_traj: ", flags.slow)
                elif evt.action == "height_debug" and evt.value > 0:
                    flags.height_debug = not flags.height_debug
                    print("height_debug: ", flags.height_debug)
                elif evt.action == "random_heading" and evt.value > 0:
                    flags.random_heading = not flags.random_heading
                    print("random_heading: ", flags.random_heading)

        if self.recording_state_change:
            if not self.recording:
                curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                curr_video_file_name = self._video_path % curr_date_time
                curr_states_file_name = self._states_path % curr_date_time
                fps = 60
                writer = imageio.get_writer(curr_video_file_name, fps=fps, macro_block_size=None)
                height, width, c = self._video_queue[0].shape
                height, width = height if height % 2 == 0 else height - 1, width if width % 2 == 0 else width - 1

                for frame in self._video_queue:
                    try:
                        writer.append_data(frame[:height, :width, :])
                    except:
                        print('image size changed???')
                        import ipdb; ipdb.set_trace()

                writer.close()
                self._video_queue = deque(maxlen = self.max_video_queue_size)
                self._write_states_to_file(curr_states_file_name)
                print(f"============ Video finished writing ============")


                if flags.server_mode:
                    shutil.copyfile(curr_states_file_name, "../crossroadweb/output/curr_motion.pkl")
                    shutil.copyfile(curr_video_file_name,"../crossroadweb/output/curr_video.mp4")
                    print("============ Video finished copying to server ============")
                self.recording_state_change = False
            # else:
            #     print(f"============ Writing video ============")

        # import pdb; pdb.set_trace()

        # if self.recording or flags.server_mode:
        if flags.render:

            # self.gym.render_all_camera_sensors(self.sim)
            # self.gym.get_viewer_camera_handle(self.viewer)

            # color_image = self.gym.get_camera_image(
            # self.sim, self.envs[self.viewing_env_idx], self.recorder_camera_handles[self.viewing_env_idx],
            # gymapi.IMAGE_COLOR)
            # self.color_image = color_image.reshape(color_image.shape[0], -1, 4)

            # fetch results
            # if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self._draw_task()
                self.gym.step_graphics(self.sim)

            if flags.no_virtual_display:
                # import pdb; pdb.set_trace()
                self.gym.render_all_camera_sensors(self.sim)
                color_image = self.gym.get_camera_image(self.sim, self.envs[self.viewing_env_idx], self.recorder_camera_handles[self.viewing_env_idx], gymapi.IMAGE_COLOR)
            # if len(color_image.shape) >= 3:
                self.color_image = color_image.reshape(color_image.shape[0], -1, 4)
                # cv2.imwrite("output.png", self.color_image) # for debugging
            else:
                img = self.virtual_display.grab()
                self.color_image = np.array(img)

            if self.recording and self.color_image is not None:
                # self._video_queue.append(self.color_image[..., :3])
                # print(self.rew_buf)
                self._video_queue.append(self.color_image)
                self._record_states()

            if self.viewer:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                # self.gym.sync_frame_time(self.sim)

            # else:
                # self.gym.poll_viewer_events(self.viewer)

    def get_actor_params_info(self, dr_params, env):
        """Returns a flat array of actor params, their names and ranges."""
        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name + '_' + str(prop_idx) + '_' + attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0 * float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name + '_' + str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    # Apply randomizations only on resets, due to current PhysX limitations
    def apply_randomizations(self, dr_params):
        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step -
                                   self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq,
                                    torch.ones_like(self.randomize_buf),
                                    torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs,
                                    as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param][
                    "schedule"] if "schedule" in dr_params[
                        nonphysical_param] else None
                sched_step = dr_params[nonphysical_param][
                    "schedule_steps"] if "schedule" in dr_params[
                        nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get(
                        "range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor,
                            corr + torch.randn_like(tensor) * params['var'] +
                            params['mu'])

                    self.dr_randomizations[nonphysical_param] = {
                        'mu': mu,
                        'var': var,
                        'mu_corr': mu_corr,
                        'var_corr': var_corr,
                        'noise_lambda': noise_lambda
                    }

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get(
                        "range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (
                            1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (
                            1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] -
                                       params['lo_corr']) + params['lo_corr']
                        return op(
                            tensor, corr + torch.rand_like(tensor) *
                            (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {
                        'lo': lo,
                        'hi': hi,
                        'lo_corr': lo_corr,
                        'hi_corr': hi_corr,
                        'noise_lambda': noise_lambda
                    }

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr)
                    for attr in dir(prop)
                }

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(prop, self.original_props["sim_params"],
                                     attr, attr_randomization_params,
                                     self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        for actor, actor_properties in dr_params["actor_params"].items():
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(
                                env, handle, n, gymapi.MESH_VISUAL,
                                gymapi.Vec3(random.uniform(0, 1),
                                            random.uniform(0, 1),
                                            random.uniform(0, 1)))
                        continue
                    if prop_name == 'scale':
                        attr_randomization_params = prop_attrs
                        sample = generate_random_samples(
                            attr_randomization_params, 1, self.last_step, None)
                        og_scale = 1
                        if attr_randomization_params['operation'] == 'scaling':
                            new_scale = og_scale * sample
                        elif attr_randomization_params[
                                'operation'] == 'additive':
                            new_scale = og_scale + sample
                        self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [{
                                attr: getattr(p, attr)
                                for attr in dir(p)
                            } for p in prop]
                        for p, og_p in zip(prop,
                                           self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items(
                            ):
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[
                                        env_id] = get_attr_val_from_sample(
                                            extern_sample,
                                            extern_offsets[env_id], p, attr)
                                apply_random_samples(
                                    p, og_p, attr, attr_randomization_params,
                                    self.last_step, smpl)
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items(
                        ):
                            smpl = None
                            if self.actor_params_generator is not None:
                                smpl, extern_offsets[
                                    env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id],
                                        prop, attr)
                            apply_random_samples(
                                prop, self.original_props[prop_name], attr,
                                attr_randomization_params, self.last_step,
                                smpl)

                    setter = param_setters_map[prop_name]
                    default_args = param_setter_defaults_map[prop_name]
                    setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id, 'extern_offset',
                              extern_offsets[env_id], 'vs extern_sample.shape',
                              extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def _physics_step(self):
        for i in range(self.control_freq_inv):
            if not self.paused and self.enable_viewer_sync:
                self.gym.simulate(self.sim)
            self.render()
        return

    def save_video(self, game, exp_name=None, rew=None, real_traj=None, ideal_traj=None):
        curr_game = game
        real_traj = np.array(real_traj)
        ideal_traj = np.array(ideal_traj)
        cur_rew = np.clip(rew['total'], -1, 1)
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        norm_rew = norm(cur_rew)
        # insert exp_name to video file name
        curr_video_file_name = self._video_path % curr_game
        curr_video_file_name = curr_video_file_name.replace(f"{curr_game}.mp4", f"{exp_name}_{curr_game}.mp4")
        save_dir = osp.dirname(curr_video_file_name)
        os.makedirs(save_dir, exist_ok=True)
        video = self._video_queue
        writer = imageio.get_writer(curr_video_file_name, fps=30, macro_block_size=None)
        # import pdb; pdb.set_trace()
        fig, ax, bg, = init_traj_plot(real_traj, ideal_traj)
        acc_rew = 0
        for f, frame in enumerate(video):
            if f % 2 == 0: # downsample
                # save color image for visualization
                # print(f"Saving frame {f}")
                if not osp.exists(f"output/vis_sim/{exp_name}_{curr_game}"):
                    os.makedirs(f"output/vis_sim/{exp_name}_{curr_game}")
                cv2.imwrite(f"output/vis_sim/{exp_name}_{curr_game}/{f}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                acc_rew += float(rew['total'][f//2])
                frame = cv2.putText(frame, f"Acc: {acc_rew:.3f}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0))
                frame = cv2.putText(frame, f"Total: {float(rew['total'][f//2]):.3f}", (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0))
                frame = cv2.putText(frame, f"Loc: {float(rew['loc'][f//2]):.3f}", (20, 150), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0))
                frame = cv2.putText(frame, f"Disc: {float(rew['disc'][f//2]):.3f}", (20, 200), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0))
                frame = cv2.putText(frame, f"Pow: {float(rew['pow'][f//2]):.3f}", (20, 250), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0))
                frame, fig, ax, bg, = update_traj_plot(fig, ax, bg, real_traj, frame, f, norm_rew)
                frame = combine_colormap(frame)
                # extract the frame
                writer.append_data(frame)
        writer.close()
        print(f"============ Video finished writing ============")
        # reset video queue
        self._video_queue = deque(maxlen=self.max_video_queue_size)

        # check if video is successfully saved
        if osp.exists(curr_video_file_name):
            print(f"Video saved to {curr_video_file_name}")
            return True
        else:
            print(f"Failed to save video to {curr_video_file_name}")
            return False

    def post_physics_step(self):
        raise NotImplementedError

def init_traj_plot(real_traj, ideal_traj):
    plt.clf()
    plt.close()
    fig, ax = plt.subplots(frameon=False, figsize=(4, 4))
    x_min = min(float(real_traj[:, 0].min()), float(ideal_traj[:, 0].min())) - 1
    x_max = max(float(real_traj[:, 0].max()), float(ideal_traj[:, 0].max())) + 1
    y_min = min(float(real_traj[:, 1].min()), float(ideal_traj[:, 1].min())) - 1
    y_max = max(float(real_traj[:, 1].max()), float(ideal_traj[:, 1].max())) + 1
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    # plot ideal trajectory (fix), lines0
    ax.plot(ideal_traj[:, 0], ideal_traj[:, 1], color='r', linestyle='--', linewidth=3)

    # plot real trajectory (agent), lines1
    ax.plot(real_traj[0, 0], real_traj[0, 1], c = 'black', marker='o', markersize=12)

    # plot red circle on start point (fix), lines2
    ax.plot(ideal_traj[0, 0], ideal_traj[0, 1], c = 'orange', marker='o', markersize=15)
    # plot the red star on the last point of agent (real traj), lines3
    ax.plot(ideal_traj[-1, 0], ideal_traj[-1, 1], c = 'orange', marker='*', markersize=15)

    # 描画と背景の保存
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(ax.bbox)

    # save initialized plot for debugging
    # plt.savefig("init_plot.png")
    return fig, ax, bg

def update_traj_plot(fig, ax, bg, real_traj, frame, f, norm_rew):
    # import pdb; pdb.set_trace()
    # marker on the last point, and line connecting all past points
    lines = ax.get_lines()

    # update the star on the latest point of agent (real traj), lines0
    lines[1].set_xdata(real_traj[f//2, 0])
    lines[1].set_ydata(real_traj[f//2, 1])
    ax.draw_artist(lines[1])
    fig.canvas.restore_region(bg)

    # plot the line connecting former points to the latest point of agent (real traj)
    rew_color = plt.cm.viridis(norm_rew[f//2])
    ax.plot(real_traj[f//2-1:f//2+1, 0], real_traj[f//2-1:f//2+1, 1], color=rew_color, linewidth=3) # TODO: change color corresponding to the former reward

    # render the plot to an image on cv2
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()
    plt.pause(0.01)

    # make sure the traj_fig size is (150, 150, 3)
    traj_fig = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

    # insert to top right corner of the frame
    frame[:traj_fig.shape[0], -traj_fig.shape[1]:, :3] = traj_fig
    # plt.savefig("update_plot.png")
    return frame, fig, ax, bg

def combine_colormap(frame):
    # import pdb; pdb.set_trace()
    # カラーバーの高さは画像と同じ、幅は適宜選択
    height, width = frame.shape[:2]
    bar_width = width // 10
    color_bar = np.linspace(1, 0, height).reshape(height, 1)  # 0～255のグラデーション
    color_bar = np.repeat(color_bar, bar_width, axis=1)  # カラーバーの幅を持たせる.

    # matplotlibのviridisカラーマップを適用
    color_bar_rgb = plt.get_cmap('viridis')(color_bar)[:, :, :3]  # RGBの最初の3チャンネルを抽出 (RGBA の A を無視)
    color_bar_rgb = (color_bar_rgb * 255).astype(np.uint8)  # 0～1の値を0～255に変換

    # OpenCVの画像形式にするためにRGBをBGRに変換
    color_bar_bgr = cv2.cvtColor(color_bar_rgb, cv2.COLOR_RGB2BGR)

    # 画像とカラーバーを横に並べて結合
    combined_image = cv2.hconcat([frame[..., :3], color_bar_rgb])

    return combined_image

def get_attr_val_from_sample(sample, offset, prop, attr):
    """Retrieves param value for the given prop and attr from the sample."""
    if sample is None:
        return None, 0
    if isinstance(prop, np.ndarray):
        smpl = sample[offset:offset + prop[attr].shape[0]]
        return smpl, offset + prop[attr].shape[0]
    else:
        return sample[offset], offset + 1
