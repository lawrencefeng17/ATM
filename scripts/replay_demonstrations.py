import argparse
import h5py
import numpy as np
import cv2
import os
import sys
from tqdm import tqdm

# Add the parent directory of 'libero' to the Python path
sys.path.append('/home/lawrence/ATM')

from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from libero.envs import TASK_MAPPING
import libero.envs.bddl_utils as BDDLUtils
from robosuite.utils.binding_utils import MjRenderContextOffscreen

def modify_object_colors(env, new_colors):
    for obj_name, color in new_colors.items():
        try:
            obj_id = env.sim.model.geom_name2id(obj_name)
            env.sim.model.geom_rgba[obj_id] = color + [1]  # RGB + alpha
        except ValueError:
            print(f"Warning: Object {obj_name} not found in the environment.")

def render_offscreen(sim, width, height, camera_name):
    # Create an offscreen render context
    render_context = MjRenderContextOffscreen(sim, device_id=-1)

    # Set the camera
    camera_id = sim.model.camera_name2id(camera_name)
    render_context.render = True
    render_context.vopt.geomgroup[:] = 1
    render_context.scn.camera = camera_id
    render_context.render()

    # Read pixels
    pixels = render_context.read_pixels(width, height, depth=False)[::-1, :, :]
    return pixels

def replay_demonstration(env, demo_file, output_dir, new_colors):
    with h5py.File(demo_file, 'r') as f:
        root = f["root"]
        actions = root["actions"][()]
        
        views_to_modify = ['agentview', 'eye_in_hand']
        original_frames = {view: root[view]["video"][0] for view in views_to_modify if view in root}

    env.reset()
    modify_object_colors(env, new_colors)

    new_frames = {view: [] for view in views_to_modify if view in original_frames}

    for action in tqdm(actions, desc="Replaying demonstration"):
        env.step(action)
        for view in views_to_modify:
            if view in original_frames:
                frame = render_offscreen(
                    env.sim,
                    width=original_frames[view].shape[2],
                    height=original_frames[view].shape[1],
                    camera_name=view
                )
                new_frames[view].append(frame)

    # Save the modified demonstration
    with h5py.File(os.path.join(output_dir, 'modified_demo.hdf5'), 'w') as new_f:
        with h5py.File(demo_file, 'r') as original_f:
            # Copy the entire structure of the original file
            original_f.copy(original_f['/'], new_f['/'])
            
            # Replace only the modified views
            for view in views_to_modify:
                if view in new_frames:
                    del new_f['root'][view]['video']
                    new_f['root'][view].create_dataset('video', data=[np.array(new_frames[view])])

    # Save videos for each modified view
    for view, frames in new_frames.items():
        video_writer = cv2.VideoWriter(
            os.path.join(output_dir, f'replay_{view}.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            20,  # fps
            (frames[0].shape[1], frames[0].shape[0])  # frame size (width, height)
        )
        for frame in frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()

def main(args):
    # Load environment configuration
    controller_config = load_controller_config(default_controller="OSC_POSE")
    
    # Use the configuration from the original demonstration
    config = {
        "robots": ['Panda'],
        "controller_configs": controller_config,
    }

    # Create environment
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    problem_name = problem_info["problem_name"]
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    env = VisualizationWrapper(env)

    # Define new colors for objects (adjust as needed)
    new_colors = {
        "akita_black_bowl_1_g11": [1, 1, 0],  # Yellow
    }

    # Replay demonstration
    replay_demonstration(env, args.demo_file, args.output_dir, new_colors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_file", type=str, required=True, help="Path to the demonstration HDF5 file")
    parser.add_argument("--bddl_file", type=str, required=True, help="Path to the BDDL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output video")
    args = parser.parse_args()

    main(args)