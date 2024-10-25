import os
import sys
import h5py
import numpy as np
from pathlib import Path
sys.path.append('/home/lawrence/ATM')
from libero.envs import OffScreenRenderEnv
from scripts.hdf5 import print_hdf5_structure


def render_new_videos(
    input_hdf5_path,
    output_hdf5_path,
    bddl_file,
    camera_names=["agentview", "eye_in_hand"],
    camera_height=128,
    camera_width=128
):
    """
    Re-renders the agentview and eye_in_hand videos from an existing HDF5 demonstration file
    assuming the simulation environment is already configured with modified XML files for object colors.

    Args:
        input_hdf5_path (str): Path to the input HDF5 demonstration file.
        output_hdf5_path (str): Path to save the new HDF5 demonstration file with updated videos.
        bddl_file (str): Path to the BDDL file defining the task and environment.
        camera_names (list, optional): List of camera names to render. Defaults to ["agentview", "eye_in_hand"].
        camera_height (int, optional): Height of the camera images. Defaults to 128.
        camera_width (int, optional): Width of the camera images. Defaults to 128.
    """

    # Ensure output directory exists
    output_parent_dir = Path(output_hdf5_path).parent
    output_parent_dir.mkdir(parents=True, exist_ok=True)

    # Open the input and output HDF5 files
    with h5py.File(input_hdf5_path, 'r') as infile, h5py.File(output_hdf5_path, 'w') as outfile:
        # Function to copy all datasets and groups except video datasets
        def copy_except_videos(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Determine if the dataset is a video dataset
                if name.endswith('video'):
                    # Skip video datasets; they will be re-rendered
                    return
                else:
                    # Create the same dataset in the output file
                    outfile.create_dataset(name, data=obj[()], dtype=obj.dtype)
            elif isinstance(obj, h5py.Group):
                # Create the same group in the output file
                outfile.create_group(name)

        # Copy all datasets and groups except for the 'video' datasets

        infile = infile['root']
        infile.visititems(copy_except_videos)

        # Extract actions and initial state
        actions = infile['actions'][:]  # Shape: (num_steps, action_dim)
        ee_states = infile['extra_states/ee_states'][:]  # Shape: (num_steps, 6)

        # Assume the initial state corresponds to the first step
        initial_state = ee_states[0]

        # Initialize the simulation environment
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": camera_height,
            "camera_widths": camera_width,
        }

        env = OffScreenRenderEnv(**env_args)
        env.reset()

        """
        TRYING TO FIGURE OUT HOW TO INITIALIZE ENVIRONMENT STATE FROM HDF5 FILE
        """
        env.set_init_state(initial_state)

        # Initialize lists to store video frames
        agentview_frames = []
        eye_in_hand_frames = []

        # Iterate through each action and step the environment
        for idx, action in enumerate(actions):
            # Perform the action
            obs, reward, done, info = env.step(action)

            # Capture images from each camera
            for cam in camera_names:
                cam_image_key = f"{cam}_image"
                if cam_image_key in obs:
                    frame = obs[cam_image_key]  # e.g., 'agentview_image', 'eye_in_hand_image'
                    
                    # Ensure frame shape is (3, height, width) and type uint8
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    
                    if frame.shape[0] == 3:
                        # Convert to (height, width, 3)
                        frame = np.transpose(frame, (1, 2, 0))
                    elif frame.shape[-1] == 3:
                        # Already in (height, width, 3)
                        pass
                    else:
                        raise ValueError(f"Unexpected frame shape: {frame.shape}")
                    
                    if cam == "agentview":
                        agentview_frames.append(frame)
                    elif cam == "eye_in_hand":
                        eye_in_hand_frames.append(frame)
            
            if done:
                break

        # Convert lists to numpy arrays
        if agentview_frames:
            agentview_video = np.stack(agentview_frames, axis=0)  # (num_steps, height, width, 3)
            # Add a new axis to match original HDF5 structure (1, num_steps, 3, height, width)
            agentview_video = np.transpose(agentview_video, (0, 3, 1, 2))  # (num_steps, 3, height, width)
            agentview_video = np.expand_dims(agentview_video, axis=0)  # (1, num_steps, 3, height, width)
        else:
            print("No frames captured for agentview.")
            agentview_video = np.empty((1, 0, 3, camera_height, camera_width), dtype=np.uint8)

        if eye_in_hand_frames:
            eye_in_hand_video = np.stack(eye_in_hand_frames, axis=0)  # (num_steps, height, width, 3)
            # Add a new axis to match original HDF5 structure (1, num_steps, 3, height, width)
            eye_in_hand_video = np.transpose(eye_in_hand_video, (0, 3, 1, 2))  # (num_steps, 3, height, width)
            eye_in_hand_video = np.expand_dims(eye_in_hand_video, axis=0)  # (1, num_steps, 3, height, width)
        else:
            print("No frames captured for eye_in_hand.")
            eye_in_hand_video = np.empty((1, 0, 3, camera_height, camera_width), dtype=np.uint8)

        # Create the video datasets in the output HDF5 file
        if 'agentview' not in outfile:
            outfile.create_group('agentview')
        outfile['agentview'].create_dataset('video', data=agentview_video, dtype='uint8')

        if 'eye_in_hand' not in outfile:
            outfile.create_group('eye_in_hand')
        outfile['eye_in_hand'].create_dataset('video', data=eye_in_hand_video, dtype='uint8')

        # Close the environment
        env.close()


# re_render_videos.py

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Re-render videos in HDF5 demonstration files with updated object colors.")
    parser.add_argument("--input_hdf5", type=str, required=True, help="Path to the input HDF5 file.")
    parser.add_argument("--output_hdf5", type=str, required=True, help="Path to save the output HDF5 file with updated videos.")
    parser.add_argument("--bddl_file", type=str, required=True, help="Path to the BDDL file defining the task and environment.")
    parser.add_argument("--camera_height", type=int, default=128, help="Height of the camera images.")
    parser.add_argument("--camera_width", type=int, default=128, help="Width of the camera images.")
    parser.add_argument("--camera_names", type=str, nargs='+', default=["agentview", "eye_in_hand"], help="Names of the cameras to render.")
    
    args = parser.parse_args()

    # Validate file paths
    input_hdf5_path = Path(args.input_hdf5)
    output_hdf5_path = Path(args.output_hdf5)
    bddl_file = Path(args.bddl_file)

    if not input_hdf5_path.is_file():
        raise FileNotFoundError(f"Input HDF5 file not found: {args.input_hdf5}")
    if not bddl_file.is_file():
        raise FileNotFoundError(f"BDD file not found: {args.bddl_file}")

    # Call the render_new_videos function
    render_new_videos(
        input_hdf5_path=str(input_hdf5_path),
        output_hdf5_path=str(output_hdf5_path),
        bddl_file=str(bddl_file),
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width
    )

    print(f"Successfully re-rendered videos and saved to {args.output_hdf5}")

if __name__ == "__main__":
    main()
