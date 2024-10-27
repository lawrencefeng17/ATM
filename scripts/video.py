import cv2
import numpy as np
import torch
import argparse
import h5py
import sys
sys.path.append('/home/lawrence/ATM')
from atm.utils.flow_utils import draw_traj_on_images, combine_track_and_img

def array_to_video(array, output_path='output.mp4', fps=30):
    """
    Convert a NumPy array of shape (1, T, C, H, W) or (T, C, H, W) to an MP4 video.
    
    Args:
        array: NumPy array of shape (1, T, C, H, W) or (T, C, H, W)
        output_path: Path where the video will be saved
        fps: Frames per second for the output video
    """
    # Remove batch dimension if present
    if array.ndim == 5 and array.shape[0] == 1:
        array = array.squeeze(0)  # (T, C, H, W)
    
    # Ensure array is uint8
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    
    # Convert from (T, C, H, W) to (T, H, W, C)
    frames = array.transpose(0, 2, 3, 1)
    
    # Convert RGB to BGR for OpenCV
    frames = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames])
    
    # Get video dimensions
    height, width = frames.shape[1:3]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame in frames:
        out.write(frame)
    
    # Release video writer
    out.release()
    
    print(f"Video saved to {output_path}")
    return output_path

def load_h5(fn):
    def h5_to_dict(h5):
        d = {}
        for k, v in h5.items():
            if isinstance(v, h5py._hl.group.Group):
                d[k] = h5_to_dict(v)
            else:
                d[k] = np.array(v)
        return d
    with h5py.File(fn, 'r') as f:
        return h5_to_dict(f)

def process_video_with_tracks(video, tracks):
    """
    This adds all tracking points, this is not what we want.
    """
    episode_frames = []
    # root/agentview/video (Dataset) - shape: (1, 98, 3, 128, 128), dtype: uint8
    # root/agentview/tracks (Dataset) - shape: (1, 98, 1098, 2), dtype: float32
    if not isinstance(tracks, torch.Tensor):
        tracks = torch.from_numpy(tracks)
    T, C, H, W = video.squeeze(0).shape

    for t in range(T):
        img = combine_track_and_img(tracks[:, t:t+1, :, :], video[:, t])
        episode_frames.append(img)
        
    episode_frames = np.stack(episode_frames, axis=1)  
    return episode_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, required=True)
    parser.add_argument('--use-lines', action='store_true', default=True,
                        help='Use lines to connect points instead of track overlay')
    args = parser.parse_args()
    
    # Load data
    demo = load_h5(args.f)
    
    # Save original videos
    array_to_video(demo['root']['agentview']['video'], "demo_agentview_0.mp4")
    array_to_video(demo['root']['eye_in_hand']['video'], "demo_eye_in_hand_0.mp4")
    
    # Process videos with tracks
    agentview_processed = process_video_with_tracks(
        demo['root']['agentview']['video'],
        demo['root']['agentview']['tracks'],
    )
    
    eye_in_hand_processed = process_video_with_tracks(
        demo['root']['eye_in_hand']['video'],
        demo['root']['eye_in_hand']['tracks'],
    )
    
    # Save processed videos
    array_to_video(agentview_processed, "demo_agentview_traj_0.mp4")
    array_to_video(eye_in_hand_processed, "demo_eye_in_hand_traj_0.mp4")

if __name__ == "__main__":
    main()