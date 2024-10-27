import cv2
import numpy as np
import torch
import argparse
import h5py

def array_to_video(array, output_path='output.mp4', fps=30):
    """
    Convert a NumPy array of shape (1, T, C, H, W) to an MP4 video.
    
    Args:
        array: NumPy array of shape (1, T, C, H, W)
               where T=time steps, C=channels (3 for RGB), H=height, W=width
        output_path: Path where the video will be saved
        fps: Frames per second for the output video
    """
    # Remove batch dimension
    if array.shape[0] == 1:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, required=True)

    args = parser.parse_args()
    file = args.f

    demo = load_h5(file)
    array_to_video(demo['root']['agentview']['video'], "demo_agentview_0.mp4")
    array_to_video(demo['root']['eye_in_hand']['video'], "demo_eye_in_hand_0.mp4")

if __name__ == "__main__":
    main()
