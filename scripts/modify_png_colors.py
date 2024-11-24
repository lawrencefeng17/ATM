import numpy as np
from PIL import Image

from pathlib import Path
import json

import argparse

def replace_color_binary_image(image_path, new_color, output_path=None, threshold=128):
    """
    Replace the white color in a binary image with a new color.
    
    Args:
        image_path (str): Path to input binary image
        new_color (list): New color in [R, G, B, A] format, values from 0-1
        output_path (str): Path to save modified image (optional)
        threshold (int): Threshold value to identify white pixels (0-255)
    
    Returns:
        PIL.Image: Modified image
    """
    # Open the image
    img = Image.open(image_path)
    # Convert to RGBA if not already
    img = img.convert('RGBA')
    # Convert to numpy array
    img_array = np.array(img)
    
    # Convert new_color from 0-1 float to 0-255 uint8
    new_color_255 = np.array([int(c * 255) for c in new_color], dtype=np.uint8)
    
    # Create mask for white pixels (assuming RGB values > threshold indicates white)
    # We check all RGB channels to be above threshold
    white_mask = np.all(img_array[:, :, :3] > threshold, axis=2)
    
    # Create output array
    output_array = img_array.copy()
    
    # Set new color for white pixels
    output_array[white_mask] = new_color_255
    
    # Convert back to PIL Image
    modified_img = Image.fromarray(output_array)
    
    # Save if output path provided
    if output_path:
        modified_img.save(output_path)
    
    return modified_img

# Example usage
if __name__ == "__main__":
    """
    Replace default texture with new color according to color_mapping JSON file.
    
    Takes in a singular color_mapping JSON file as argument.
    
    By default, loops through all JSON files in the scripts directory.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--color-mapping", type=str)
    
    args = parser.parse_args()
    
    assets_path = "/home/lawrence/ATM/libero/assets/stable_scanned_objects"
    
    if args.color_mapping:
        color_mapping = args.color_mapping 

        with open(color_mapping, 'r') as f:
            data = json.load(f)
            for obj, color in data.items():
                # Add new texture png 
                image_path = Path(assets_path) / obj / "texture.png"
                new_color = color
                output_path = Path(assets_path) / obj / f"texture_{str(color)}.png"
                replace_color_binary_image(image_path, new_color, output_path=output_path)
                print(f"Modified image saved to {output_path}") 
    else:
        color_mappings = "/home/lawrence/ATM/scripts/color_mappings"
    
        # Loop through json files in scripts directory
        for json_file in Path(color_mappings).rglob("*.json"):
            # object : color in json
            with open(json_file, 'r') as f:
                data = json.load(f)
                for obj, color in data.items():
                    # Add new texture png 
                    image_path = Path(assets_path) / obj / "texture.png"
                    new_color = color
                    output_path = Path(assets_path) / obj / f"texture_{str(color)}.png"
                    replace_color_binary_image(image_path, new_color, output_path=output_path)
                    print(f"Modified image saved to {output_path}")