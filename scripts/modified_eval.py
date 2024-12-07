import os
import json
import argparse
from pathlib import Path

import xml.etree.ElementTree as ET

from modify_png_colors import replace_color_binary_image

def replace_texture_filename_in_xml(file_name, new_texture):
    """
    Replace the texture filename in a MuJoCo XML file.
    
    Args:
        file_name (str): Path to the .xml file
        new_texture (str): New texture filename
    """
    tree = ET.parse(file_name)
    root = tree.getroot()
    
    # Find all texture elements
    for texture in root.findall(".//texture"):
        texture.set("file", new_texture)
    
    tree.write(file_name)

def replace_texture_filename_in_mtl(file_name, new_texture):
    """
    Replace the texture filename in a Wavefront .mtl file.
    
    Args:
        file_name (str): Path to the .mtl file
        new_texture (str): New texture filename
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    # Replace the texture filename
    modified_lines = []
    for line in lines:
        if line.startswith("map_Kd"):
            modified_lines.append(f"map_Kd {new_texture}\n")
        else:
            modified_lines.append(line)
    
    # Write the modified lines back to file
    with open(file_name, 'w') as f:
        f.writelines(modified_lines)

def reset_texture_filename_in_xml(file_name, old_texture="texture.png"):
    """
    Reset the texture filename in a MuJoCo XML file.
    
    Args:
        file_name (str): Path to the .xml file
        old_texture (str): Old texture filename
    """
    tree = ET.parse(file_name)
    root = tree.getroot()
    
    # Find all texture elements
    for texture in root.findall(".//texture"):
        texture.set("file", old_texture)
    
    tree.write(file_name)
    
def reset_texture_filename_in_mtl(file_name, old_texture="texture.png"):
    """
    Reset the texture filename in a Wavefront .mtl file.
    
    Args:
        file_name (str): Path to the .mtl file
        old_texture (str): Old texture filename
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    # Replace the texture filename
    modified_lines = []
    for line in lines:
        if line.startswith("map_Kd"):
            modified_lines.append(f"map_Kd {old_texture}\n")
        else:
            modified_lines.append(line)
    
    # Write the modified lines back to file
    with open(file_name, 'w') as f:
        f.writelines(modified_lines)
    
def main():
    """
    Given a color_mapping JSON file, replace the texture .png files in the corresponding .xml and .mtl files.
    
    Then, run the evaluation script to run the policy on the modified environment.
    """
    parser = argparse.ArgumentParser(
        description='Update texture filenames in MuJoCo XML files'
    )

    parser.add_argument(
        '--suite',
        type=str,
        default='libero_spatial',
    )

    parser.add_argument(
        '-f',
        type=str,
        required=True,
        help='Path to the color mapping JSON file'
    )
    
    parser.add_argument(
        '--policy',
        type=str,
        required=True,
        help='Path to the policy directory'
    )
    

    args = parser.parse_args() 
    
    libero_suite = args.suite
    json_file = args.f
    policy_dir = args.policy
    
    # Configure according to json file
    scripts_dir = Path(__file__).parent
    atm_dir = Path(scripts_dir).parent
    assets_path = atm_dir / "libero/assets/stable_scanned_objects"

    with open(json_file, 'r') as f:
        data = json.load(f)
        for obj, color in data.items():
            object_folder = Path(assets_path) / obj
            if color is None: 
                # Reset texture to default
                reset_texture_filename_in_xml(object_folder / f"{obj}.xml")
                reset_texture_filename_in_mtl(object_folder / f"{obj}.mtl")
                continue
            else:
                new_texture = f"texture_{str(color)}.png"

                # Update texture in XML file
                xml_file = object_folder / f"{obj}.xml"
                replace_texture_filename_in_xml(xml_file, new_texture)
                
                # Update texture in MTL file
                mtl_file = list(Path(object_folder).rglob("*.mtl"))
                if len(mtl_file) == 1:
                    mtl_file = mtl_file[0]
                else:
                    print(f"Error: Found {len(mtl_file)} MTL files for {obj}")
                    continue
                
                replace_texture_filename_in_mtl(mtl_file, new_texture)

    # Run evaluation script
    
    command = f"python -m scripts.eval_libero_policy --suite {libero_suite} --exp-dir {policy_dir}"
    os.system(command)

        
if __name__ == "__main__":
    main()