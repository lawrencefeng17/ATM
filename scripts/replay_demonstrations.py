import sys
sys.path.append('/home/lawrence/ATM')

import argparse
import os
from pathlib import Path
import json
import h5py
import numpy as np

import robosuite
from robosuite import load_controller_config
import robosuite.macros as macros
import robosuite.utils.transform_utils as T
import robosuite.utils.camera_utils as camera_utils

from xml.dom import minidom
from xml.etree import ElementTree as ET

import libero.utils.utils as libero_utils
from libero.envs import TASK_MAPPING
from libero import get_libero_path

def change_object_colors(model_xml, object_color_mapping):
    """
    Modify the RGBA colors of specified objects in the Mujoco XML model.

    Args:
        model_xml (str): The original XML model as a string.
        object_color_mapping (dict): Mapping from object names to new RGBA colors.

    Returns:
        str: The modified XML model as a string.
    """
    # Parse the XML string
    xml_tree = ET.ElementTree(ET.fromstring(model_xml))
    root = xml_tree.getroot()

    # Iterate over all geom elements
    for geom in root.findall('.//geom'):
        geom_name = geom.get('name')
        if geom_name in object_color_mapping:
            rgba = object_color_mapping[geom_name]
            geom.set('rgba', ' '.join(map(str, rgba)))

    # Convert back to XML string
    modified_model_xml = ET.tostring(root, encoding='unicode')
    return modified_model_xml

def map_bddl_path(original_path, bddl_base_path):
    """
    Replace the hard-coded BDDL path prefix with the correct base path.

    Args:
        original_path (str): The original BDDL file path from the demo file.
        bddl_base_path (str): The correct base path where BDDL files are located.

    Returns:
        str: The updated BDDL file path.
    """
    # Define the prefix to replace
    # For example, replace 'chiliocosm/bddl_files/' with '/home/lawrence/ATM/libero/bddl_files/'
    prefix_to_replace = 'chiliocosm/bddl_files/'

    if original_path.startswith(prefix_to_replace):
        relative_path = original_path[len(prefix_to_replace):]  # Remove the prefix
        new_path = os.path.join(bddl_base_path, relative_path)
        return new_path
    else:
        # If the original path does not start with the expected prefix, return as is or handle accordingly
        print(f"[Warning] Original BDDL path '{original_path}' does not start with '{prefix_to_replace}'. Using original path.")
        return original_path

def print_geom_names(model_xml):
    """
    Print all geom names in the given Mujoco XML model.

    Args:
        model_xml (str): The XML model as a string.
    """
    xml_tree = ET.ElementTree(ET.fromstring(model_xml))
    root = xml_tree.getroot()
    geom_names = [geom.get('name') for geom in root.findall('.//geom')]
    print("Geom names:", geom_names)

def verify_actions(original_actions, new_actions):
    """
    Verify that the original and new actions are identical.

    Args:
        original_actions (np.ndarray): Actions from the original demo file.
        new_actions (np.ndarray): Actions from the new demo file.

    Returns:
        bool: True if actions match, False otherwise.
    """
    if original_actions.shape != new_actions.shape:
        print("[Error] Action shapes do not match.")
        return False
    if not np.allclose(original_actions, new_actions):
        print("[Error] Actions do not match.")
        return False
    print("[Success] Actions match between original and new demo files.")
    return True


def pretty_print_xml(xml_string):
    """
    Pretty prints an XML string with indentation for readability.

    Args:
        xml_string (str): The XML string to be formatted and printed.
    
    Returns:
        str: The formatted XML string with indentation.
    """
    # Parse the XML string into an ElementTree element
    xml_element = ET.fromstring(xml_string)
    
    # Convert the element to a string with encoding
    raw_xml = ET.tostring(xml_element, 'utf-8')
    
    # Use minidom to parse and pretty print
    parsed_xml = minidom.parseString(raw_xml)
    pretty_xml = parsed_xml.toprettyxml(indent="  ")
    
    # Print the pretty XML
    print(pretty_xml)
    
    # Return the pretty XML string
    return pretty_xml

def fix_mujoco_paths(xml_content, replacements):
    """
    Fix file paths in MuJoCo XML content.
    
    Args:
        xml_content (str): Original XML content
        replacements (dict): Dictionary of path replacements
    
    Returns:
        str: Modified XML content
    """
    # Parse the XML
    root = ET.fromstring(xml_content)
    
    # Find all elements with 'file' attribute
    for elem in root.findall(".//*[@file]"):
        file_path = elem.get('file')
        
        # Apply replacements
        new_path = file_path
        for old_path, new_path_base in replacements.items():
            if old_path in file_path:
                # Replace the base path while keeping the relative path
                relative_path = file_path[len(old_path):]
                new_path = new_path_base + relative_path
                break
                
        # Update the file attribute
        elem.set('file', new_path)
    
    # Convert back to string
    return ET.tostring(root, encoding='unicode')

# Define the path replacements
path_replacements = {
    "/Users/yifengz/workspace/libero-dev/chiliocosm/assets/": "/home/lawrence/ATM/libero/assets/",
    "/Users/yifengz/workspace/robosuite-master/robosuite/models/assets/": "/home/lawrence/ATM/third_party/robosuite/robosuite/models/assets/"
}

def main():
    parser = argparse.ArgumentParser(description="Replay demonstrations with modified object colors.")
    parser.add_argument('--original-demo-file', type=str, required=True,
                        help='Path to the original demo file containing multiple demonstrations.')
    parser.add_argument('--preprocessed-demos-folder', type=str, required=True,
                        help='Path to the preprocessed demo folder with one demonstration to reuse metadata.')
    parser.add_argument('--output-demos-folder', type=str, default='new_demos.hdf5',
                        help='Path to the output demo folder to save the new demonstrations.')
    parser.add_argument('--color-mapping', type=str, default='',
                        help='Path to a JSON file containing object color mappings. Example format: {"Cube": [1.0, 0.0, 0.0, 1.0]}')
    parser.add_argument('--bddl-base-path', type=str, required=True,
                        help='Base path where BDDL files are located on your filesystem (e.g., /home/lawrence/ATM/libero/bddl_files/).')
    args = parser.parse_args()

    original_demo_file = args.original_demo_file
    preprocessed_demos_folder = args.preprocessed_demos_folder
    output_demos_folder = args.output_demos_folder
    bddl_base_path = args.bddl_base_path

    # Validate the BDDL base path
    if not os.path.isdir(bddl_base_path):
        print(f"[Error] The specified BDDL base path '{bddl_base_path}' does not exist or is not a directory.")
        return

    # Define the object color mapping
    # You can also load this from a JSON file for flexibility
    if args.color_mapping:
        with open(args.color_mapping, 'r') as f:
            object_color_mapping = json.load(f)
    else:
        # Default color mapping; modify as needed
        object_color_mapping = {
            'akita_black_bowl': [1.0, 0.0, 0.0, 1.0],      # Red
            'plate:': [0.0, 1.0, 0.0, 1.0],                   # Green
            'glazed_rim_porcelain_ramekin': [0.0, 0.0, 1.0, 1.0],  # Blue
            # Add more objects and their colors here
        }

    print("Object Color Mapping:", object_color_mapping)
    print("BDDLM Base Path:", bddl_base_path)

    preprocessed_demo_files = sorted([f for f in os.listdir(preprocessed_demos_folder) if f.endswith('.hdf5')], key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Open the original and preprocessed demo files
    with h5py.File(original_demo_file, 'r') as orig_f:
        demos = sorted(list(orig_f['data'].keys()), key=lambda x: int(x.split('_')[-1]))
        if len(demos) != len(preprocessed_demo_files):
            print(f"[Error] Number of original demos ({len(demos)}) does not match number of preprocessed demos ({len(preprocessed_demo_files)}).")
            return

        for demo_idx, (demo_name, preproc_demo_file) in enumerate(zip(demos, preprocessed_demo_files)):
            preproc_demo_path = os.path.join(preprocessed_demos_folder, preproc_demo_file)
            with h5py.File(preproc_demo_path, 'r') as preproc_f:
                # Copy global attributes from preprocessed demo file to new file
                # Assuming global attributes are under 'root'
                preproc_root = preproc_f['root']
                new_demo_name = f'demo_{demo_idx}.hdf5'
                new_demo_path = os.path.join(output_demos_folder, new_demo_name)
                with h5py.File(new_demo_path, 'w') as new_f:
                    new_root = new_f.create_group('root')

                    # Copy attributes
                    for attr_name, attr_value in preproc_root.attrs.items():
                        new_root.attrs[attr_name] = attr_value
                        print(f"Copied global attribute: {attr_name}")

                    # Copy global datasets like 'task_emb_bert' if they exist
                    if 'task_emb_bert' in preproc_root:
                        new_root.create_dataset('task_emb_bert', data=preproc_root['task_emb_bert'][:])
                        print("Copied dataset: task_emb_bert")

                    # Load environment information from the original demo file
                    env_name = orig_f['data'].attrs['env_name']
                    env_args = json.loads(orig_f['data'].attrs['env_args'])
                    problem_info = json.loads(orig_f['data'].attrs['problem_info'])
                    problem_name = problem_info["problem_name"]
                    language_instruction = problem_info.get("language_instruction", "")

                    print(f"Environment Name: {env_name}")
                    print(f"Problem Name: {problem_name}")
                    print(f"Language Instruction: {language_instruction}")

                    bddl_file_name = orig_f["data"].attrs["bddl_file_name"]
                    bddl_file_name = '/home/lawrence/ATM/' + '/'.join(bddl_file_name.split('/')[1:])

                    # Update environment kwargs
                    libero_utils.update_env_kwargs(
                        env_args['env_kwargs'],
                        bddl_file_name=bddl_file_name,
                        has_renderer=False,
                        has_offscreen_renderer=True,
                        ignore_done=True,
                        use_camera_obs=True,
                        camera_depths=False,  # Set to True if depth is used
                        camera_names=["robot0_eye_in_hand", "agentview"],
                        reward_shaping=True,
                        control_freq=20,
                        camera_heights=128,
                        camera_widths=128,
                        camera_segmentations=None,
                    )

                    # Initialize the environment (without resetting it yet)
                    env = TASK_MAPPING[problem_name](
                        **env_args['env_kwargs'],
                    )
                    print("Environment initialized.")

                    preproc_demo_grp = preproc_f[f'root/']
                    preproc_agentview_grp = preproc_demo_grp['agentview']
                    preproc_eye_in_hand_grp = preproc_demo_grp['eye_in_hand']

                    # Extract tracks and vis from preprocessed demo file
                    preproc_agentview_tracks = preproc_agentview_grp['tracks'][:]
                    preproc_agentview_vis = preproc_agentview_grp['vis'][:]
                    preproc_eye_in_hand_tracks = preproc_eye_in_hand_grp['tracks'][:]
                    preproc_eye_in_hand_vis = preproc_eye_in_hand_grp['vis'][:]

                    # Extract states from preprocessed demo file's extra_states
                    preproc_extra_states_grp = preproc_demo_grp['extra_states']
                    preproc_extra_states = {}
                    for state_name in preproc_extra_states_grp:
                        preproc_extra_states[state_name] = preproc_extra_states_grp[state_name][:]

                    # for dataset_name in preproc_root.keys():
                    #     if dataset_name not in ['agentview', 'eye_in_hand']:
                    #         dataset = preproc_root[dataset_name]
                    #         new_f.create_dataset(dataset_name, data=dataset[:])
                    #         print(f"  Copied dataset: {dataset_name}")

                    print("Loaded tracks, vis, and extra_states from preprocessed demo file.")

                    print(f"\nProcessing demonstration {demo_idx}/{len(demos)}: {demo_name}")

                    orig_demo_grp = orig_f[f'data/{demo_name}']

                    for attr_name, attr_value in orig_demo_grp.attrs.items():
                        # Map the BDDL file path if it's a BDDL file
                        if attr_name == 'bddl_file_name':
                            original_bddl_path = attr_value
                            new_bddl_path = map_bddl_path(original_bddl_path, bddl_base_path)
                            new_root.attrs[attr_name] = new_bddl_path
                            print(f"  Mapped BDDL path from '{original_bddl_path}' to '{new_bddl_path}'")
                        else:
                            new_root.attrs[attr_name] = attr_value
                            print(f"  Copied attribute: {attr_name}")

                    # Copy and save datasets except for videos
                    # Actions
                    actions = orig_f[f'data/{demo_name}/actions'][:]
                    new_root.create_dataset('actions', data=actions)
                    print("  Copied dataset: actions")

                    # Dones
                    if f'data/dones' in orig_f:
                        dones = orig_f[f'data/{demo_name}/dones'][:]
                        new_root.create_dataset('dones', data=dones)
                        print("  Copied dataset: dones")

                    # Rewards
                    if f'data/{demo_name}/rewards' in orig_f:
                        rewards = orig_f[f'data/{demo_name}/rewards'][:]
                        new_root.create_dataset('rewards', data=rewards)
                        print("  Copied dataset: rewards")

                    # States
                    states = orig_f[f'data/{demo_name}/states'][:]
                    new_root.create_dataset('states', data=states)
                    print("  Copied dataset: states")

                    # Robot States
                    robot_states = orig_f[f'data/{demo_name}/robot_states'][:]
                    new_root.create_dataset('robot_states', data=robot_states)
                    print("  Copied dataset: robot_states")

                    # Extra States - Copy from preprocessed demo file
                    extra_states_grp = new_root.create_group('extra_states')
                    for state_name, state_data in preproc_extra_states.items():
                        extra_states_grp.create_dataset(state_name, data=state_data)
                        print(f"  Copied extra_state dataset from preprocessed demo: {state_name}")

                    # Agentview
                    agentview_grp = new_root.create_group('agentview')
                    # Copy tracks and vis from preprocessed demo file
                    agentview_grp.create_dataset('tracks', data=preproc_agentview_tracks)
                    agentview_grp.create_dataset('vis', data=preproc_agentview_vis)
                    print("  Copied datasets: agentview/tracks and agentview/vis from preprocessed demo")

                    # Eye in Hand
                    eye_in_hand_grp = new_root.create_group('eye_in_hand')
                    # Copy tracks and vis from preprocessed demo file
                    eye_in_hand_grp.create_dataset('tracks', data=preproc_eye_in_hand_tracks)
                    eye_in_hand_grp.create_dataset('vis', data=preproc_eye_in_hand_vis)
                    print("  Copied datasets: eye_in_hand/tracks and eye_in_hand/vis from preprocessed demo")

                    # Agentview and Eye in Hand - will be replaced by new videos
                    # Create groups for agentview and eye_in_hand
                    # agentview_grp = new_root.create_group('agentview')
                    # eye_in_hand_grp = new_root.create_group('eye_in_hand')
                    
                    # # Copy tracks and vis from preprocessed demo file
                    # if 'agentview' in preproc_root:
                    #     preproc_agentview = preproc_root['agentview']
                    #     if 'tracks' in preproc_agentview:
                    #         agentview_tracks = preproc_agentview['tracks'][:]
                    #         agentview_grp.create_dataset('tracks', data=agentview_tracks)
                    #         print("    Copied dataset: agentview/tracks")
                    #     if 'vis' in preproc_agentview:
                    #         agentview_vis = preproc_agentview['vis'][:]
                    #         agentview_grp.create_dataset('vis', data=agentview_vis)
                    #         print("    Copied dataset: agentview/vis")
                    # if 'eye_in_hand' in preproc_root:
                    #     preproc_eye_in_hand = preproc_root['eye_in_hand']
                    #     if 'tracks' in preproc_eye_in_hand:
                    #         eye_in_hand_tracks = preproc_eye_in_hand['tracks'][:]
                    #         eye_in_hand_grp.create_dataset('tracks', data=eye_in_hand_tracks)
                    #         print("    Copied dataset: eye_in_hand/tracks")
                    #     if 'vis' in preproc_eye_in_hand:
                    #         eye_in_hand_vis = preproc_eye_in_hand['vis'][:]
                    #         eye_in_hand_grp.create_dataset('vis', data=eye_in_hand_vis)
                    #         print("    Copied dataset: eye_in_hand/vis")

                    # Modify the model XML to change object colors
                    original_model_xml = orig_demo_grp.attrs['model_file']
                    # Check if model_file is a path or XML string
                    if original_model_xml.endswith('.bddl'):
                        # Assume it's a path to the BDDL file
                        if os.path.isfile(original_model_xml):
                            with open(original_model_xml, 'r') as file:
                                model_xml_content = file.read()
                        else:
                            print(f"  [Error] BDDL file '{original_model_xml}' not found. Skipping demo {demo_idx}.")
                            continue
                    else:
                        # Assume it's the XML string
                        model_xml_content = original_model_xml

                    modified_model_xml = change_object_colors(model_xml_content, object_color_mapping)

                    # Fix paths in the modified model XML
                    modified_model_xml = fix_mujoco_paths(modified_model_xml, path_replacements)

                    # Verify model XML changes
                    # pretty_print_xml(modified_model_xml)

                    # Reset the environment with the modified model
                    try:
                        env.reset_from_xml_string(modified_model_xml)
                        env.sim.reset()
                        init_state = orig_demo_grp.attrs['init_state']
                        env.sim.set_state_from_flattened(init_state)
                        env.sim.forward()
                        print("  Environment reset with modified model and initial state.")
                    except Exception as e:
                        print(f"  [Error] Failed to reset environment for demo {demo_idx}: {e}")
                        break

                    # Initialize lists to store new videos
                    agentview_images = []
                    eye_in_hand_images = []

                    # Replay the actions and record new observations
                    for step_idx, action in enumerate(actions):
                        obs, reward, done, info = env.step(action)

                        # Record images after the step
                        agentview_image = obs.get('agentview_image', None)
                        eye_in_hand_image = obs.get('robot0_eye_in_hand_image', None)

                        if agentview_image is not None:
                            agentview_images.append(agentview_image)
                        else:
                            print(f"  [Warning] 'agentview_image' not found at step {step_idx}.")

                        if eye_in_hand_image is not None:
                            eye_in_hand_images.append(eye_in_hand_image)
                        else:
                            print(f"  [Warning] 'eye_in_hand_image' not found at step {step_idx}.")

                        # if done:
                        #     print(f"  [Info] Demo {demo_idx} terminated at step {step_idx}.")
                        #     break  # Proceed to next demo if done

                    # Convert image lists to numpy arrays with appropriate shapes
                    if agentview_images:
                        agentview_video = np.stack(agentview_images, axis=0)  # (num_steps, H, W, C)
                        # Rearrange to (1, num_steps, C, H, W)
                        agentview_video = agentview_video.transpose(0, 3, 1, 2)
                        agentview_video = agentview_video[np.newaxis, ...].astype(np.uint8)
                        agentview_grp.create_dataset('video', data=agentview_video)
                        print("  Saved new agentview video.")
                    else:
                        print("  [Warning] No agentview images recorded.")

                    if eye_in_hand_images:
                        eye_in_hand_video = np.stack(eye_in_hand_images, axis=0)  # (num_steps, H, W, C)
                        # Rearrange to (1, num_steps, C, H, W)
                        eye_in_hand_video = eye_in_hand_video.transpose(0, 3, 1, 2)
                        eye_in_hand_video = eye_in_hand_video[np.newaxis, ...].astype(np.uint8)
                        eye_in_hand_grp.create_dataset('video', data=eye_in_hand_video)
                        print("  Saved new eye_in_hand video.")
                    else:
                        print("  [Warning] No eye_in_hand images recorded.")

                    # Update 'num_samples' attribute based on the number of recorded frames
                    num_samples = len(agentview_images) if agentview_images else 0
                    new_root.attrs['num_samples'] = num_samples
                    print(f"  Set 'num_samples' to {num_samples}.")

                    print(f"  Demonstration {demo_idx} processed successfully.")
                    env.close()

    print("\nAll demonstrations have been processed and saved.")

if __name__ == '__main__':
    main()
