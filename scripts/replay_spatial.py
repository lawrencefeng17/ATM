import os
import argparse

# python replay_demonstrations.py --original-demo-file /home/lawrence/ATM/data/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5 
# --preprocessed-demos-folder /home/lawrence/ATM/data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo/ 
# --output-demos-folder /home/lawrence/ATM/data/atm_libero/colored/test/ 
# --bddl-base-path  /home/lawrence/ATM/libero/bddl_files/

output_demos_folder = "/home/lawrence/ATM/data/atm_libero/colored/libero_spatial"
bddl_base_path = "/home/lawrence/ATM/libero/bddl_files/"
original = "/home/lawrence/ATM/data/libero_spatial"
preprocessed = "/home/lawrence/ATM/data/atm_libero/libero_spatial"
for task in os.listdir(preprocessed):
    print("Executing command: ", f"python replay_demonstrations.py --original-demo-file {os.path.join(original, task)} ")
    if not os.path.exists(os.path.join(output_demos_folder, task)):
        os.mkdir(os.path.join(output_demos_folder, task))
    os.system(f"python replay_demonstrations.py --original-demo-file {os.path.join(original, task)}.hdf5 "
               f"--preprocessed-demos-folder {os.path.join(preprocessed, task)} "
               f"--output-demos-folder {os.path.join(output_demos_folder, task)} --bddl-base-path {bddl_base_path}")
    
