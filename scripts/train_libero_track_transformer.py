import os
import argparse
from glob import glob


# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_goal",  
                    help="The name of the desired suite, where libero_10 is the alias of libero_long.")
args = parser.parse_args()

# training configs
CONFIG_NAME = "libero_track_transformer"

gpu_ids = [0]

root_dir = "./data/atm_libero/"
suite_name = args.suite

# setup number of epoches and dataset path
if suite_name == "libero_100":
    EPOCH = 301
    train_dataset_list = glob(os.path.join(root_dir, "libero_90/*/train/")) + glob(os.path.join(root_dir, "libero_10/*/train/"))
    val_dataset_list = glob(os.path.join(root_dir, "libero_90/*/val/")) + glob(os.path.join(root_dir, "libero_10/*/val/"))
else:
    EPOCH = 1001
    # train_dataset_list = glob(os.path.join(root_dir, f"{suite_name}"))
    train_dataset_list = [f"{root_dir}/{suite_name}/{task_dir}" for task_dir in os.listdir(os.path.join(root_dir, suite_name))]
    val_dataset_list = [f"{root_dir}/{suite_name}/{task_dir}" for task_dir in os.listdir(os.path.join(root_dir, suite_name))]
    # val_dataset_list = ["./data/atm_libero//libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo"]

command = (f'python -m engine.train_track_transformer --config-name={CONFIG_NAME} '
           f'train_gpus="{gpu_ids}" '
           f'experiment={CONFIG_NAME}_{suite_name.replace("_", "-")}_ep{EPOCH} '
           f'epochs={EPOCH} '
           f'train_dataset="{train_dataset_list}" val_dataset="{val_dataset_list}" ')

os.system(command)
