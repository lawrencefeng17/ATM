import h5py
import argparse

def print_hdf5_structure(file_name):
    with h5py.File(file_name, 'r') as file:
        def print_structure(name, obj):
            indent = " " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}{name}/ (Group)")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}{name} (Dataset) - shape: {obj.shape}, dtype: {obj.dtype}")
            else:
                print(f"{indent}{name} (Unknown object)")

        file.visititems(print_structure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, required=True)

    args = parser.parse_args()

    print_hdf5_structure(args.f)

