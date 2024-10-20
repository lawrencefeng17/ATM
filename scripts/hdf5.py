import h5py

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

# Replace 'your_file.h5' with the path to your HDF5 file
print_hdf5_structure('/home/lawrence/ATM/data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo/demo_0.hdf5')

