import h5py

# Replace 'your_file.h5' with the path to your HDF5 file
with h5py.File('/home/lawrence/ATM/data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo/demo_0.hdf5', 'r') as hdf:
    # Print file-level attributes
    print("File Attributes:")
    for attr in hdf.attrs:
        print(f"{attr}: {hdf.attrs[attr]}")
    
    # If you want to check attributes of datasets or groups within the file
    def print_attrs(name, obj):
        if len(obj.attrs) > 0:
            print(f"\nAttributes of '{name}':")
            for attr in obj.attrs:
                print(f"  {attr}: {obj.attrs[attr]}")

    # Traverse all groups and datasets and print their attributes
    hdf.visititems(print_attrs)

