# standard
import h5py
import os
import glob

# 3rd party
import numpy

# our own 
import skeletalModel
import pose2D
import pose2Dto3D
import pose3D 


def save(fname, lst):
    T, dim = lst[0].shape
    with open(fname, "w") as f:
        for t in range(T):
            for i in range(dim):
                for j in range(len(lst)):
                    f.write("%e\t" % lst[j][t, i])
            f.write("\n")


if __name__ == "__main__":
    
    dtype = "float32"
    randomNubersGenerator = numpy.random.RandomState(1234)

    structure = skeletalModel.getSkeletalModelStructure()

    # Folders
    base_data_dir = "./Data"

    all_folders = sorted(os.listdir(base_data_dir))

    total_parts = 1
    part_index = 0  # Change this index to process different parts (0 to total_parts-1)

    chunk_size = len(all_folders) // total_parts

    start = part_index * chunk_size
    end = (part_index + 1) * chunk_size if part_index < total_parts - 1 else len(all_folders)

    print(f"\n=== PART {part_index + 1} | Folders {start} to {end - 1} ===\n")
    total_folders_in_part = end - start

    for idx, folder in enumerate(all_folders[start:end], start=1):
        print(f"Processing {idx}/{total_folders_in_part}: {folder}")
        folder_path = os.path.join(base_data_dir, folder)

        if not os.path.isdir(folder_path):
            continue
        
        input_folder = os.path.join(folder_path, "OP_intermediate")

        if not os.path.exists(input_folder):
            print(f"[SKIP] No OP_intermediate in: {folder}")
            continue
        
        output_demo5_folder = os.path.join(folder_path, "OP_0")

        # Create folders if not exist
        os.makedirs(output_demo5_folder, exist_ok=True)

        # Loop through all .h5 files
        h5_files = glob.glob(os.path.join(input_folder, "*.h5"))

        if not h5_files:
            print(f"[SKIP] No .h5 files in: {input_folder}")
            continue
        
        print("=" * 60)
        print(f"Processing folder: {folder}")

        for h5_path in h5_files:
            base_name = os.path.splitext(os.path.basename(h5_path))[0]

            output_demo5_path = os.path.join(output_demo5_folder, f"{base_name}.txt")

            if os.path.exists(output_demo5_path) and os.path.getsize(output_demo5_path) > 0:
                print(f"[SKIP] Already processed: {base_name}")
                continue


            print(f"Processing: {base_name}")

            # Read H5 use dynamic dataset name
            with h5py.File(h5_path, "r") as f:
                dataset_name = list(f.keys())[0]  
                inputSequence_2D = numpy.array(f[dataset_name])

            # Split
            X = inputSequence_2D
            Xx = X[:, 0::3]
            Xy = X[:, 1::3]
            Xw = X[:, 2::3]

            # Normalize
            Xx, Xy = pose2D.normalization(Xx, Xy)
            # save(os.path.join(output_demo14_folder, f"{base_name}_demo1.txt"), [Xx, Xy, Xw])

            # Prune
            Xx, Xy, Xw = pose2D.prune(Xx, Xy, Xw, (0,1,2,3,4,5,6,7), 0.3, dtype)
            # save(os.path.join(output_demo14_folder, f"{base_name}_demo2.txt"), [Xx, Xy, Xw])

            # Interpolation
            Xx, Xy, Xw = pose2D.interpolation(Xx, Xy, Xw, 0.99, dtype)
            # save(os.path.join(output_demo14_folder, f"{base_name}_demo3.txt"), [Xx, Xy, Xw])

            # Initial 3D
            lines0, rootsx0, rootsy0, rootsz0, anglesx0, anglesy0, anglesz0, Yx0, Yy0, Yz0 = pose2Dto3D.initialization(
                Xx, Xy, Xw, structure, 0.001, randomNubersGenerator, dtype
            )
            # save(os.path.join(output_demo14_folder, f"{base_name}_demo4.txt"), [Yx0, Yy0, Yz0])

            # Final 3D
            Yx, Yy, Yz = pose3D.backpropagationBasedFiltering(
                lines0, rootsx0, rootsy0, rootsz0,
                anglesx0, anglesy0, anglesz0,
                Xx, Xy, Xw,
                structure,
                dtype,
            )

            # ⭐ Save demo5 in OP folder with same name as .h5
            output_demo5_path = os.path.join(output_demo5_folder, f"{base_name}.txt")
            save(output_demo5_path, [Yx, Yy, Yz])

    print("All files processed successfully!")