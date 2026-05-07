import numpy as np
import json
import os
import glob

# Base directory
base_data_dir = "./Data"

all_folders = sorted(os.listdir(base_data_dir))

total_processed = 0
total_skipped = 0

print(f"Scanning {len(all_folders)} folders...\n")

for folder in all_folders:
    folder_path = os.path.join(base_data_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    input_folder = os.path.join(folder_path, "OP_0")
    output_folder = os.path.join(folder_path, "OP")

    # Skip if no OP_0
    if not os.path.exists(input_folder):
        print(f"[SKIP] No OP_0 in: {folder}")
        total_skipped += 1
        continue

    os.makedirs(output_folder, exist_ok=True)

    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))

    if not txt_files:
        print(f"[SKIP] No TXT files in: {folder}")
        total_skipped += 1
        continue

    print("=" * 60)
    print(f"Processing folder: {folder}")
    print(f"Found {len(txt_files)} files\n")

    for file_path in txt_files:
        try:
            filename = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_folder, filename + ".json")

            # ✅ Skip if already converted
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"[SKIP] Already done: {filename}")
                continue

            # Load data
            data = np.loadtxt(file_path)

            # Ensure 2D
            if data.ndim == 1:
                data = data.reshape(1, -1)

            print(f"Processing: {filename} | Shape: {data.shape}")

            # Save JSON
            with open(output_path, "w") as f:
                json.dump(data.tolist(), f)

            total_processed += 1

        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")

print("\n" + "=" * 60)
print("Done.")
print(f"Processed files : {total_processed}")
print(f"Skipped folders : {total_skipped}")
