import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
source_root = "./g_drive_data"
destination_root = "./Data"

# Function to process each folder
def process_folder(folder_name):
    source_folder = os.path.join(source_root, folder_name)
    destination_folder = os.path.join(destination_root, folder_name)

    # If source folder not valid
    if not os.path.isdir(source_folder):
        return (folder_name, 0, 0, "skip")

    # If destination folder missing
    if not os.path.isdir(destination_folder):
        return (folder_name, 0, 0, "missing_destination")

    pose_source = os.path.join(source_folder, "pose")
    pose_destination = os.path.join(destination_folder, "pose")

    # If source pose missing
    if not os.path.exists(pose_source):
        return (folder_name, 0, 0, "missing_source")

    # Create destination pose folder if missing
    os.makedirs(pose_destination, exist_ok=True)

    copied = 0
    skipped = 0

    # Copy files
    for file_name in os.listdir(pose_source):
        source_file = os.path.join(pose_source, file_name)
        dest_file = os.path.join(pose_destination, file_name)

        if os.path.isfile(source_file):
            if not os.path.exists(dest_file):
                shutil.copy2(source_file, dest_file)
                copied += 1
            else:
                skipped += 1

    return (folder_name, copied, skipped, "done")


# Tune this (start with 4–6)
MAX_WORKERS = 6

# Counters
total_folders = 0
total_copied = 0
total_skipped = 0
missing_source_pose = 0
missing_destination_folder = 0

# Run in parallel
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_folder, f) for f in os.listdir(source_root)]

    for future in as_completed(futures):
        folder_name, copied, skipped, status = future.result()

        if status == "done":
            print(f"Processed -> {folder_name} | Copied: {copied}, Skipped: {skipped}")
            total_folders += 1
            total_copied += copied
            total_skipped += skipped

        elif status == "missing_source":
            print(f"Source pose missing -> {folder_name}")
            missing_source_pose += 1

        elif status == "missing_destination":
            print(f"Destination folder missing -> {folder_name}")
            missing_destination_folder += 1


# Final summary
print("\n===== FINAL SUMMARY =====")
print(f"Total folders processed: {total_folders}")
print(f"Total files copied: {total_copied}")
print(f"Total files skipped: {total_skipped}")
print(f"Folders missing source pose: {missing_source_pose}")
print(f"Folders missing destination: {missing_destination_folder}")
print("Done!")