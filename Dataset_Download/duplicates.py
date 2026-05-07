import os
from collections import defaultdict

def find_duplicate_names(directory):
    files_dict = defaultdict(list)

    for root, dirs, files in os.walk(directory):
        for file in files:
            files_dict[file].append(os.path.join(root, file))

    duplicates = {name: paths for name, paths in files_dict.items() if len(paths) > 1}
    return duplicates


if __name__ == "__main__":
    folder = "./"  # Change this to the desired folder path

    duplicates = find_duplicate_names(folder)

    if duplicates:
        print("\nFiles with same name and extension:\n")
        for name, paths in duplicates.items():
            print(f"{name}:")
            for path in paths:
                print("  ", path)
            print()
    else:
        print("No duplicates found.")