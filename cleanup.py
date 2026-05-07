import os
import shutil

def identify_and_clean(data_dir, video_dir, trash_dir="Trash_Bin"):
    # 1. IDENTIFICATION PHASE
    to_remove = []
    
    # Check if directories exist
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found.")
        return

    print(f"--- Scanning {data_dir} for empty transcript folders ---")
    
    video_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    for folder in video_folders:
        text_folder_path = os.path.join(data_dir, folder, "text")
        
        is_empty = True
        if os.path.exists(text_folder_path):
            # Check for files with actual content (size > 0 bytes)
            valid_files = [f for f in os.listdir(text_folder_path) 
                           if os.path.getsize(os.path.join(text_folder_path, f)) > 0]
            if len(valid_files) > 0:
                is_empty = False
        
        if is_empty:
            to_remove.append(folder)
            print(f"Found empty: {folder}")

    # 2. SUMMARY & CONFIRMATION
    count = len(to_remove)
    if count == 0:
        print("\nAll folders have valid transcripts. Nothing to remove!")
        return

    print("\n" + "="*40)
    print(f"SUMMARY: Found {count} folders with missing/empty transcripts.")
    print("="*40)
    
    confirm = input(f"\nDo you want to move these {count} folders (and their corresponding videos) to '{trash_dir}'? (type 'yes' to proceed): ").lower()

    # 3. EXECUTION PHASE
    if confirm == 'yes':
        trash_data = os.path.join(trash_dir, "Data")
        trash_video = os.path.join(trash_dir, "Video")
        
        for d in [trash_data, trash_video]:
            if not os.path.exists(d):
                os.makedirs(d)

        for folder in to_remove:
            data_path = os.path.join(data_dir, folder)
            video_path = os.path.join(video_dir, folder)
            
            try:
                # Move Data folder
                shutil.move(data_path, os.path.join(trash_data, folder))
                
                # Move Video folder if it exists
                if os.path.exists(video_path):
                    shutil.move(video_path, os.path.join(trash_video, folder))
                
                print(f"Moved: {folder}")
            except Exception as e:
                print(f"Error moving {folder}: {e}")
        
        print(f"\nSuccess: {count} folders relocated to '{trash_dir}'.")
    else:
        print("\nOperation cancelled. No files were moved.")

if __name__ == "__main__":
    # Ensure these match your actual folder names
    DATA_DIR = "Data"
    VIDEO_DIR = "Video"
    
    identify_and_clean(DATA_DIR, VIDEO_DIR)