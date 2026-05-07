import os
import sys
sys.stdout.reconfigure(encoding='utf-8')


def check_subtitles(directory):
    files = os.listdir(directory)

    mp4_files = [f for f in files if f.endswith(".mp4")]

    missing_subtitles = 0
    total_files = 0

    for mp4 in mp4_files:
        base_name = os.path.splitext(mp4)[0]
        subtitle = base_name + ".en.vtt"

        if subtitle not in files:
            print(f"Missing subtitle for: {mp4}")
            missing_subtitles += 1

        total_files += 1
    print(f"\nTotal MP4 files: {total_files}")
    print(f"Missing subtitles: {missing_subtitles}")

folder = "./"  # Change this to the desired folder path
check_subtitles(folder)
