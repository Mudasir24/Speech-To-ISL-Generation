#!/usr/bin/env python3
"""
Convert OpenPose JSON keypoint files to H5 format matching demo-sequence.h5.

JSON files are named like:  0000_000000000000_keypoints.json
                             ^^^^  segment prefix
Files are grouped by segment prefix → one H5 per segment.

Output H5 structure:
  Dataset name: <segment_prefix>  (e.g. "0000")
  Shape: (N_frames, 150)  — 50 keypoints x 3 values (x, y, likelihood)
  Dtype: float32

50 keypoints layout:
  [  0- 23]  Body pose kps 0-7: Nose, Neck, RShoulder, RElbow, RWrist,
                                  LShoulder, LElbow, LWrist  (8 x 3)
  [ 24- 86]  Left hand 21 keypoints                          (21 x 3)
  [ 87-149]  Right hand 21 keypoints                         (21 x 3)

Output folder: OP_intermediate/   (created next to input_dir if not specified)

Usage:
  python convert_openpose_to_h5.py --input_dir <pose_folder> [--output_dir <OP_intermediate>]
"""

import os
import re
import json
import glob
import argparse
import numpy as np
import h5py
from collections import defaultdict


def extract_keypoints_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = np.zeros(150, dtype=np.float32)
    people = data.get('people', [])
    if not people:
        return result

    person = people[0]

    # Body pose: keypoints 0-7 (Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist)
    pose_kps = person.get('pose_keypoints_2d', [])
    n_body = min(len(pose_kps[:24]), 24)
    result[0:n_body] = pose_kps[:n_body]

    # Left hand: 21 keypoints
    lhand_kps = person.get('hand_left_keypoints_2d', [])
    n_lhand = min(len(lhand_kps), 63)
    result[24:24 + n_lhand] = lhand_kps[:n_lhand]

    # Right hand: 21 keypoints
    rhand_kps = person.get('hand_right_keypoints_2d', [])
    n_rhand = min(len(rhand_kps), 63)
    result[87:87 + n_rhand] = rhand_kps[:n_rhand]

    return result


def group_files_by_segment(input_dir):
    """Group *_keypoints.json files by the leading segment prefix (e.g. '0000')."""
    all_files = sorted(glob.glob(os.path.join(input_dir, '*_keypoints.json')))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    if not all_files:
        raise FileNotFoundError(f"No JSON keypoint files found in: {input_dir}")

    segments = defaultdict(list)
    for fpath in all_files:
        match = re.match(r'^(\d+)_', os.path.basename(fpath))
        seg = match.group(1) if match else 'unknown'
        segments[seg].append(fpath)

    return dict(sorted(segments.items()))


def convert_segment(json_files, output_h5, dataset_name):
    frames = []
    for i, jf in enumerate(json_files):
        frames.append(extract_keypoints_from_json(jf))
        if (i + 1) % 100 == 0:
            print(f"    [{dataset_name}] {i+1}/{len(json_files)} frames processed...")

    data = np.array(frames, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_h5)), exist_ok=True)
    with h5py.File(output_h5, 'w') as hf:
        hf.create_dataset(dataset_name, data=data, dtype='float32')
    print(f"  OK  {output_h5}  |  dataset='{dataset_name}'  shape={data.shape}")


def main():

    base_data_dir = "./Data"

    print(f"Scanning base directory: {base_data_dir}\n")

    all_folders = sorted(os.listdir(base_data_dir))
    total_processed = 0
    total_skipped = 0

    for folder in all_folders:
        folder_path = os.path.join(base_data_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        input_dir = os.path.join(folder_path, "pose")

        if not os.path.exists(input_dir):
            print(f"[SKIP] No pose folder in: {folder}")
            continue

        output_dir = os.path.join(folder_path, "OP_intermediate")

        # ✅ Skip if already processed
        if os.path.exists(output_dir):
            existing_h5 = glob.glob(os.path.join(output_dir, "*.h5"))
            if len(existing_h5) > 0:
                print(f"[SKIP] Already processed: {folder}")
                total_skipped += 1
                continue

        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print(f"Processing folder: {folder}")
        print(f"Input  : {input_dir}")
        print(f"Output : {output_dir}\n")

        try:
            segments = group_files_by_segment(input_dir)
            print(f"Found {len(segments)} segment(s): {list(segments.keys())}\n")

            MIN_FRAMES = 10
            for seg, files in segments.items():
                if len(files) < MIN_FRAMES:
                    print(f"Segment '{seg}': {len(files)} frames (less than {MIN_FRAMES}) - skipping")
                    continue

                output_h5 = os.path.join(output_dir, f"{seg}.h5")
                print(f"Segment '{seg}': {len(files)} frames -> {output_h5}")
                convert_segment(files, output_h5, dataset_name=seg)

            total_processed += 1

        except Exception as e:
            print(f"[ERROR] Failed for folder {folder}: {e}")

    print("\n" + "=" * 60)
    print(f"Done.")
    print(f"Processed folders : {total_processed}")
    print(f"Skipped folders   : {total_skipped}")

if __name__ == '__main__':
    main()