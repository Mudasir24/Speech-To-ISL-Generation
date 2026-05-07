# ISL Speech-to-Sign Language Translation

An end-to-end deep learning pipeline that translates **spoken audio into Indian Sign Language (ISL) pose sequences**. Given an audio clip, the system produces a skeleton animation of a signer performing the equivalent sign language, bridging the communication gap between hearing and Deaf communities.

This project is inspired by and builds upon the **Progressive Transformers for End-to-End Sign Language Production** framework ([Saunders et al., 2020](https://arxiv.org/abs/2004.14874)), adapted and extended for Indian Sign Language using a custom YouTube-sourced dataset.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Step-by-Step Usage](#step-by-step-usage)
  - [Step 1: Download Dataset](#step-1-download-dataset)
  - [Step 2: Preprocess](#step-2-preprocess)
  - [Step 3: Train the Model](#step-3-train-the-model)
  - [Step 4: Test and Evaluate](#step-4-test-and-evaluate)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Overview

The system takes a raw audio signal as input and outputs a time-series of 3D skeletal keypoints representing ISL signs. The full pipeline consists of four major stages:

1. **Dataset Collection** — YouTube ISL videos are downloaded along with auto-generated subtitles.
2. **Preprocessing** — Videos are segmented by subtitle timing, audio features (mel-spectrograms) are extracted, and body/hand pose keypoints are estimated using OpenPose.
3. **3D Pose Lifting** — 2D OpenPose keypoints are lifted to 3D using an Inverse Kinematics (IK) model.
4. **Transformer Training** — A Progressive Transformer (encoder-decoder) is trained to map audio mel-spectrograms to 3D skeletal pose sequences, with an optional adversarial discriminator for improved realism.

---

## Pipeline Architecture

```
Audio Input (WAV, 16kHz)
        │
        ▼
  Mel-Spectrogram (80 bins)
        │
        ▼
  Transformer Encoder
  (2 layers, 8 heads, hidden=512)
        │
        ▼
  Transformer Decoder  ◄──── GT Counter (timing signal)
  (2 layers, 8 heads, hidden=512)
        │
        ▼
  3D Pose Output (150 dims)
  50 keypoints × 3 (x, y, z)
  [8 body + 21 left hand + 21 right hand]
        │
        ▼
  Skeleton Video Rendering
```

The 150-dimensional output represents 50 keypoints across the body and both hands, derived from OpenPose detections lifted to 3D via the IK pipeline.

---

## Project Structure

```
.
├── Dataset/
│   ├── train.txt                  # YouTube video IDs for training
│   ├── dev.txt                    # YouTube video IDs for validation
│   └── test.txt                   # YouTube video IDs for testing
│
├── Download Dataset/
│   ├── download_videos.py         # Downloads MP4 + VTT subtitles from YouTube
│   ├── duplicates.py              # Detects duplicate files in a directory
│   └── pair.py                    # Checks MP4/VTT pairing consistency
│
├── Preprocess/
│   ├── cut_segments.py            # Cuts videos into subtitle-aligned segments
│   ├── audio_preprocessing.py     # Mel-spectrogram feature extraction
│   ├── cleanup.py                 # Removes folders with missing transcripts
│   ├── organizing_data_files.py   # Copies pose files to the data directory
│   ├── save_preprocessed_data.py  # Aggregates audio, pose, and text into datasets
│   ├── pose_extraction.ipynb      # Google Colab notebook for OpenPose keypoint extraction
│   └── Inverse Kinematics/
│       ├── convert_openpose_to_h5.py  # Converts OpenPose JSON → H5 format
│       ├── convert_txt_to_json.py     # Converts IK output TXT → JSON
│       ├── demo.py                    # Runs the full 2D → 3D IK pipeline
│       ├── pose2D.py                  # 2D pose normalization, pruning, interpolation
│       ├── pose2Dto3D.py              # Initial 3D pose estimation from 2D keypoints
│       ├── pose3D.py                  # TensorFlow backpropagation-based 3D refinement
│       └── skeletalModel.py           # Skeletal tree structure definition (50 joints)
│
├── Transformer Training/
│   ├── __main__.py                # Entry point (train / test)
│   ├── model.py                   # Full encoder-decoder model definition
│   ├── training.py                # Training loop, validation, checkpointing
│   ├── data.py                    # Dataset loading and SignProdDataset class
│   ├── encoders.py                # Transformer encoder (audio)
│   ├── decoders.py                # Transformer decoder (pose)
│   ├── transformer_layers.py      # Multi-head attention, positional encoding, FFN
│   ├── embeddings.py              # Embedding layers with masked normalization
│   ├── discriminator.py           # Adversarial discriminator (optional)
│   ├── loss.py                    # MSE / L1 regression loss + cross-entropy loss
│   ├── prediction.py              # Inference loop, DTW and PCK evaluation
│   ├── search.py                  # Greedy autoregressive decoding
│   ├── batch.py                   # Batch preparation with masks
│   ├── builders.py                # Optimizer and scheduler factory functions
│   ├── helpers.py                 # Utilities (DTW, PCK, checkpointing, logging)
│   ├── dtw.py                     # Dynamic Time Warping implementation
│   ├── vocabulary.py              # Vocabulary class for text tokens
│   ├── constants.py               # Global constants (PAD, BOS, EOS tokens)
│   ├── initialization.py          # Xavier / uniform / normal weight initializers
│   ├── plot_videos.py             # Skeleton animation rendering with OpenCV
│   ├── test_dtw.py                # DTW score aggregation from saved test videos
│   └── Configs/
│       └── Base.yaml              # Main configuration file
│
├── data_aud_text/                 # Generated: preprocessed training data
│   ├── train.audio / .skels / .text / .files
│   ├── dev.audio   / .skels / .text / .files
│   └── test.audio  / .skels / .text / .files
│
├── Models/                        # Generated: saved model checkpoints
│   └── best_model.ckpt
│
└── requirements.txt
```

---

## Requirements

- Python 3.8+
- NVIDIA GPU (4GB VRAM minimum, 8GB+ recommended)
- FFmpeg (for audio/video processing)
- ~100–200 GB storage for the full pipeline

### Key Python Dependencies

| Package | Version |
|---|---|
| torch | 1.9.0 |
| torchtext | 0.6.0 |
| librosa | 0.11.0 |
| numpy | 1.23.5 |
| scipy | 1.9.3 |
| h5py | 3.14.0 |
| opencv-python | latest |
| nltk | 3.9.2 |
| soundfile | 0.13.1 |
| tensorboard | 2.20.0 |
| webvtt-py | 0.5.1 |
| yt-dlp | 2025.10.14 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Mudasir24/Speech-To-ISL-Generation.git
cd isl-speech-to-sign

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (Ubuntu/Debian)
sudo apt-get install ffmpeg

# Download NLTK data (required for text lemmatization)
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

---

## Step-by-Step Usage

### Step 1: Download Dataset

The dataset consists of ISL YouTube videos. Video IDs are listed in `train.txt`, `dev.txt`, and `test.txt` (one ID per line).

**Update the path in `download_videos.py`** to point to your `train.txt`, then run:

```bash
python download_videos.py
```

This downloads each video as `.mp4` along with its English auto-subtitle file (`.en.vtt`).

**Verify the downloads:**

```bash
# Check for duplicate files
python duplicates.py

# Check that every MP4 has a matching .vtt subtitle
python pair.py
```

> Edit the `folder` variable in `duplicates.py` and `pair.py` to point to your download directory.

---

### Step 2: Preprocess

The preprocessing pipeline transforms raw videos into aligned audio features, pose sequences, and text transcripts.

#### 2a. Segment Videos by Subtitle

```bash
python cut_segments.py
```

Edit the directory variables at the top of the script:

```python
src_dir = "path/to/downloaded/videos"   # folder with .mp4 and .vtt files
dst_dir = "Data"                         # output folder for audio + text
vid_dir = "Video"                        # output folder for video segments
```

Output per video segment:
- `Data/<video_name>/audio/<segment>.wav` — 16kHz WAV audio
- `Data/<video_name>/text/<segment>.txt` — transcript text
- `Video/<video_name>/<segment>.mp4` — video clip

#### 2b. Extract Pose Keypoints (GPU Required)

Open `pose_extraction.ipynb` in **Google Colab** (GPU runtime recommended). The notebook:
- Installs and builds OpenPose from source
- Mounts your Google Drive containing the segmented videos
- Runs OpenPose to extract 50 body/hand keypoints per frame
- Saves JSON files to `Data/<video_name>/pose/`

**Keypoint layout (50 keypoints × 3 values each = 150 dims):**
- Indices 0–23: Body (Nose, Neck, shoulders, elbows, wrists)
- Indices 24–86: Left hand (21 keypoints)
- Indices 87–149: Right hand (21 keypoints)

#### 2c. Lift 2D Poses to 3D (Inverse Kinematics)

```bash
# Step 1: Convert OpenPose JSONs to H5 format
python convert_openpose_to_h5.py

# Step 2: Run the 2D → 3D IK pipeline
python demo.py

# Step 3: Convert IK output TXT to JSON
python convert_txt_to_json.py
```

This lifts flat 2D screen coordinates to 3D skeletal coordinates using a backpropagation-based IK model (pose3D.py with TensorFlow).

#### 2d. Aggregate into Training Data

```bash
python save_preprocessed_data.py
```

Update the `path` variable to your `Data/` directory. This script combines audio mel-spectrograms, 3D pose sequences, and text transcripts into flat training files:

```
data_aud_text/
  train.audio   train.skels   train.text   train.files
  dev.audio     dev.skels     dev.text     dev.files
  test.audio    test.skels    test.text    test.files
```

Data split: **70% train / 15% dev / 15% test**

#### 2e. (Optional) Clean Up

```bash
python cleanup.py
```

Removes video folders that have empty or missing transcript files.

---

### Step 3: Train the Model

```bash
python __main__.py train ./Configs/Base.yaml
```

Training logs and checkpoints are saved to `./Models/` by default. Monitor training with TensorBoard:

```bash
tensorboard --logdir=Models/tensorboard
```

**Key configuration options** in `Configs/Base.yaml`:

```yaml
data:
  train: "./data_aud_text/train"
  dev:   "./data_aud_text/dev"
  test:  "./data_aud_text/test"

training:
  batch_size: 32          # Reduce if out of GPU memory
  learning_rate: 0.001
  epochs: 2000
  use_cuda: True          # Set False for CPU-only
  early_stopping_metric: "dtw"

model:
  src_size: 80            # Mel-spectrogram bins (fixed)
  trg_size: 150           # Pose output dimensions (fixed)
  encoder:
    num_layers: 2
    num_heads: 8
    hidden_size: 512
  decoder:
    num_layers: 2
    num_heads: 8
    hidden_size: 512
```

---

### Step 4: Test and Evaluate

```bash
python __main__.py test ./Configs/Base.yaml Models/best_model.ckpt
```

This runs inference on the test set and saves skeleton animation videos to `Models/test_videos/`. Each video filename includes its DTW score.

To compute the average DTW score across all test videos:

```bash
python test_dtw.py Models/test_videos/
```

---

## Configuration

All training hyperparameters live in `Configs/Base.yaml`. Key sections:

| Parameter | Default | Description |
|---|---|---|
| `src_fps` | 100 | Audio feature frames per second |
| `trg_fps` | 25 | Target pose frames per second |
| `num_sec` | 6 | Maximum audio clip length (seconds) |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.001 | Initial learning rate |
| `patience` | 100 | Epochs before LR reduction |
| `decrease_factor` | 0.7 | LR reduction multiplier |
| `future_prediction` | 10 | Future frame prediction horizon |
| `regloss_weight` | 1.0 | Weight for pose regression loss (MSE) |
| `xentloss_weight` | 0.001 | Weight for text cross-entropy loss |
| `advloss_weight` | 0.0001 | Weight for adversarial loss |

---

## Model Architecture

The model is a sequence-to-sequence **Progressive Transformer**:

**Encoder** — Processes mel-spectrogram audio features (80-dim per frame) through a stack of Transformer encoder layers with multi-head self-attention and positional encoding.

**Decoder** — Autoregressively generates 3D pose sequences (150-dim per frame). A counter signal is injected at each decoding step to drive timing, avoiding the need for explicit length prediction.

**Discriminator (optional)** — A cross-modal classifier that takes paired audio and pose sequences and learns to distinguish real from generated motion, providing an adversarial training signal.

**Loss functions:**
- **MSE Loss** — Primary regression loss between predicted and ground-truth pose sequences, applied only over non-padded frames.
- **Cross-Entropy Loss** — Secondary loss on a parallel text prediction head, encouraging the model to remain semantically grounded.
- **Adversarial Loss** — Optional discriminator loss to improve motion naturalness.

---

## Evaluation Metrics

**DTW (Dynamic Time Warping)** — Measures the similarity between the predicted and reference pose sequences while allowing for temporal warping. Lower is better (0 = perfect match). Used as the primary training and validation metric.

**PCK (Percentage of Correct Keypoints)** — Measures what fraction of predicted keypoints fall within a threshold distance of the ground truth. Higher is better (1.0 = perfect). Reported at α = 0.2.

---

## Troubleshooting

**Training loss doesn't decrease**
- Lower `learning_rate` (try 0.0005)
- Verify data paths in `Base.yaml` are correct
- Ensure GPU is available (`use_cuda: True`)

**Out of GPU memory**
- Reduce `batch_size` (try 16 or 8)
- Reduce `hidden_size` (try 256 or 384)

**NaN loss (training explodes)**
- Reduce `learning_rate`
- Increase gradient clipping: `clip_grad_norm: 10`

**Pose extraction is very slow**
- Use Google Colab with a T4 GPU runtime via `pose_extraction.ipynb`

**`ffmpeg` not found**
- Install from [https://ffmpeg.org](https://ffmpeg.org) or via `sudo apt-get install ffmpeg`

**`yt-dlp` errors on some videos**
- Expected behavior — videos may be unavailable or region-locked. The `--ignore-errors` flag handles this automatically.

---

## Acknowledgements

This project builds upon the **Progressive Transformers for End-to-End Sign Language Production** framework by Ben Saunders, Necati Cihan Camgoz, and Richard Bowden (ECCV 2020). Their work on counter-based progressive decoding for sign language generation forms the core of the transformer architecture used here.

- Paper: [https://arxiv.org/abs/2004.14874](https://arxiv.org/abs/2004.14874)
- Original DTW implementation: [https://github.com/pierre-rouanet/dtw](https://github.com/pierre-rouanet/dtw)
- Pose estimation: [CMU OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
