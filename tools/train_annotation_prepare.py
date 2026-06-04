"""
Script to prepare training annotation files for RS-STE.

Two-stage training requires different annotation formats:

1. **Synth Pair Training (Stage 2)**:
   Uses paired synthetic data. Each sample has:
   - image1_paths: source image paths
   - image2_paths: target image paths (paired ground truth)
   - image1_rec: source text
   - image2_rec: target text

   The synthetic dataset can be downloaded from:
   https://huggingface.co/datasets/Nineve/RS-STE-Dataset

2. **Real Cycle Training (Stage 3)**:
   Uses unpaired real data. Each sample has:
   - image1_paths: source image paths (real scene text images)
   - image1_rec: source text (ground truth recognition)
   - image2_rec: target text (randomly perturbed text for editing)

   No image2_paths needed (cycle-consistency training).

Usage:
    # For synthetic pair data (already paired):
    python tools/train_annotation_prepare.py --mode synth --data_dir /path/to/synth_dataset --output data/annotation/train_annotations.pkl

    # For real data (prepare cycle training):
    python tools/train_annotation_prepare.py --mode real --data_dir /path/to/real_images --rec_file /path/to/recognition_labels.txt --output data/annotation/real_train.pkl
"""

import os
import sys
import pickle
import argparse
import random
from tqdm import tqdm


def prepare_synth_pair_annotations(data_dir, output_file, train_ratio=0.9, seed=42):
    """
    Prepare annotations for synthetic pair training.

    The synthetic data directory should be organized as:
        data_dir/
            train-1M-1/
                i_s.txt   # source text
                i_t.txt   # target text
                *.png      # source images (paired)
            train-1M-2/
                ...
            test/
                ...

    Or the HuggingFace RS-STE-Dataset format.
    """
    random.seed(seed)

    annotations = {
        "image1_paths": [],
        "image2_paths": [],
        "image1_rec": [],
        "image2_rec": [],
    }

    # Collect all subdirectories
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        i_s_file = os.path.join(subdir_path, "i_s.txt")
        i_t_file = os.path.join(subdir_path, "i_t.txt")

        if not os.path.exists(i_s_file) or not os.path.exists(i_t_file):
            print(f"Skipping {subdir}: missing i_s.txt or i_t.txt")
            continue

        with open(i_s_file, "r") as f:
            source_texts = f.read().strip().split("\n")
        with open(i_t_file, "r") as f:
            target_texts = f.read().strip().split("\n")

        # Get all PNG images in the subdirectory
        images = sorted([f for f in os.listdir(subdir_path) if f.endswith(".png")])

        assert len(source_texts) == len(target_texts) == len(images), \
            f"Mismatch in {subdir}: {len(source_texts)} src texts, {len(target_texts)} tgt texts, {len(images)} images"

        for img_name, src_text, tgt_text in zip(images, source_texts, target_texts):
            img_path = os.path.join(subdir_path, img_name)
            annotations["image1_paths"].append(img_path)
            annotations["image2_paths"].append(img_path)  # Paired: source=target image, text changes
            annotations["image1_rec"].append(src_text)
            annotations["image2_rec"].append(tgt_text)

        print(f"Loaded {len(images)} samples from {subdir}")

    total = len(annotations["image1_paths"])
    print(f"Total samples: {total}")

    if total == 0:
        print("Warning: No samples found. Please check the data directory structure.")
        return

    # Split into train/test
    indices = list(range(total))
    random.shuffle(indices)
    split_idx = int(total * train_ratio)
    train_indices = sorted(indices[:split_idx])
    test_indices = sorted(indices[split_idx:])

    train_annotations = {
        "image1_paths": [annotations["image1_paths"][i] for i in train_indices],
        "image2_paths": [annotations["image2_paths"][i] for i in train_indices],
        "image1_rec": [annotations["image1_rec"][i] for i in train_indices],
        "image2_rec": [annotations["image2_rec"][i] for i in train_indices],
    }

    test_annotations = {
        "image1_paths": [annotations["image1_paths"][i] for i in test_indices],
        "image2_paths": [annotations["image2_paths"][i] for i in test_indices],
        "image1_rec": [annotations["image1_rec"][i] for i in test_indices],
        "image2_rec": [annotations["image2_rec"][i] for i in test_indices],
    }

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(train_annotations, f)
    print(f"Train annotations saved to {output_file} ({len(train_annotations['image1_paths'])} samples)")

    test_file = output_file.replace(".pkl", "_test.pkl")
    with open(test_file, "wb") as f:
        pickle.dump(test_annotations, f)
    print(f"Test annotations saved to {test_file} ({len(test_annotations['image1_paths'])} samples)")


def prepare_real_cycle_annotations(rec_file, image_dir, output_file, seed=42):
    """
    Prepare annotations for real data cycle training.

    Args:
        rec_file: Path to recognition label file, format:
                  image_name text
                  (one per line, e.g., "img_001.jpg HELLO")
        image_dir: Directory containing real scene text images.
        output_file: Path to output pickle file.

    The script generates perturbed target texts for cycle training.
    """
    random.seed(seed)

    annotations = {
        "image1_paths": [],
        "image1_rec": [],
        "image2_rec": [],
    }

    with open(rec_file, "r") as f:
        lines = f.read().strip().split("\n")

    prev_word = "Family"
    for line in tqdm(lines, desc="Processing real data"):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_name = parts[0]
        text = " ".join(parts[1:])

        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            # Try without extension variants
            found = False
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                alt_path = img_path.rsplit('.', 1)[0] + ext if '.' in img_path else img_path + ext
                if os.path.exists(alt_path):
                    img_path = alt_path
                    found = True
                    break
            if not found:
                print(f"Warning: Image not found: {img_path}, skipping")
                continue

        annotations["image1_paths"].append(img_path)
        annotations["image1_rec"].append(text)

        # Generate perturbed target text for editing
        tgt_text = _perturb_text(text, prev_word)
        annotations["image2_rec"].append(tgt_text)
        prev_word = text

    total = len(annotations["image1_paths"])
    print(f"Total real samples: {total}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(annotations, f)
    print(f"Real train annotations saved to {output_file} ({total} samples)")


def _perturb_text(source_text, prev_word):
    """Generate a perturbed target text for cycle training."""
    if len(prev_word) < 3:
        prev_word = 'the'

    # Generate target text of similar length but different content
    target = prev_word
    while len(target) < len(source_text) + 2:
        target += prev_word

    if len(source_text) <= 2:
        return target[:len(source_text)]
    else:
        bias = random.choice([-2, -1, 0, 1, 2])
        return target[:len(source_text) + bias]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training annotations for RS-STE")
    parser.add_argument("--mode", type=str, required=True, choices=["synth", "real"],
                        help="synth: paired synthetic data | real: unpaired real data for cycle training")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset directory (for synth mode)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to real images directory (for real mode)")
    parser.add_argument("--rec_file", type=str, default=None,
                        help="Path to recognition label file (for real mode, one 'image_name text' per line)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output pickle file path")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Train/test split ratio (synth mode only)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "synth":
        if args.data_dir is None:
            print("Error: --data_dir is required for synth mode")
            sys.exit(1)
        prepare_synth_pair_annotations(args.data_dir, args.output, args.train_ratio, args.seed)
    elif args.mode == "real":
        if args.image_dir is None or args.rec_file is None:
            print("Error: --image_dir and --rec_file are required for real mode")
            sys.exit(1)
        prepare_real_cycle_annotations(args.rec_file, args.image_dir, args.output, args.seed)
