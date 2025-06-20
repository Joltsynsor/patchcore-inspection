#!/usr/bin/env python3
"""
Script to organize dataset for masonry inspection.
- Takes 10 normal images and puts them in masonry_dataset/wall/test/good
- Takes remaining normal images and puts them in masonry_dataset/wall/train/good
- Takes all anomaly images and puts them in masonry_dataset/wall/test/defect
"""

import os
import shutil
import glob
import random
from pathlib import Path

def create_directories():
    """Create the target directory structure."""
    dirs = [
        "masonry_dataset/wall/test/good",
        "masonry_dataset/wall/train/good",
        "masonry_dataset/wall/test/defect"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def organize_normal_images():
    """Organize normal images into test/good (10 images) and train/good (rest)."""
    normal_images = glob.glob("dataset/normal/*.png")
    print(f"Found {len(normal_images)} normal images")

    if len(normal_images) == 0:
        print("No normal images found!")
        return

    # Shuffle to get random selection
    random.shuffle(normal_images)

    # Split: first 10 for test, rest for train
    test_images = normal_images[:10]
    train_images = normal_images[10:]

    print(f"Moving {len(test_images)} images to test/good")
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        dest_path = f"masonry_dataset/wall/test/good/{img_name}"
        shutil.copy2(img_path, dest_path)
        print(f"  Copied: {img_name}")

    print(f"Moving {len(train_images)} images to train/good")
    for img_path in train_images:
        img_name = os.path.basename(img_path)
        dest_path = f"masonry_dataset/wall/train/good/{img_name}"
        shutil.copy2(img_path, dest_path)
        print(f"  Copied: {img_name}")

def organize_anomaly_images():
    """Move all anomaly images to test/defect."""
    anomaly_images = glob.glob("dataset/anomaly/*.png")
    print(f"Found {len(anomaly_images)} anomaly images")

    if len(anomaly_images) == 0:
        print("No anomaly images found!")
        return

    print(f"Moving {len(anomaly_images)} images to test/defect")
    for img_path in anomaly_images:
        img_name = os.path.basename(img_path)
        dest_path = f"masonry_dataset/wall/test/defect/{img_name}"
        shutil.copy2(img_path, dest_path)
        print(f"  Copied: {img_name}")

def main():
    """Main function to organize the dataset."""
    print("Starting dataset organization...")

    # Set random seed for reproducible results
    random.seed(42)

    # Create target directories
    create_directories()

    # Organize images
    organize_normal_images()
    organize_anomaly_images()

    print("\nDataset organization complete!")

    # Print summary
    test_good_count = len(glob.glob("masonry_dataset/wall/test/good/*.png"))
    train_good_count = len(glob.glob("masonry_dataset/wall/train/good/*.png"))
    test_defect_count = len(glob.glob("masonry_dataset/wall/test/defect/*.png"))

    print(f"\nSummary:")
    print(f"  Test good images: {test_good_count}")
    print(f"  Train good images: {train_good_count}")
    print(f"  Test defect images: {test_defect_count}")

if __name__ == "__main__":
    main()