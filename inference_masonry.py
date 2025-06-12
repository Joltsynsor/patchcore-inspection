#!/usr/bin/env python3

import sys
sys.path.append('src')

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import patchcore.datasets.masonry as masonry
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
from torchvision import transforms

def load_model(model_path, device):
    """Load the trained PatchCore model"""
    print(f"Loading model from {model_path}")

    # Create model architecture
    backbone = patchcore.backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    nn_method = patchcore.common.FaissNN(False, 4)  # CPU FAISS

    patchcore_model = patchcore.patchcore.PatchCore(device)
    patchcore_model.load_from_path(
        load_path=model_path,
        device=device,
        nn_method=nn_method
    )

    print("Model loaded successfully!")
    return patchcore_model

def generate_heatmap(model, image_path, device, save_path):
    """Generate anomaly heatmap for a single image"""
    # Create a minimal dataset instance just to access the transforms
    # We'll override the problematic methods to avoid file system checks
    class MinimalMasonryDataset(masonry.MasonryDataset):
        def __init__(self):
            # Skip the parent __init__ to avoid file system operations
            self.transform_img = None
            self.transform_mask = None
            self._setup_transforms()

        def _setup_transforms(self):
            # Use the exact same transform setup as the real masonry dataset
            from torchvision import transforms
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]

            resize = 256
            imagesize = 224

            self.transform_img = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    # Create the minimal dataset to get transforms
    temp_dataset = MinimalMasonryDataset()

    # Load original image (keep original size for display)
    original_image = Image.open(image_path)
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')

    # Also load and transform for model input
    model_image = Image.open(image_path)
    if model_image.mode != 'RGB':
        model_image = model_image.convert('RGB')

    # Transform image for model using the dataset's transform
    image_tensor = temp_dataset.transform_img(model_image).unsqueeze(0)  # Add batch dimension

    # Generate prediction
    with torch.no_grad():
        scores, masks = model._predict(image_tensor)

    # Get the anomaly score and mask
    anomaly_score = scores[0]
    anomaly_mask = masks[0]

    # Resize anomaly mask to match original image size
    original_size = original_image.size  # (width, height)
    anomaly_mask_resized = np.array(Image.fromarray(anomaly_mask).resize(original_size, Image.BILINEAR))

    # Create visualization with consistent 0-6 scale
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # Anomaly heatmap (with fixed 0-6 scale)
    im1 = axes[1].imshow(anomaly_mask_resized, cmap='jet', alpha=0.8, vmin=0, vmax=6)
    axes[1].set_title(f'Anomaly Heatmap (Score: {anomaly_score:.3f})', fontsize=14)
    axes[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Anomaly Score', rotation=270, labelpad=15)

    # Overlay - with consistent scale
    axes[2].imshow(original_image)
    im2 = axes[2].imshow(anomaly_mask_resized, cmap='jet', alpha=0.8, vmin=0, vmax=6)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved: {save_path} (Score: {anomaly_score:.3f})")
    return anomaly_score, anomaly_mask

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "results/masonry_model"

    # Load model
    model = load_model(model_path, device)

    # Create output directory
    output_dir = "anomaly_heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    # Process test images
    test_dirs = [
        "masonry_dataset/wall/test/good",
        "masonry_dataset/wall/test/defect"
    ]

    all_scores = []

    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            continue

        category = os.path.basename(test_dir)
        print(f"\nProcessing {category} images...")

        for filename in os.listdir(test_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_path = os.path.join(test_dir, filename)
                save_path = os.path.join(output_dir, f"{category}_{filename}_heatmap.png")

                try:
                    score, mask = generate_heatmap(model, image_path, device, save_path)
                    all_scores.append((category, filename, score))
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    # Print summary
    print(f"\n{'='*50}")
    print("ANOMALY DETECTION SUMMARY")
    print(f"{'='*50}")

    for category, filename, score in sorted(all_scores, key=lambda x: x[2], reverse=True):
        print(f"{category:10} | {filename:20} | Score: {score:.3f}")

    print(f"\nHeatmaps saved in: {output_dir}/")
    print("Higher scores indicate more anomalous regions")

if __name__ == "__main__":
    main()