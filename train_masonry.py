import sys
sys.path.append('src')

import torch
import argparse
import patchcore.datasets.masonry as masonry
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PatchCore model for masonry anomaly detection')
    parser.add_argument('--version', type=str, default='v1.0',
                       help='Model version tag (default: v1.0)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training (default: cuda:0)')
    parser.add_argument('--train_size', type=int, default=None,
                       help='Number of training images to use (default: use all available)')
    parser.add_argument('--coreset_ratio', type=float, default=0.1,
                       help='Coreset sampling ratio (default: 0.1 = 10%%)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model version: {args.version}")
    print(f"Training set size: {'All available' if args.train_size is None else args.train_size}")
    print(f"Coreset ratio: {args.coreset_ratio} ({args.coreset_ratio*100:.1f}%)")
    print(f"Batch size: {args.batch_size}")

    # Load dataset (training only)
    train_dataset = masonry.MasonryDataset(
        source="masonry_dataset",
        classname="wall",
        resize=256,
        imagesize=224,
        split=masonry.DatasetSplit.TRAIN
    )

    # Limit training set size if specified
    if args.train_size is not None and args.train_size < len(train_dataset):
        print(f"Limiting training set from {len(train_dataset)} to {args.train_size} images")
        # Create a subset of the dataset
        indices = list(range(args.train_size))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"Training dataset loaded: {len(train_dataset)} images")

    # Create PatchCore model
    backbone = patchcore.backbones.load("wideresnet50")
    backbone.name = "wideresnet50"

    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(args.coreset_ratio, device)
    nn_method = patchcore.common.FaissNN(False, 4)  # CPU FAISS

    patchcore_model = patchcore.patchcore.PatchCore(device)
    patchcore_model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, 224, 224),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        featuresampler=sampler,
        anomaly_scorer_num_nn=1,
        nn_method=nn_method,
    )

    print("PatchCore model created")
    print(f"Coreset sampler configured with {args.coreset_ratio*100:.1f}% sampling ratio")

    # Train the model
    print("Starting training...")
    patchcore_model.fit(train_dataloader)
    print("Training completed!")

    # Save the model with comprehensive version tag
    import os

    # Check if version already contains experiment parameters to avoid redundancy
    if "train" in args.version and "coreset" in args.version:
        # Version already contains experiment info (e.g., from run_experiments.sh)
        version_suffix = args.version
    else:
        # Build version suffix from individual parameters
        version_suffix = f"{args.version}"
        if args.train_size is not None:
            version_suffix += f"_train{args.train_size}"
        version_suffix += f"_coreset{args.coreset_ratio}"

    save_path = f"results/masonry_model_{version_suffix}"
    os.makedirs(save_path, exist_ok=True)
    patchcore_model.save_to_path(save_path)

    # Save training configuration
    config_path = os.path.join(save_path, "training_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Model Version: {args.version}\n")
        f.write(f"Training Set Size: {'All available' if args.train_size is None else args.train_size}\n")
        f.write(f"Coreset Ratio: {args.coreset_ratio} ({args.coreset_ratio*100:.1f}%)\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Backbone: wideresnet50\n")
        f.write(f"Layers: layer2, layer3\n")
        f.write(f"Input Size: 224x224\n")
        f.write(f"Patch Size: 3\n")

    print(f"Model saved to: {save_path}")
    print(f"Training configuration saved to: {config_path}")
    print(f"Use for inference: python inference_masonry.py --version {version_suffix}")

    print("Training and saving completed successfully!")

if __name__ == "__main__":
    main()
