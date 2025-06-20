#!/bin/bash

# Masonry Wall Anomaly Detection Training Script
# Train only, no test evaluation

# Model versioning - can be overridden by setting VERSION environment variable
VERSION=${VERSION:-"v1.0"}

datapath="masonry_dataset"
datasets=('wall')

echo "Starting PatchCore training for Masonry Wall Anomaly Detection..."
echo "Dataset path: $datapath"
echo "Model version: $VERSION"
echo "Configuration: IM224 Baseline with CPU FAISS (training only)"

# Add src directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run the training script with version parameter
python train_masonry.py --version "$VERSION"

echo "Training completed!"
echo "Model saved as version: $VERSION"
echo "Use: python inference_masonry.py --version $VERSION"