#!/bin/bash

# Masonry Wall Anomaly Detection Training Script
# Train only, no test evaluation

datapath="masonry_dataset"
datasets=('wall')

echo "Starting PatchCore training for Masonry Wall Anomaly Detection..."
echo "Dataset path: $datapath"
echo "Configuration: IM224 Baseline with CPU FAISS (training only)"

# Add src directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run the training script
python train_masonry.py

echo "Training completed!"
echo "Model saved for inference!"