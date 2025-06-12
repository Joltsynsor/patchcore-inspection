# Masonry Wall Anomaly Detection Workflow

This document describes the complete workflow for training and running inference with the PatchCore-based masonry wall anomaly detection system.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Required packages: `pip install -r requirements.txt`

## Dataset Structure

```
masonry_dataset/
└── wall/
    ├── train/
    │   └── good/          # Normal masonry images (148 images)
    └── test/
        ├── good/          # Normal test images (5 images)
        └── defect/        # Defective test images (2 images)
```

## Training Workflow

### 1. Train the Model

```bash
# Run the complete training pipeline
bash train_masonry.sh
```

This script:
- Sets up the Python environment
- Loads the masonry dataset (148 training images)
- Creates a PatchCore model with WideResNet50 backbone
- Extracts features from layers 2 & 3
- Performs coreset subsampling (10% of patches)
- Saves the trained model to `results/masonry_model/`

**Training Configuration:**
- Input size: 224x224 (RGB converted from grayscale)
- Backbone: WideResNet50 (ImageNet pretrained)
- Feature layers: layer2 (512D) + layer3 (1024D)
- Final embedding: 1024D
- Patch size: 3x3, stride=1
- Coreset sampling: 10% (~1,160 features from ~11,600 total)
- FAISS: CPU implementation (avoids GPU CUBLAS issues)

### 2. Training Output

- Model files saved to: `results/masonry_model/`
- Training time: ~30 seconds on NVIDIA L4 GPU
- Memory usage: ~2GB GPU memory

## Inference Workflow

### 1. Run Anomaly Detection

```bash
# Generate anomaly heatmaps for all test images
python inference_masonry.py
```

This script:
- Loads the trained model from `results/masonry_model/`
- Processes all test images (both good and defect)
- Generates three-panel visualizations for each image:
  - Original image
  - Anomaly heatmap (0-6 scale)
  - Overlay visualization
- Saves results to `anomaly_heatmaps/` directory

### 2. Inference Output

**Anomaly Scores:**
- Normal images: 2.2-2.5 (low anomaly scores)
- Defective images: 4.3-5.9 (high anomaly scores)
- Clear separation indicates successful anomaly detection

**Visualization Files:**
- `anomaly_heatmaps/good_*.png_heatmap.png` - Normal image results
- `anomaly_heatmaps/defect_*.png_heatmap.png` - Defective image results

## File Structure

```
├── train_masonry.sh          # Training shell script
├── train_masonry.py          # Training Python implementation
├── inference_masonry.py      # Inference and visualization script
├── masonry_dataset/          # Dataset directory
├── results/masonry_model/    # Trained model files
└── anomaly_heatmaps/         # Inference output visualizations
```

## Key Features

1. **Pipeline Consistency**: Inference reuses exact same transforms as training
2. **GPU Memory Efficient**: Uses CPU FAISS to avoid GPU memory issues
3. **Robust Preprocessing**: Handles grayscale→RGB conversion for pretrained models
4. **Clear Visualization**: Consistent 0-6 scale across all heatmaps
5. **Production Ready**: No experimental code, clean error handling

## Troubleshooting

- **CUDA out of memory**: Reduce batch size in `train_masonry.py`
- **FAISS GPU errors**: Script uses CPU FAISS by default (recommended)
- **Import errors**: Ensure `PYTHONPATH` includes `src/` directory
- **Dataset not found**: Verify `masonry_dataset/` structure matches expected format

## Performance Metrics

- **Training**: 148 images → ~1,160 representative features
- **Inference**: Clear anomaly score separation (2x difference)
- **Speed**: ~30s training, ~1s per test image inference
- **Memory**: 45MB model size, 2GB GPU training memory