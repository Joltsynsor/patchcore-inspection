#!/bin/bash

# Run inference on all trained experiment models
# Processes all models in results/masonry_model_exp_* directories

echo "🔍 Running Inference on All Experiment Models"
echo "=============================================="

# Add src directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create logs directory for inference
mkdir -p results/experiment_logs/inference

# Find all experiment models
MODEL_DIRS=($(ls -d results/masonry_model_exp_* 2>/dev/null))

if [ ${#MODEL_DIRS[@]} -eq 0 ]; then
    echo "❌ No experiment models found!"
    echo "Expected models in: results/masonry_model_exp_*"
    echo "Run training first: ./run_experiments.sh"
    exit 1
fi

echo "Found ${#MODEL_DIRS[@]} trained models:"
for model_dir in "${MODEL_DIRS[@]}"; do
    version=$(basename "$model_dir" | sed 's/masonry_model_//')
    echo "  - $version"
done
echo ""

TOTAL_MODELS=${#MODEL_DIRS[@]}
CURRENT_MODEL=0
SUCCESSFUL_INFERENCE=0
FAILED_INFERENCE=0

echo "Starting inference at $(date)"
echo ""

# Run inference on each model
for model_dir in "${MODEL_DIRS[@]}"; do
    CURRENT_MODEL=$((CURRENT_MODEL + 1))

    # Extract version from directory name
    VERSION=$(basename "$model_dir" | sed 's/masonry_model_//')

    echo "=========================================="
    echo "Inference $CURRENT_MODEL/$TOTAL_MODELS"
    echo "Model: $VERSION"
    echo "Started at: $(date)"
    echo "=========================================="

    # Create inference log file
    INFERENCE_LOG="results/experiment_logs/inference/${VERSION}_inference.log"

    echo "🔍 Running inference..." | tee -a "$INFERENCE_LOG"
    INFERENCE_START_TIME=$(date +%s)

    # Run inference
    python inference_masonry.py \
        --version "$VERSION" \
        --device cuda:0 2>&1 | tee -a "$INFERENCE_LOG"

    # Check if inference was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        INFERENCE_END_TIME=$(date +%s)
        INFERENCE_DURATION=$((INFERENCE_END_TIME - INFERENCE_START_TIME))
        INFERENCE_MINUTES=$((INFERENCE_DURATION / 60))
        INFERENCE_SECONDS=$((INFERENCE_DURATION % 60))

        SUCCESSFUL_INFERENCE=$((SUCCESSFUL_INFERENCE + 1))

        echo "✅ Inference completed successfully!" | tee -a "$INFERENCE_LOG"
        echo "   Duration: ${INFERENCE_MINUTES}m ${INFERENCE_SECONDS}s" | tee -a "$INFERENCE_LOG"
        echo "   Heatmaps: results/anomaly_heatmaps/$VERSION/" | tee -a "$INFERENCE_LOG"
        echo "   Raw masks: results/anomaly_masks/$VERSION/" | tee -a "$INFERENCE_LOG"
        echo ""
    else
        FAILED_INFERENCE=$((FAILED_INFERENCE + 1))

        echo "❌ Inference failed!" | tee -a "$INFERENCE_LOG"
        echo "   Check log: $INFERENCE_LOG"
        echo ""

        # Ask if user wants to continue
        read -p "Continue with remaining models? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Inference stopped by user."
            break
        fi
    fi

    # Small delay between inferences
    sleep 2
done

echo "=========================================="
echo "All inference completed at $(date)"
echo "=========================================="

# Generate inference summary
INFERENCE_SUMMARY="results/experiment_logs/inference_summary.txt"
echo "Inference Summary Generated: $(date)" > "$INFERENCE_SUMMARY"
echo "==========================================" >> "$INFERENCE_SUMMARY"
echo "" >> "$INFERENCE_SUMMARY"

echo "Inference Results:" >> "$INFERENCE_SUMMARY"
echo "------------------" >> "$INFERENCE_SUMMARY"
echo "Total models processed: $TOTAL_MODELS" >> "$INFERENCE_SUMMARY"
echo "Successful inferences: $SUCCESSFUL_INFERENCE" >> "$INFERENCE_SUMMARY"
echo "Failed inferences: $FAILED_INFERENCE" >> "$INFERENCE_SUMMARY"
echo "" >> "$INFERENCE_SUMMARY"

echo "Model Status:" >> "$INFERENCE_SUMMARY"
echo "-------------" >> "$INFERENCE_SUMMARY"
printf "%-30s %-10s %-15s\n" "Model Version" "Status" "Duration" >> "$INFERENCE_SUMMARY"
printf "%-30s %-10s %-15s\n" "------------------------------" "----------" "---------------" >> "$INFERENCE_SUMMARY"

for model_dir in "${MODEL_DIRS[@]}"; do
    VERSION=$(basename "$model_dir" | sed 's/masonry_model_//')
    HEATMAP_DIR="results/anomaly_heatmaps/$VERSION"
    INFERENCE_LOG="results/experiment_logs/inference/${VERSION}_inference.log"

    if [ -d "$HEATMAP_DIR" ]; then
        STATUS="✅ SUCCESS"

        # Extract duration if available
        DURATION="N/A"
        if [ -f "$INFERENCE_LOG" ]; then
            DURATION_LINE=$(grep "Duration:" "$INFERENCE_LOG" | tail -1)
            if [ ! -z "$DURATION_LINE" ]; then
                DURATION=$(echo "$DURATION_LINE" | awk '{print $2}')
            fi
        fi
    else
        STATUS="❌ FAILED"
        DURATION="N/A"
    fi

    printf "%-30s %-10s %-15s\n" "$VERSION" "$STATUS" "$DURATION" >> "$INFERENCE_SUMMARY"
done

echo "" >> "$INFERENCE_SUMMARY"
echo "Output Locations:" >> "$INFERENCE_SUMMARY"
echo "-----------------" >> "$INFERENCE_SUMMARY"
echo "Heatmaps: results/anomaly_heatmaps/" >> "$INFERENCE_SUMMARY"
echo "Raw masks: results/anomaly_masks/" >> "$INFERENCE_SUMMARY"
echo "Inference logs: results/experiment_logs/inference/" >> "$INFERENCE_SUMMARY"
echo "" >> "$INFERENCE_SUMMARY"

echo "Analysis Commands:" >> "$INFERENCE_SUMMARY"
echo "------------------" >> "$INFERENCE_SUMMARY"
echo "# View all experiment results:" >> "$INFERENCE_SUMMARY"
echo "ls results/anomaly_heatmaps/" >> "$INFERENCE_SUMMARY"
echo "" >> "$INFERENCE_SUMMARY"
echo "# Compare specific experiment heatmaps:" >> "$INFERENCE_SUMMARY"
echo "ls results/anomaly_heatmaps/exp_train100_coreset0.1/" >> "$INFERENCE_SUMMARY"
echo "ls results/anomaly_heatmaps/exp_train400_coreset0.3/" >> "$INFERENCE_SUMMARY"

# Display summary
echo ""
echo "📊 INFERENCE SUMMARY"
echo "===================="
cat "$INFERENCE_SUMMARY"

echo ""
echo "📁 Results Structure:"
echo "   Heatmaps: results/anomaly_heatmaps/"
echo "   Raw Masks: results/anomaly_masks/"
echo "   Logs: results/experiment_logs/inference/"
echo "   Summary: $INFERENCE_SUMMARY"

echo ""
echo "🎉 Inference complete! Ready for analysis and comparison."