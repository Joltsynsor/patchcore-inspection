#!/bin/bash

# Masonry PatchCore Experiments Grid Search
# Tests combinations of training set sizes and coreset ratios
# Includes both training and inference for complete pipeline

echo "=========================================="
echo "PatchCore Masonry Experiments Grid Search"
echo "=========================================="
echo "Train sizes: 100, 200, 400"
echo "Coreset ratios: 0.1, 0.2, 0.3"
echo "Total experiments: 9"
echo "Pipeline: Training + Inference"
echo ""

# Create logs directory
mkdir -p results/experiment_logs

# Define experiment parameters
TRAIN_SIZES=(100 200 400)
CORESET_RATIOS=(0.1 0.2 0.3)

# Track experiment progress
TOTAL_EXPERIMENTS=9
CURRENT_EXPERIMENT=0

# Add src directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "Starting experiments at $(date)"
echo ""

# Run grid search
for train_size in "${TRAIN_SIZES[@]}"; do
    for coreset_ratio in "${CORESET_RATIOS[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))

        # Create experiment version name
        VERSION="exp_train${train_size}_coreset${coreset_ratio}"

        echo "=========================================="
        echo "Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
        echo "Version: $VERSION"
        echo "Train size: $train_size"
        echo "Coreset ratio: $coreset_ratio"
        echo "Started at: $(date)"
        echo "=========================================="

        # Create log file for this experiment
        LOG_FILE="results/experiment_logs/${VERSION}.log"

        # ========================================
        # TRAINING PHASE
        # ========================================
        echo "🚀 TRAINING PHASE"
        echo "Training started..." | tee -a "$LOG_FILE"
        TRAIN_START_TIME=$(date +%s)

        python train_masonry.py \
            --version "$VERSION" \
            --train_size "$train_size" \
            --coreset_ratio "$coreset_ratio" \
            --batch_size 8 \
            --device cuda:0 2>&1 | tee -a "$LOG_FILE"

        # Check if training was successful
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            TRAIN_END_TIME=$(date +%s)
            TRAIN_DURATION=$((TRAIN_END_TIME - TRAIN_START_TIME))
            TRAIN_MINUTES=$((TRAIN_DURATION / 60))
            TRAIN_SECONDS=$((TRAIN_DURATION % 60))

            echo "✅ Training completed successfully!" | tee -a "$LOG_FILE"
            echo "   Training duration: ${TRAIN_MINUTES}m ${TRAIN_SECONDS}s" | tee -a "$LOG_FILE"
            echo "   Model saved: results/masonry_model_${VERSION}" | tee -a "$LOG_FILE"
            echo ""

            # ========================================
            # INFERENCE PHASE
            # ========================================
            echo "🔍 INFERENCE PHASE"
            echo "Inference started..." | tee -a "$LOG_FILE"
            INFERENCE_START_TIME=$(date +%s)

            python inference_masonry.py \
                --version "$VERSION" \
                --device cuda:0 2>&1 | tee -a "$LOG_FILE"

            # Check if inference was successful
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                INFERENCE_END_TIME=$(date +%s)
                INFERENCE_DURATION=$((INFERENCE_END_TIME - INFERENCE_START_TIME))
                INFERENCE_MINUTES=$((INFERENCE_DURATION / 60))
                INFERENCE_SECONDS=$((INFERENCE_DURATION % 60))

                TOTAL_DURATION=$((TRAIN_DURATION + INFERENCE_DURATION))
                TOTAL_MINUTES=$((TOTAL_DURATION / 60))
                TOTAL_SECONDS=$((TOTAL_DURATION % 60))

                echo "✅ Inference completed successfully!" | tee -a "$LOG_FILE"
                echo "   Inference duration: ${INFERENCE_MINUTES}m ${INFERENCE_SECONDS}s" | tee -a "$LOG_FILE"
                echo "   Total experiment time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" | tee -a "$LOG_FILE"
                echo "   Results saved in: results/anomaly_heatmaps/${VERSION}/" | tee -a "$LOG_FILE"
                echo "   Raw masks saved in: results/anomaly_masks/${VERSION}/" | tee -a "$LOG_FILE"
                echo ""
                echo "🎉 Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS COMPLETED"
                echo ""
            else
                echo "❌ Inference failed!" | tee -a "$LOG_FILE"
                echo "   Training succeeded but inference failed"
                echo "   Check log: $LOG_FILE"
                echo ""
            fi

        else
            echo "❌ Training failed!" | tee -a "$LOG_FILE"
            echo "   Check log: $LOG_FILE"
            echo ""

            # Ask if user wants to continue
            read -p "Continue with remaining experiments? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Experiments stopped by user."
                exit 1
            fi
        fi

        # Small delay between experiments
        sleep 3
    done
done

echo "=========================================="
echo "All experiments completed at $(date)"
echo "=========================================="

# Generate comprehensive summary report
SUMMARY_FILE="results/experiment_logs/experiments_summary.txt"
echo "Experiment Summary Generated: $(date)" > "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "Complete Pipeline Results (Training + Inference)" >> "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Experiment Results:" >> "$SUMMARY_FILE"
echo "-------------------" >> "$SUMMARY_FILE"
printf "%-25s %-8s %-12s %-12s %-10s\n" "Version" "Status" "Train Time" "Infer Time" "Total Time" >> "$SUMMARY_FILE"
printf "%-25s %-8s %-12s %-12s %-10s\n" "-------------------------" "--------" "------------" "------------" "----------" >> "$SUMMARY_FILE"

for train_size in "${TRAIN_SIZES[@]}"; do
    for coreset_ratio in "${CORESET_RATIOS[@]}"; do
        VERSION="exp_train${train_size}_coreset${coreset_ratio}"
        MODEL_DIR="results/masonry_model_${VERSION}"
        HEATMAP_DIR="results/anomaly_heatmaps/${VERSION}"

        if [ -d "$MODEL_DIR" ] && [ -d "$HEATMAP_DIR" ]; then
            STATUS="✅ COMPLETE"

            # Extract timing information
            LOG_FILE="results/experiment_logs/${VERSION}.log"
            TRAIN_TIME="N/A"
            INFERENCE_TIME="N/A"
            TOTAL_TIME="N/A"

            if [ -f "$LOG_FILE" ]; then
                TRAIN_LINE=$(grep "Training duration:" "$LOG_FILE" | tail -1)
                INFERENCE_LINE=$(grep "Inference duration:" "$LOG_FILE" | tail -1)
                TOTAL_LINE=$(grep "Total experiment time:" "$LOG_FILE" | tail -1)

                if [ ! -z "$TRAIN_LINE" ]; then
                    TRAIN_TIME=$(echo "$TRAIN_LINE" | awk '{print $3}')
                fi
                if [ ! -z "$INFERENCE_LINE" ]; then
                    INFERENCE_TIME=$(echo "$INFERENCE_LINE" | awk '{print $3}')
                fi
                if [ ! -z "$TOTAL_LINE" ]; then
                    TOTAL_TIME=$(echo "$TOTAL_LINE" | awk '{print $4}')
                fi
            fi

        elif [ -d "$MODEL_DIR" ]; then
            STATUS="⚠️  PARTIAL"
            TRAIN_TIME="Done"
            INFERENCE_TIME="Failed"
            TOTAL_TIME="N/A"
        else
            STATUS="❌ FAILED"
            TRAIN_TIME="Failed"
            INFERENCE_TIME="N/A"
            TOTAL_TIME="N/A"
        fi

        printf "%-25s %-8s %-12s %-12s %-10s\n" "$VERSION" "$STATUS" "$TRAIN_TIME" "$INFERENCE_TIME" "$TOTAL_TIME" >> "$SUMMARY_FILE"
    done
done

echo "" >> "$SUMMARY_FILE"
echo "Results Directory Structure:" >> "$SUMMARY_FILE"
echo "----------------------------" >> "$SUMMARY_FILE"
echo "results/" >> "$SUMMARY_FILE"
echo "├── masonry_model_*/ (trained models)" >> "$SUMMARY_FILE"
echo "├── anomaly_heatmaps/*/ (visualization outputs)" >> "$SUMMARY_FILE"
echo "├── anomaly_masks/*/ (raw numerical masks)" >> "$SUMMARY_FILE"
echo "└── experiment_logs/ (training and inference logs)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Analysis Commands:" >> "$SUMMARY_FILE"
echo "------------------" >> "$SUMMARY_FILE"
echo "# View all heatmaps for a specific experiment:" >> "$SUMMARY_FILE"
echo "ls results/anomaly_heatmaps/exp_train100_coreset0.1/" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "# Compare experiment results:" >> "$SUMMARY_FILE"
echo "ls results/anomaly_heatmaps/" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "# Check individual experiment logs:" >> "$SUMMARY_FILE"
echo "cat results/experiment_logs/exp_train100_coreset0.1.log" >> "$SUMMARY_FILE"

# Display summary
echo ""
echo "📊 COMPREHENSIVE EXPERIMENT SUMMARY"
echo "===================================="
cat "$SUMMARY_FILE"

echo ""
echo "📁 Complete Results Structure:"
echo "   Models: results/masonry_model_*/"
echo "   Heatmaps: results/anomaly_heatmaps/*/"
echo "   Raw Masks: results/anomaly_masks/*/"
echo "   Logs: results/experiment_logs/"
echo "   Summary: $SUMMARY_FILE"

echo ""
echo "✨ All experiments complete! Ready for analysis."