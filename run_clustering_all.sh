#!/bin/bash
# Run hierarchical clustering for all datasets and all gradient methods sequentially.
# Usage: bash run_clustering_all.sh

DATASETS=("GENEEG" "MCIvsHC" "ADvsFTDvsHC" "CAUEEG")
GRADIENTS=("vanilla" "input_x_gradient" "smoothgrad" "smoothgrad_sq" "vargrad" "integrated_gradients")

TOTAL=$(( ${#DATASETS[@]} * ${#GRADIENTS[@]} ))
RUN=0

for dataset in "${DATASETS[@]}"; do
    for gradient in "${GRADIENTS[@]}"; do
        RUN=$(( RUN + 1 ))
        echo "=========================================="
        echo "Run ${RUN}/${TOTAL}: dataset=${dataset}  gradient=${gradient}"
        echo "=========================================="
        python clustering_script.py --dataset "$dataset" --gradient "$gradient"
        STATUS=$?
        if [ $STATUS -ne 0 ]; then
            echo "ERROR: run failed (exit code ${STATUS}). Continuing to next combination."
        fi
    done
done

echo "All ${TOTAL} runs completed."
