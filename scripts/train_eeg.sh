#!/bin/bash
### Train UBP EEG encoder on local workstation serially across subjects.
###
### Usage:
###   ./scripts/train_eeg.sh 1,2,3,4,5,7,9
###   ./scripts/train_eeg.sh 1 2 3
###   EXP_SETTING=inter-subject ./scripts/train_eeg.sh 1,2
###   VISION_BACKBONE=ViT-H-14 ./scripts/train_eeg.sh 1

set -euo pipefail

# 1. Path Setup
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

LOG_DIR="${REPO}/logs"
mkdir -p "$LOG_DIR"

# 2. Environment Activation
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ubp


# Target GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export UBP_GPU=0



CONFIG=configs/eeg/ubp.yaml

# brain_backbones=("EEGProjectLayer" "Shallownet" "Deepnet" "EEGnet" "TSconv")
# vision_backbones=("RN50" "RN101" "ViT-B-16" "ViT-B-32" "ViT-L-14" "ViT-H-14" "ViT-g-14" "ViT-bigG-14")
EXP_SETTING=intra-subject
BRAIN_BACKBONE=EEGProjectLayer
VISION_BACKBONE=RN50


EPOCH=${EPOCH:-70}
SEED=${SEED:-0}
# intra-subject trains on 1e-4; inter-subject (leave-one-out) on 1e-5.
if [ "$EXP_SETTING" = "inter-subject" ]; then
    LR=${LR:-1e-5}
else
    LR=${LR:-1e-4}
fi


# 5. Parse Subjects Argument (handles "1,2,3", "sub-01,sub-02", or space-separated arguments)
RAW_INPUT="${*:-1}"
# Replace commas with spaces to normalize
CLEAN_INPUT="${RAW_INPUT//,/ }"

SUBJECTS=()
for item in $CLEAN_INPUT; do
    # Strip any existing "sub-" prefix, then format to 2 digits (e.g. 1 -> sub-01)
    num=$(echo "$item" | sed 's/[^0-9]//g')
    if [ -n "$num" ]; then
        SUBJECTS+=("sub-$(printf "%02d" "$num")")
    fi
done

echo "============================================================"
echo "Host             : $(hostname)"
echo "GPU              : $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)"
echo "Config           : $CONFIG"
echo "Setting          : $EXP_SETTING | Backbone: $BRAIN_BACKBONE / $VISION_BACKBONE"
echo "Subjects to run  : ${SUBJECTS[*]}"
echo "Logging to       : $LOG_DIR"
echo "============================================================"

# 6. Serial Execution Loop
for SUB in "${SUBJECTS[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUT_LOG="${LOG_DIR}/train_${SUB}_${TIMESTAMP}.out"
    ERR_LOG="${LOG_DIR}/train_${SUB}_${TIMESTAMP}.err"

    echo ""
    echo ">>> [$(date -Is)] Starting ${SUB}..."
    echo "    stdout -> ${OUT_LOG}"
    echo "    stderr -> ${ERR_LOG}"

    # Runs python command: captures stdout to .out and stderr to .err while streaming stdout to console
    python main.py \
        --config "$CONFIG" \
        --dataset eeg \
        --subjects "$SUB" \
        --seed "$SEED" \
        --exp_setting "$EXP_SETTING" \
        --brain_backbone "$BRAIN_BACKBONE" \
        --vision_backbone "$VISION_BACKBONE" \
        --epoch "$EPOCH" \
        --lr "$LR" \
        > >(tee "$OUT_LOG") \
        2> >(tee "$ERR_LOG" >&2)

    echo ">>> [$(date -Is)] Completed ${SUB} successfully."
done

echo ""
echo "============================================================"
echo "All runs completed at $(date -Is)"
echo "============================================================"