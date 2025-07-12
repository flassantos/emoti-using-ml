#!/usr/bin/env bash

# Fixed settings
BASE_DIR="./UsabilitySmellsDataset/"
LEVEL="task"
MAX_EPOCHS=80
BATCH_SIZE=1
PATIENCE=10
TRAIN_RATIO=0.8
TEST_RATIO=0.2
K_FOLDS=5
FULL_HISTORY="--full_history"
USE_SPECTROGRAMS=""
FORCE_ALIGN=""
INNER_CAT=""

# Proposed single values for the other hyperparameters
LR=0.0005
WEIGHT_DECAY=0.001
BETA1=0.9
BETA2=0.999
OPTIMIZER="adamw"
HIDDEN_DIM=64
MINOR_CLASS_WEIGHT=1.0
DROPOUT=0.5
KERNEL_SIZE=5
GRAD_CLIP_VAL=1.0
ACCUMULATE_GRAD_BATCHES=1
LABEL_TYPE="binary"

# Values to search
TARGET_NETWORK=("SN_1" "SN_2" "SN_3" "ALL")
USE_EEG_BVPS=("--use_eeg_bvp" "")
USE_FACES=("--use_face" "")

for target_network in "${TARGET_NETWORK[@]}"; do
  for use_eeg_bvp in "${USE_EEG_BVPS[@]}"; do
    for use_face in "${USE_FACES[@]}"; do

      CMD="python3 train_nn.py \
        --base_dir $BASE_DIR \
        --level $LEVEL \
        --label_type $LABEL_TYPE \
        --batch_size $BATCH_SIZE \
        --max_epochs $MAX_EPOCHS \
        --patience $PATIENCE \
        --gradient_clip_val $GRAD_CLIP_VAL \
        --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --dropout $DROPOUT \
        --beta1 $BETA1 \
        --beta2 $BETA2 \
        --optimizer $OPTIMIZER \
        --minor_class_weight $MINOR_CLASS_WEIGHT \
        --hidden_dim $HIDDEN_DIM \
        --kernel_size $KERNEL_SIZE \
        --train_ratio $TRAIN_RATIO \
        --test_ratio $TEST_RATIO \
        --target_network $target_network \
        --k_folds $K_FOLDS \
        $FULL_HISTORY \
        $USE_SPECTROGRAMS \
        $use_eeg_bvp \
        $use_face \
        $INNER_CAT \
        $FORCE_ALIGN"

      echo "$CMD"
      $CMD
    done
  done
done

