# Training Guide

## Training Approaches

The framework supports two main approaches:

1. **Traditional ML**: Gradient Boosting with scikit-learn
2. **Deep Learning**: Hierarchical multimodal neural networks with PyTorch Lightning


## Gradient Boosting Models (Scikit-learn)

### Basic Training

```bash
# Task-level binary classification
python train_gb.py \
    --dataset_path dfs/df_all.pkl \
    --level task \
    --label_type binary \
    --use_eeg True \
    --use_face True \
    --use_log True \
    --use_sam True

# Action-level multilabel classification
python train_gb.py \
    --dataset_path dfs/df_all.pkl \
    --level action \
    --label_type multilabel \
    --use_eeg True \
    --use_face False
```

### Grid Search Optimization

For hyperparameter tuning:

```bash
# Binary classification grid search
python grid_search_binary_gb.py task True True True True

# Multilabel classification grid search  
python grid_search_multilabel_gb.py action True True True True
```

Parameters: `level log_features eeg_features face_features sam_features`




### Feature Ablation Studies

Run comprehensive ablation studies to understand feature importance:

```bash
# Task-level binary ablation
python ablate_gb.py task binary

# Action-level multilabel ablation  
python ablate_gb.py action multilabel
```

This will test the following feature combinations:
- All features (Log + EEG + Face + SAM)
- No SAM features
- No Face features  
- No EEG features
- Log features only
- No Log features


---


## Neural Network Models (PyTorch Lightning)

### Basic Training

```bash
# Task-level binary classification
python train_nn.py \
    --base_dir UsabilitySmellsDataset/ \
    --level task \
    --label_type binary \
    --use_eeg_bvp \
    --use_face \
    --batch_size 1 \
    --max_epochs 50 \
    --lr 0.001

# Action-level multilabel with full history
python train_nn.py \
    --base_dir UsabilitySmellsDataset/ \
    --level action \
    --label_type multilabel \
    --full_history \
    --use_eeg_bvp \
    --use_face \
    --hidden_dim 128 \
    --dropout 0.5
```

### Cross-Validation Training

User-based cross-validation for robust evaluation:

```bash
# 5-fold cross-validation
python train_nn.py \
    --level task \
    --label_type binary \
    --k_folds 5 \
    --target_network ALL \
    --use_eeg_bvp \
    --use_face

# Network-specific evaluation
python train_nn.py \
    --level task \
    --label_type binary \
    --k_folds 5 \
    --target_network SN_1
```

### Ablation Studies

Automated feature ablation during training:

```bash
# Comprehensive ablation study
python train_nn.py \
    --ablate \
    --level task \
    --label_type binary \
    --k_folds 5 \
    --target_network ALL \
    --max_epochs 30
```

Tests combinations of:
- EEG/BVP features
- Facial expression features
- SAM features


### Examples

See more examples in the `run_scripts/` folder.



## Model Saving and Checkpointing

### Gradient Boosting Models

Models are automatically saved to `saved_models/`:
```
saved_models/
├── gb_task_binary_eeg1_face1.pkl
├── gb_action_multilabel_eeg1_face0.pkl
└── ...
```

### Neural Network Models

PyTorch Lightning automatically saves best models:
```
saved_models/
├── task_binary_eeg1_face1.ckpt
├── action_multilabel_eeg1_face1.ckpt
└── ...
```

### Final Model Training

Train production models on all available data:

```bash
# Train final model without validation split
python train_nn.py \
    --train_final \
    --level task \
    --label_type binary \
    --use_eeg_bvp \
    --use_face \
    --save_dir final_models/
```

## Monitoring and Logging

### Comet ML Integration

```bash
# Train with Comet logging
python train_nn.py \
    --logger comet \
    --project_name usability_smells \
    --run_name task_binary_experiment
```
