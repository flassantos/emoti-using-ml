# emoti-using-ml

A comprehensive machine learning framework for detecting usability smells in user interfaces using multimodal data including EEG signals, facial expressions, interaction logs, and self-assessment 


## Project Structure

```
├── dataset/                 # Data loading and preprocessing
│   ├── data_module.py      # PyTorch Lightning data module
│   ├── dataset.py          # Custom dataset classes
│   ├── eeg_features.py     # EEG feature extraction
│   ├── face_features.py    # Facial feature extraction
│   └── utils.py            # Utility functions
├── models/                 # Neural network architectures
│   ├── model.py            # Base multimodal model
│   └── model_full_history.py # Extended model with full history
├── run_scripts/            # Training and evaluation scripts
├── saved_models/           # Pre-trained model checkpoints
├── dfs/                    # Processed datasets
└── docs/                   # Documentation
```


## Quick Start

### Prerequisites

Install the requirmeents in a virtual env:

```bash
pip install -r requirements.txt
```


### Dataset Setup

1. **For scikit-learn models**: The processed AMUSED dataset `dfs/df_all.pkl` is included. Alternatively, check the [AMUSED repo](https://github.com/flassantos/amused/).


2. **For neural network models**: Download the full dataset following the instructions in the [AMUSED repo](https://github.com/flassantos/amused/). Afterwards, we will set the input path to the `UsabilitySmellsDataset/` directory.
  

More information can be found in [docs/DATA_FORMAT.md](/docs/DATA_FORMAT.md).


### Training Models

#### Gradient Boosting (scikit-learn)

```bash
# Train a binary classification model at task level
python train_gb.py --dataset_path dfs/df_all.pkl --level task --label_type binary

# Train a multilabel model at action level
python train_gb.py --dataset_path dfs/df_all.pkl --level action --label_type multilabel
```


#### Neural Networks (PyTorch Lightning)

```bash
# Train task-level binary classification
python train_nn.py --level task --label_type binary --use_eeg_bvp --use_face  --full_history

# Train with cross-validation
python train_nn.py --level task --label_type binary --k_folds 5 --target_network ALL  --full_history

# Run ablation study
python train_nn.py --ablate --level task --label_type binary --k_folds 5  --full_history
```


More information can be found in [docs/TRAINING.md](/docs/TRAINING.md). Examples are also available in [run_scripts/](/run_scripts/).



### Making Predictions

#### Gradient Boosting Models

```bash
python predict_gb.py saved_models/gb_task_binary.pkl dfs/df_all.pkl --output_path predictions/
```

#### Neural Network Models

```bash
python predict_nn.py --model_path saved_models/nn_task_binary.ckpt --base_dir UsabilitySmellsDataset/
```

### Feature Ablation

```bash
# Gradient Boosting ablation
python ablate_gb.py task binary

# Neural Network ablation (automatically included in training)
python train_nn.py --ablate --level task --label_type multilabel --k_folds 5
```


## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and macro-averaged
- **Jaccard Score**: Set similarity for multilabel tasks
- **Hamming Score**: Label-wise accuracy for multilabel
- **AUPRC**: Area under precision-recall curve
- **Balanced Accuracy**: Class-balanced accuracy measure



## Configuration Options

### Model Parameters

- `--level`: Classification granularity (`task` or `action`)
- `--label_type`: Label format (`binary` or `multilabel`)
- `--use_eeg_bvp`: Include EEG/BVP features
- `--use_face`: Include facial expression features
- `--hidden_dim`: Neural network hidden dimensions
- `--dropout`: Dropout rate for regularization
- `--full_history`: Wether to use the model with attention mechanism (**recommended**)

### Training Parameters

- `--batch_size`: Batch size for training
- `--max_epochs`: Maximum training epochs
- `--lr`: Learning rate
- `--k_folds`: Number of cross-validation folds
- `--target_network`: Specific network for evaluation


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
