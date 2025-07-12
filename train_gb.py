import argparse
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Import the get_features function from the existing scripts
from run_grid_search_binary import get_features


def train_and_save_model(dataset_path, level='task', label_type='binary', use_eeg=True, use_face=True, use_log=True, use_sam=True):
    """
    Train a final GradientBoostingClassifier model on all data and save it.

    Parameters:
    -----------
    dataset_path: str
        Path to pickled dataset
    level : str, 'task' or 'action'
        Level of prediction.
    label_type : str, 'binary' or 'multilabel'
        Type of labels to predict.
    use_eeg : bool
        Whether to use EEG features.
    use_face : bool
        Whether to use facial expression features.
    use_log : bool
        Whether to use log-based features.
    use_sam : bool
        Whether to use SAM features.
    """
    print(f"Training final model for level: {level}, label_type: {label_type}")
    print(f"Features - log: {use_log}, eeg: {use_eeg}, face: {use_face}, sam: {use_sam}")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load dataset
    df_all = pd.read_pickle(dataset_path)

    # Filter data based on level
    if level == 'task':
        df = df_all[(df_all['eid'] == 0)].copy()  # task-level
    else:
        df = df_all[(df_all['eid'] > 0)].copy()  # action-level

    print(f"Filtered data for level '{level}' with shape: {df.shape}")

    # Define features (X)
    columns_to_keep = get_features(log=use_log, eeg=use_eeg, face=use_face, sam=use_sam, norm=True, non_norm=False)
    X = df[columns_to_keep]

    # Define target (y) based on label_type
    if label_type == 'binary':
        y = df['y_binary']
        # For binary classification, use regular GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    else:  # multilabel
        if level == 'task':
            y = df['y_task_multilabel']
        else:
            y = df['y_action_multilabel']

        # Convert empty or [0] labels to empty lists for MultiLabelBinarizer
        y = y.map(lambda x: [] if len(x) == 1 and x[0] == 0 else x)

        # Create and fit the MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        y_bin = mlb.fit_transform(y)

        # For multilabel classification, use MultiOutputClassifier
        model = MultiOutputClassifier(
            GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        )

    # Convert categorical features to numeric codes
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    # Handle missing values
    X = X.fillna(0)
    X[['sam_valence', 'sam_arousal']] = X[['sam_valence', 'sam_arousal']].fillna(5)

    # Create model directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)

    # Prepare data for training
    X_train = X.to_numpy()

    # Define model name
    model_name = f"{level}_{label_type}_eeg{int(use_eeg)}_face{int(use_face)}"

    # Create a dictionary to save all necessary information
    model_data = {
        'model': None,  # Will be set after training
        'feature_names': columns_to_keep,
        'model_parameters': {
            'level': level,
            'label_type': label_type,
            'use_eeg': use_eeg,
            'use_face': use_face,
            'use_log': use_log,
            'use_sam': use_sam
        }
    }

    # If multilabel, also save the label binarizer
    if label_type == 'multilabel':
        model_data['mlb'] = mlb
        y_train = y_bin
    else:
        y_train = y.to_numpy()

    print(f"Training model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")

    # Train the model
    model.fit(X_train, y_train)

    # Store the trained model in the dictionary
    model_data['model'] = model

    # Ensure the directory exists
    os.makedirs("saved_models", exist_ok=True)

    # Save the model data
    model_path = f"saved_models/gb_{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model trained and saved to {model_path}")


def str_to_bool(value):
    return value.lower() in {"true", "1", "yes"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a GradientBoostingClassifier.")
    
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset file')
    parser.add_argument('--level', type=str, default='task',
                        help='Level of classification (e.g., task, session)')
    parser.add_argument('--label_type', type=str, default='binary',
                        help='Type of label to use (e.g., binary, multiclass)')
    parser.add_argument('--use_log', type=lambda x: str(x).lower() in ['true', '1'], default=True,
                        help='Whether to use log features (True/False)')
    parser.add_argument('--use_eeg', type=lambda x: str(x).lower() in ['true', '1'], default=True,
                        help='Whether to use EEG features (True/False)')
    parser.add_argument('--use_face', type=lambda x: str(x).lower() in ['true', '1'], default=True,
                        help='Whether to use facial features (True/False)')
    parser.add_argument('--use_sam', type=lambda x: str(x).lower() in ['true', '1'], default=True,
                        help='Whether to use SAM features (True/False)')

    args = parser.parse_args()

    train_and_save_model(
        args.dataset_path,
        level=args.level,
        label_type=args.label_type,
        use_eeg=args.use_eeg,
        use_face=args.use_face,
        use_log=args.use_log,
        use_sam=args.use_sam
    )
