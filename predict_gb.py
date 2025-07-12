import argparse
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np


def load_model(model_path):
    """
    Load a saved model from a pickle file.

    Parameters:
    -----------
    model_path : str
        Path to the saved model.

    Returns:
    --------
    dict
        Dictionary containing the model and related information.
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"Successfully loaded model from {model_path}")
        return model_data
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def prepare_input_data(df, feature_names):
    """
    Prepare input data for prediction.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    feature_names : list
        List of feature names to use.

    Returns:
    --------
    np.ndarray
        Processed features ready for prediction.
    """
    # in case a feature_name is not present in the dataframe, fill it with NaN
    for feature in feature_names:
        if feature not in df.columns:
            print("Warning: Feature '{}' not found in the input data. Filling with NaN.".format(feature))
            df[feature] = np.nan

    # Extract features
    X = df[feature_names].copy()

    # Convert categorical features to numeric codes
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    # Fill missing values
    X = X.fillna(0)

    return X.to_numpy()


def generate_and_save_predictions(model_path, input_data_path, output_path=None):
    """
    Generate predictions using a saved model and save them to a JSON file.

    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    input_data_path : str
        Path to the input data file.
    output_path : str, optional
        Path to save the predictions. If None, a default path will be used.
    """
    # Load the model
    model_data = load_model(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    model_params = model_data['model_parameters']

    # Read pickled dataset
    df = pd.read_pickle(input_data_path)

    # Filter data based on level if needed
    level = model_params.get('level', 'task')
    if 'eid' in df.columns:
        if level == 'task':
            df = df[(df['eid'] == 0)].copy()  # task-level
        else:
            df = df[(df['eid'] > 0)].copy()  # action-level
        print(f"Filtered data for level '{level}' with shape: {df.shape}")

    # Prepare input features
    X = prepare_input_data(df, feature_names)

    # Generate predictions
    print(f"Generating predictions for {X.shape[0]} samples...")

    # Make predictions
    label_type = model_params.get('label_type', 'binary')

    if label_type == 'binary':
        # For binary classification
        y_pred_prob = model.predict_proba(X)
        y_pred = model.predict(X)

        # Create predictions dictionary
        predictions = {
            'binary_predictions': y_pred.tolist(),
            'probability_predictions': [prob[1] for prob in y_pred_prob]  # Probability of class 1
        }
    else:  # multilabel
        # For multilabel classification
        y_pred = model.predict(X)

        # If available, get class labels from the MultiLabelBinarizer
        if 'mlb' in model_data:
            mlb = model_data['mlb']
            y_pred_labels = [
                mlb.classes_[pred].tolist() if any(pred) else []
                for pred in y_pred
            ]

            # Create predictions dictionary
            predictions = {
                'multilabel_predictions': y_pred.tolist(),
                'label_predictions': y_pred_labels
            }
        else:
            # If MultiLabelBinarizer is not available
            predictions = {
                'multilabel_predictions': y_pred.tolist()
            }

    # Add metadata
    predictions['metadata'] = {
        'model_parameters': model_params,
        'num_samples': X.shape[0],
        'num_features': X.shape[1]
    }

    # Determine output path if not provided
    if output_path is None:
        model_name = os.path.basename(model_path).replace('.pkl', '')
        input_name = os.path.basename(input_data_path).replace('.pkl', '').replace('.csv', '')
        output_path = f"predictions_{model_name}_{input_name}.json"

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save predictions to JSON
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions using a gradient boosting model.')
    parser.add_argument('model_path', type=str, help='Path to the saved model')
    parser.add_argument('input_data_path', type=str, help='Path to the input data file (.pkl)')
    parser.add_argument('--output_path', type=str, help='Path to save the predictions (optional)')

    args = parser.parse_args()

    generate_and_save_predictions(args.model_path, args.input_data_path, args.output_path)
