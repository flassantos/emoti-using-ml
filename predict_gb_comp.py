import argparse
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import random





task_smells_mapping = {
    'none': 0,
    'laborious_task': 1,
    'cyclic_task': 2,
    'too_many_layers': 3,
    'high_interaction_distance': 4,
    'repetition_in_text_fields': 5,
    'missing_task_feedback': 6,
    'late_validation': 7,
}
rev_task_smells_mapping = {v: k for k, v in task_smells_mapping.items()}
action_smells_mapping = {
    'none': 0,
    'unnecessary_action': 1,
    'misleading_action': 2,
    'missing_action_feedback': 3,
    'undescriptive_element': 4
}
rev_action_smells_mapping = {v: k for k, v in action_smells_mapping.items()}


def seed_everything(seed: int = 42):
    """Seed Python, NumPy, and common ML libraries for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 Seeding everything with seed {seed}")


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
            # print(f"Warning: Feature '{feature}' not found in the input data. Filling with NaN.")
            df[feature] = np.nan

    # Extract features
    X = df[feature_names].copy()

    # Convert categorical features to numeric codes
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    # Fill missing values
    X = X.fillna(0)

    return X.to_numpy()


def generate_comprehensive_predictions(model_type, dataset_path, output_path):
    """
    Generate comprehensive predictions using all available models (task/action, binary/multilabel).

    Parameters:
    -----------
    model_type : str
        Type of model to use ('gb' or 'nn').
    dataset_path : str
        Path to the input data file.
    output_path : str
        Path to save the comprehensive predictions.
    """
    # Read pickled dataset
    df = pd.read_pickle(dataset_path)

    # Add a level column to the dataset
    df['level'] = df['eid'].apply(lambda x: 'task' if x == 0 else 'action')

    # Create helper functions for safe type casting
    int_cast = lambda x: int(x) if not np.isnan(x) else 0
    float_cast = lambda x: float(x) if not np.isnan(x) else 0

    # Separate task and action data for model predictions
    task_df = df[df['level'] == 'task'].copy()
    action_df = df[df['level'] == 'action'].copy()

    print(f"Task data shape: {task_df.shape}")
    print(f"Action data shape: {action_df.shape}")

    # List of model configurations to use
    model_configs = [
        {'level': 'task', 'label_type': 'binary'},
        {'level': 'task', 'label_type': 'multilabel'},
        {'level': 'action', 'label_type': 'binary'},
        {'level': 'action', 'label_type': 'multilabel'}
    ]

    # Dictionary to store all model predictions
    model_results = {}

    # Load and run predictions for each model configuration
    for config in model_configs:
        level = config['level']
        label_type = config['label_type']

        # Determine model path
        model_path = os.path.join('saved_models', f'{model_type}_{level}_{label_type}.pkl')

        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found. Skipping.")
            continue

        print(f"Running predictions for {level}-{label_type} using {model_path}")

        # Load model
        model_data = load_model(model_path)
        model = model_data['model']
        feature_names = model_data['feature_names']

        # Select appropriate dataset
        df_to_use = task_df if level == 'task' else action_df

        # Prepare features
        X = prepare_input_data(df_to_use, feature_names)

        # Generate predictions
        if label_type == 'binary':
            y_pred_prob = model.predict_proba(X)
            y_pred = model.predict(X)

            # Store in results
            rev_mapping = {0: 'no', 1: 'yes'}
            model_results[f'{level}_{label_type}_pred'] = y_pred.tolist()
            model_results[f'{level}_{label_type}_labels'] = [rev_mapping.get(y_i) for y_i in y_pred.tolist()]
            model_results[f'{level}_{label_type}_probas'] = y_pred_prob[:, 1].tolist()

        else:  # multilabel
            y_pred_prob_raw = model.predict_proba(X)
            y_pred_prob = np.array([proba[:, 1] for proba in y_pred_prob_raw]).T
            y_pred = model.predict(X)

            rev_mapping = rev_task_smells_mapping if level == 'task' else rev_action_smells_mapping
            y_pred_labels = []
            y_pred_probas = []

            for pred_row, proba_row in zip(y_pred, y_pred_prob):
                y_indices = np.nonzero(pred_row)[0]
                y_pred_labels.append([
                    rev_mapping[y_i + 1] for y_i in y_indices
                ])
                y_pred_probas.append([
                    float(proba_row[y_i]) for y_i in y_indices
                ])

            # Store in results
            model_results[f'{level}_{label_type}_pred'] = y_pred.tolist()
            model_results[f'{level}_{label_type}_labels'] = y_pred_labels
            model_results[f'{level}_{label_type}_probas'] = y_pred_probas

    # Create lookup dictionaries to map row identifiers to predictions
    task_indices = {idx: i for i, idx in enumerate(task_df.index)}
    action_indices = {idx: i for i, idx in enumerate(action_df.index)}

    # Now construct the unified event-based output
    all_events = []

    # Process all events in a single loop
    for i, row in df.iterrows():
        level = row['level']
        is_task = level == 'task'

        # Get the correct index for predictions
        pred_idx = task_indices.get(row.name, 0) if is_task else action_indices.get(row.name, 0)

        # Get predictions based on level
        if is_task:
            binary_pred = model_results.get('task_binary_pred')[pred_idx]
            binary_label = model_results.get('task_binary_labels')[pred_idx]
            binary_proba = model_results.get('task_binary_probas')[pred_idx]
            multilabel_pred = model_results.get('task_multilabel_pred')[pred_idx]
            multilabel_labels = model_results.get('task_multilabel_labels')[pred_idx]
            multilabel_probas = model_results.get('task_multilabel_probas')[pred_idx]
        else:
            binary_pred = model_results.get('action_binary_pred')[pred_idx]
            binary_label = model_results.get('action_binary_labels')[pred_idx]
            binary_proba = model_results.get('action_binary_probas')[pred_idx]
            multilabel_pred = model_results.get('action_multilabel_pred')[pred_idx]
            multilabel_labels = model_results.get('action_multilabel_labels')[pred_idx]
            multilabel_probas = model_results.get('action_multilabel_probas')[pred_idx]


        # import ipdb; ipdb.set_trace()
        # Create the event data
        event = {
            'id': i,
            'eid': int(row.get('pid', 0)),
            'pid': int(row.get('eid', 0)),
            'rid': float(row.get('rid', 0)),
            'srid': float(row.get('srid', 0)),
            'rtid': float(row.get('rtid', 0)),
            'task_id': int(row.get('task_id', 0)),
            'event_type': str(row.get('event', '')),
            'dom_object': str(row.get('dom_object', '')),
            'xpath': str(row.get('xpath', '')),
            'url': str(row.get('url', '')),
            'timestamp': str(row.get('time', '')),
            'duration': float(row.get('event_duration', 0.0)),
            'level': level,

            'has_smell_binary': bool(binary_pred),
            'smell_probas_binary': binary_proba,
            'detected_smells_multilabel': multilabel_labels,
            'smell_probas_multilabel': [float(p) for p in multilabel_probas],

            'sam_valence': int(row.get('sam_valence', 5)),
            'sam_arousal': int(row.get('sam_arousal', 5)),

            'episode_details': {} if not is_task else {
                'episode_tabchange_duration': float(row.get('episode_tabchange_duration', 0)),
                'episode_change_duration': float(row.get('episode_change_duration', 0)),
                'episode_click_duration': float(row.get('episode_click_duration', 0)),
                'episode_scroll_duration': float(row.get('episode_scroll_duration', 0)),
                'episode_total_duration': float(row.get('episode_total_duration', 0)),
                'episode_tabchange_num': int(row.get('episode_tabchange_num', 0)),
                'episode_change_num': int(row.get('episode_change_num', 0)),
                'episode_click_num': int(row.get('episode_click_num', 0)),
                'episode_scroll_num': int(row.get('episode_scroll_num', 0)),
                'episode_keypress_num': int(row.get('episode_keypress_num', 0)),
                'episode_total_events': int(row.get('episode_total_events', 0)),
            },
            'event_details': {} if is_task else {
                'click_text': str(row.get('event_click_text', '')),
                'click_button': str(row.get('event_click_button', '')),
                'click_pos_x': float(row.get('event_click_pos_x_norm', 0)),
                'click_pos_y': float(row.get('event_click_pos_y_norm', 0)),
                'change_text': str(row.get('event_change_text', '')),
                'change_value': str(row.get('event_change_value', '')),
                'scroll_time': int(row.get('event_scroll_time', 0)),
                'keypress_num': int(row.get('event_keypress_num', 0)),
            },
            'biometrics': {
                'episode_eeg': {} if not is_task else {
                    'theta_avg': float_cast(row.get('episode_bit_eeg_theta_avg', 0)),
                    'alpha_low_avg': float_cast(row.get('episode_bit_eeg_alpha_low_avg', 0)),
                    'alpha_high_avg': float_cast(row.get('episode_bit_eeg_alpha_high_avg', 0)),
                    'beta_avg': float_cast(row.get('episode_bit_eeg_beta_avg', 0)),
                    'gamma_avg': float_cast(row.get('episode_bit_eeg_gamma_avg', 0)),
                },
                'episode_bvp': {} if not is_task else {
                    'hr_avg': float_cast(row.get('episode_bit_bvp_hr_avg', 0)),
                    'hr_std': float_cast(row.get('episode_bit_bvp_hr_std', 0)),
                },
                'episode_face': {} if not is_task else {
                    'most_likely': int_cast(row.get('episode_face_avg_most_likely', 0)),
                    'anger': float_cast(row.get('episode_face_avg_0_anger_proba', 0)),
                    'disgust': float_cast(row.get('episode_face_avg_1_disgust_proba', 0)),
                    'fear': float_cast(row.get('episode_face_avg_2_fear_proba', 0)),
                    'enjoyment': float_cast(row.get('episode_face_avg_3_enjoyment_proba', 0)),
                    'contempt': float_cast(row.get('episode_face_avg_4_contempt_proba', 0)),
                    'sadness': float_cast(row.get('episode_face_avg_5_sadness_proba', 0)),
                    'surprise': float_cast(row.get('episode_face_avg_6_surprise_proba', 0)),
                },
                'event_eeg': {} if is_task else {
                    'theta_avg': float_cast(row.get('event_bit_eeg_theta_avg', 0)),
                    'alpha_low_avg': float_cast(row.get('event_bit_eeg_alpha_low_avg', 0)),
                    'alpha_high_avg': float_cast(row.get('event_bit_eeg_alpha_high_avg', 0)),
                    'beta_avg': float_cast(row.get('event_bit_eeg_beta_avg', 0)),
                    'gamma_avg': float_cast(row.get('event_bit_eeg_gamma_avg', 0)),
                },
                'event_bvp': {} if is_task else {
                    'hr_avg': float_cast(row.get('event_bit_bvp_hr_avg', 0)),
                    'hr_std': float_cast(row.get('event_bit_bvp_hr_std', 0)),
                },
                'event_face': {} if is_task else {
                    'most_likely': int_cast(row.get('event_face_avg_most_likely', 0)),
                    'anger': float_cast(row.get('event_face_avg_0_anger_proba', 0)),
                    'disgust': float_cast(row.get('event_face_avg_1_disgust_proba', 0)),
                    'fear': float_cast(row.get('event_face_avg_2_fear_proba', 0)),
                    'enjoyment': float_cast(row.get('event_face_avg_3_enjoyment_proba', 0)),
                    'contempt': float_cast(row.get('event_face_avg_4_contempt_proba', 0)),
                    'sadness': float_cast(row.get('event_face_avg_5_sadness_proba', 0)),
                    'surprise': float_cast(row.get('event_face_avg_6_surprise_proba', 0)),
                }
            }
        }
        all_events.append(event)

    # Calculate various smell statistics
    smell_stats = {
        'task_level': {
            'total': len(task_df),
            'binary': {
                'total_smells': sum(model_results.get('task_binary_pred', [False] * len(task_df))),
                'smell_ratio': float_cast(
                    sum(model_results.get('task_binary_pred', [False] * len(task_df))) / len(task_df) if len(
                        task_df) > 0 else 0)
            },
            'multilabel': {
                'total_items_with_smells': sum(
                    any(pred) for pred in model_results.get('task_multilabel_pred', [[False] * 7] * len(task_df))),
                'smell_counts': {}
            }
        },
        'action_level': {
            'total': len(action_df),
            'binary': {
                'total_smells': sum(model_results.get('action_binary_pred', [False] * len(action_df))),
                'smell_ratio': float_cast(
                    sum(model_results.get('action_binary_pred', [False] * len(action_df))) / len(action_df) if len(
                        action_df) > 0 else 0)
            },
            'multilabel': {
                'total_items_with_smells': sum(
                    any(pred) for pred in model_results.get('action_multilabel_pred', [[False] * 4] * len(action_df))),
                'smell_counts': {}
            }
        }
    }

    # Count occurrences of each smell type for task-level multilabel
    if 'task_multilabel_labels' in model_results:
        smell_counts = {}
        for labels in model_results['task_multilabel_labels']:
            for smell in labels:
                smell_counts[smell] = smell_counts.get(smell, 0) + 1
        smell_stats['task_level']['multilabel']['smell_counts'] = smell_counts

    # Count occurrences of each smell type for action-level multilabel
    if 'action_multilabel_labels' in model_results:
        smell_counts = {}
        for labels in model_results['action_multilabel_labels']:
            for smell in labels:
                smell_counts[smell] = smell_counts.get(smell, 0) + 1
        smell_stats['action_level']['multilabel']['smell_counts'] = smell_counts

    # Create the comprehensive output
    comprehensive_output = {
        'events': all_events,
        'statistics': smell_stats,
        'metadata': {
            'model_type': model_type,
            'dataset_path': dataset_path,
            'task_count': len(task_df),
            'action_count': len(action_df),
            'total_count': len(df),
            'total_url_changes': len(df[df['event'] == 'tabchange']),
            'total_clicks': len(df[df['event'] == 'click']),
            'total_changes': len(df[df['event'] == 'change']),
            'total_scrolls': len(df[df['event'] == 'scroll']),
            'total_keypresses': len(df[df['event'] == 'keypress']),
            'available_models': [f"{config['level']}_{config['label_type']}" for config in model_configs if
                                 os.path.exists(os.path.join('saved_models',
                                                             f'{model_type}_{config["level"]}_{config["label_type"]}.pkl'))]
        }
    }

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save the comprehensive output
    with open(output_path, 'w') as f:
        json.dump(comprehensive_output, f, indent=2)

    print(f"Comprehensive predictions saved to {output_path}")
    return comprehensive_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions using a gradient boosting model.')
    parser.add_argument('dataset_path', type=str, help='Path to the input data file (.pkl)')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the predictions')
    args = parser.parse_args()

    seed_everything(seed=42)

    # model type is 'gb'
    model_type = 'gb'

    # Determine output path if not provided
    if args.output_path is None:
        args.output_path = os.path.join(os.path.dirname(args.dataset_path), f'predictions_{model_type}.json')

    generate_comprehensive_predictions(model_type, args.dataset_path, args.output_path)
