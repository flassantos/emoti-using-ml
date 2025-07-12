import random
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
# Metrics
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score,
    average_precision_score,
    jaccard_score,
    hamming_loss
)
# Machine Learning and Cross-Validation
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from run_grid_search_binary import get_features, get_predefined_split, print_label_distribution


def hamming_score(*args, **kwargs):
    # Define hamming_score as 1 - hamming_loss
    return 1 - hamming_loss(*args, **kwargs)


def evaluate_feature_ablation(df, X_columns, y_column, level='task', network='ALL', n_splits=5,
                              is_multilabel=False, feature_combinations=None):
    """
    Evaluate different feature combinations through ablation.
    """
    if feature_combinations is None:
        raise ValueError("feature_combinations must be provided.")

    # Define X and y
    X_df = df[X_columns].copy()
    y_original = df[y_column]

    # Process multilabel data
    if is_multilabel:
        # Handle multilabel data - converting it to binary format
        # Convert [0] (none label) to an empty array
        y_processed = y_original.map(lambda x: [] if len(x) == 1 and x[0] == 0 else x)

        # Use MultiLabelBinarizer to convert to binary matrix
        mlb = MultiLabelBinarizer()
        y_bin = mlb.fit_transform(y_processed)

        # For CV splitting, we'll use a simplified representation
        y_strat = y_processed  # Simple heuristic for stratification

        # For actual evaluation
        y = y_bin
    else:
        # For binary classification, use as is
        y = y_original
        y_strat = y_original

    # Convert categorical features to numeric codes
    for col in X_df.select_dtypes(include=['object']).columns:
        X_df[col] = X_df[col].astype('category').cat.codes

    # Fill NaN values in the original dataset
    X_df[['sam_valence', 'sam_arousal']] = X_df[['sam_valence', 'sam_arousal']].fillna(5)
    X_df = X_df.fillna(0)

    # Define cross-validation strategy
    if network != 'ALL':
        cv = get_predefined_split(df, network=network, n_splits=n_splits)
    else:
        if is_multilabel:
            # Use KFold for multilabel to avoid stratification issues
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Define metrics based on task type
    if is_multilabel:
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro', zero_division=0),
            'recall': make_scorer(recall_score, average='macro', zero_division=0),
            'f1': make_scorer(f1_score, average='samples', zero_division=0),
            'f1_micro': make_scorer(f1_score, average='micro', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
            'jaccard': make_scorer(jaccard_score, average='samples'),
            'hamming_score': make_scorer(hamming_score),
            'auprc': make_scorer(average_precision_score, average='macro')
        }
    else:
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro', zero_division=0),
            'recall': make_scorer(recall_score, average='macro', zero_division=0),
            'f1': make_scorer(f1_score, pos_label=1, zero_division=0),
            'f1_micro': make_scorer(f1_score, average='micro', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'jaccard': make_scorer(jaccard_score),
            'hamming_score': make_scorer(hamming_score),
            'auprc': make_scorer(average_precision_score, needs_proba=True)
        }

    results = {}

    # Identify columns for each feature type - define outside the loop
    log_cols = get_features(log=True, eeg=False, face=False, sam=False, norm=True, non_norm=False)
    eeg_cols = get_features(log=False, eeg=True, face=False, sam=False, norm=True, non_norm=False)
    face_cols = get_features(log=False, eeg=False, face=True, sam=False, norm=True, non_norm=False)
    sam_cols = get_features(log=False, eeg=False, face=False, sam=True, norm=True, non_norm=False)

    # For each feature combination
    for include_log, include_eeg, include_face, include_sam in feature_combinations:
        # Create a name for this combination
        combo_name = f"Log:{include_log} EEG:{include_eeg} Face:{include_face} SAM:{include_sam}"
        print(f"\nEvaluating: {combo_name}")

        # Initialize the classifier
        if is_multilabel:
            if level == 'task':
                model = MultiOutputClassifier(
                    GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                )
            else:
                model = MultiOutputClassifier(
                    GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                )
        else:
            if level == 'task':
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42
                )

        # Store fold scores
        fold_scores = {metric: [] for metric in scoring.keys()}

        # Perform cross-validation manually - use y_strat for splitting
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_df, y_strat), 1):
            print(f"Fold {fold_idx}:")
            print(f"  Train shape: {X_df.iloc[train_idx].shape}")
            print(f"  Test shape: {X_df.iloc[test_idx].shape}")

            # For display, use the original format if multilabel
            if is_multilabel:
                y_train_display = y_original.iloc[train_idx].to_numpy()
                y_test_display = y_original.iloc[test_idx].to_numpy()
            else:
                y_train_display = y.iloc[train_idx].to_numpy()
                y_test_display = y.iloc[test_idx].to_numpy()

            print_label_distribution(y_train_display, y_test_display)

            # Get the original train and test features
            X_train = X_df.iloc[train_idx].copy()
            X_test = X_df.iloc[test_idx].copy()

            # Make a copy of X_test for masking
            X_test_masked = X_test.copy()

            # Apply masking only to the test set
            if not include_log:
                X_test_masked[log_cols] = np.nan

            if not include_eeg:
                X_test_masked[eeg_cols] = np.nan

            if not include_face:
                X_test_masked[face_cols] = np.nan

            if not include_sam:
                X_test_masked[sam_cols] = np.nan

            # Replace NaN values with zeros in the test set
            X_test_masked = X_test_masked.fillna(0)

            # Get the target values
            if isinstance(y, np.ndarray):
                y_train = y[train_idx]
                y_test = y[test_idx]
            else:
                y_train = y.iloc[train_idx].to_numpy()
                y_test = y.iloc[test_idx].to_numpy()

            # Convert to numpy arrays for training
            X_train_array = X_train.to_numpy()
            X_test_array = X_test_masked.to_numpy()

            # Fit the model
            model.fit(X_train_array, y_train)

            # Calculate metrics
            if is_multilabel:
                y_pred = model.predict(X_test_array)

                for metric_name, scorer in scoring.items():
                    try:
                        if metric_name == 'auprc':
                            # For multilabel, we use the predictions directly
                            score = average_precision_score(y_test, y_pred, average='macro')
                        elif metric_name == 'precision':
                            score = precision_score(y_test, y_pred, average='macro', zero_division=0)
                        elif metric_name == 'recall':
                            score = recall_score(y_test, y_pred, average='macro', zero_division=0)
                        elif metric_name == 'f1':
                            score = f1_score(y_test, y_pred, average='samples', zero_division=0)
                        elif metric_name == 'f1_micro':
                            score = f1_score(y_test, y_pred, average='micro', zero_division=0)
                        elif metric_name == 'f1_macro':
                            score = f1_score(y_test, y_pred, average='macro', zero_division=0)
                        elif metric_name == 'f1_weighted':
                            score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        elif metric_name == 'jaccard':
                            score = jaccard_score(y_test, y_pred, average='macro')
                        else:
                            score = scorer._score_func(y_test, y_pred)
                        fold_scores[metric_name].append(score)
                    except Exception as e:
                        print(f"Error calculating {metric_name}: {e}")
                        # Use a default value or skip
                        fold_scores[metric_name].append(0)
            else:
                y_pred = model.predict(X_test_array)
                y_pred_proba = model.predict_proba(X_test_array)

                for metric_name, scorer in scoring.items():

                    try:
                        if metric_name == 'auprc':
                            # For multilabel, we use the predictions directly
                            score = average_precision_score(y_test, y_pred_proba[:, 1])
                        elif metric_name == 'precision':
                            score = precision_score(y_test, y_pred, average='macro', zero_division=0)
                        elif metric_name == 'recall':
                            score = recall_score(y_test, y_pred, average='macro', zero_division=0)
                        elif metric_name == 'f1':
                            score = f1_score(y_test, y_pred, average='binary', zero_division=0)
                        elif metric_name == 'f1_micro':
                            score = f1_score(y_test, y_pred, average='micro', zero_division=0)
                        elif metric_name == 'f1_macro':
                            score = f1_score(y_test, y_pred, average='macro', zero_division=0)
                        elif metric_name == 'f1_weighted':
                            score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        elif metric_name == 'jaccard':
                            score = jaccard_score(y_test, y_pred, average='macro')
                        else:
                            score = scorer._score_func(y_test, y_pred)
                        fold_scores[metric_name].append(score)
                    except Exception as e:
                        print(f"Error calculating {metric_name}: {e}")
                        # Use a default value or skip
                        fold_scores[metric_name].append(0)

        # Calculate average scores across folds
        avg_scores = {metric: np.mean(scores) for metric, scores in fold_scores.items()}
        results[combo_name] = avg_scores

        # Print results
        print("=" * 40)
        print(f"Results for {combo_name}:")
        print(f"{'Metric':<15} {'Score':<10}")
        print("-" * 40)

        for metric_name, score in avg_scores.items():
            print(f"{metric_name:<15} {score:.4f}")

        print("=" * 40)

    return results


if __name__ == '__main__':
    # Define level and label type
    level = sys.argv[1] if len(sys.argv) > 1 else 'task'
    label_type = sys.argv[2] if len(sys.argv) > 2 else 'binary'

    # Set random seed
    random.seed(42)
    np.random.seed(42)

    # Load dataset
    df_all = pd.read_pickle('dfs/df_all.pkl')

    # Filter data based on level and label type
    if label_type == 'binary':
        is_multilabel = False
        if level == 'task':
            df = df_all[(df_all['eid'] == 0)].copy()  # task-level
            target_column = 'y_binary'
        else:
            df = df_all[(df_all['eid'] > 0)].copy()  # action-level
            target_column = 'y_binary'
    else:  # multilabel
        is_multilabel = True
        if level == 'task':
            df = df_all[(df_all['eid'] == 0)].copy()  # task-level
            target_column = 'y_task_multilabel'
        else:
            df = df_all[(df_all['eid'] > 0)].copy()  # action-level
            target_column = 'y_action_multilabel'

    print('Running for level:', level)
    print('Running for label type:', label_type)
    print(f"Target column: {target_column}, Multilabel: {is_multilabel}")

    # Create output directory if it doesn't exist
    import os

    os.makedirs('results', exist_ok=True)

    # Get all potential features (we'll use normalized features only)
    all_features = get_features(log=True, eeg=True, face=True, sam=True, norm=True, non_norm=False)

    # Define feature combinations to test
    feature_combinations = [
        (True, True, True, True),   # Log, EEG, Face, SAM
        (True, True, True, False),  # No SAM
        (True, True, False, False),  # No SAM, No Face
        (True, False, False, False),  # No SAM, No Face, No EEG
        (True, True, False, True),  # No Face
        (True, False, True, True)   # No LOG
    ]

    # Define networks to test
    networks = [
        # 'SN_1',
        # 'SN_2',
        # 'SN_3',
        'ALL'
    ]

    # Number of CV splits
    n_splits = 5

    # Store all results
    all_results = {}

    # For each network
    for network in networks:
        print(f"\n{'=' * 50}")
        print(f"Network: {network}")
        print(f"{'=' * 50}")

        # Evaluate feature ablation
        results = evaluate_feature_ablation(
            df=df,
            X_columns=all_features,
            y_column=target_column,
            level=level,
            network=network,
            n_splits=n_splits,
            is_multilabel=is_multilabel,
            feature_combinations=feature_combinations
        )

        all_results[network] = results

    # Print summary of results
    print("\n" + "=" * 80)
    print(f"SUMMARY OF RESULTS FOR {level.upper()} LEVEL, {label_type.upper()} LABELS")
    print("=" * 80)

    # Create a summary table with F1 scores for each network and feature combination
    for network in networks:
        print(f"\nNetwork: {network}")
        print(f"{'Feature Combination':<35} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Jaccard':<10} {'Hamming':<10}")
        print("-" * 70)

        for combo_name, metrics in all_results[network].items():
            f1_macro = metrics.get('f1_macro', 0)
            prec_macro = metrics.get('precision', 0)
            rec_macro = metrics.get('recall', 0)
            acc = metrics.get('accuracy', 0)
            jaccard = metrics.get('jaccard', 0)
            hamming = metrics.get('hamming_score', 0)
            print(f"{combo_name:<35}    {acc:.4f}    {prec_macro:.4f}    {rec_macro:.4f}    {f1_macro:.4f}    {jaccard:.4f}    {hamming:.4f}")

    print("\n" + "=" * 80)

    # Convert results to a format that can be serialized to JSON
    serializable_results = {}
    for network, network_results in all_results.items():
        serializable_results[network] = {}
        for combo, metrics in network_results.items():
            serializable_results[network][combo] = {k: float(v) for k, v in metrics.items()}

    # Save results to file
    import json

    with open(f'results/feature_ablation_results_{level}_{label_type}_new.json', 'w') as f:
        json.dump(serializable_results, f)
