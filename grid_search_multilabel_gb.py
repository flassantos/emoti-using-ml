import sys
import random
from collections import Counter
import warnings


import pandas as pd
import numpy as np


# Machine Learning and Cross-Validation
from sklearn.model_selection import cross_validate, StratifiedKFold, PredefinedSplit, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier


# Metrics
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score,
    average_precision_score,
    jaccard_score,
    hamming_loss
)


# Suppress ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")


def get_test_usernames_for_network(df, network='SN_1', n_splits=5):
    networks_usernames = list(sorted(df[['network', 'username']].agg(lambda x: x['network'] + '-' + x['username'], axis=1).unique()))
    valid_usernames = [x.split('-')[1] for x in networks_usernames if x.startswith(network)]
    np.random.shuffle(valid_usernames)
    return np.array_split(valid_usernames, n_splits)


def get_predefined_split(df, network='SN_1', n_splits=5):
    target_folds = get_test_usernames_for_network(df, network=network, n_splits=n_splits)
    # Print split indices for the test network
    print(target_folds)
    test_fold = -np.ones(df.shape[0], dtype=int)  # Initialize with -1 (indicating train)
    for fold_idx, usernames in enumerate(target_folds):
        fold_indices = np.where(df['username'].apply(lambda x: x in usernames))
        test_fold[fold_indices] = fold_idx  # Assign test indices for the fold
    # Create the PredefinedSplit object
    ps = PredefinedSplit(test_fold=test_fold)
    return ps


def print_label_distribution(y_train, y_test):
    
    # Binary case: print percentage distribution of 0s and 1s
    if not isinstance(y_train[0], (tuple, list)):  # Binary classification
        train_dist = Counter(y_train)
        test_dist = Counter(y_test)
        
        # Calculate percentages
        train_total = sum(train_dist.values())
        test_total = sum(test_dist.values())
        train_0_pct = (train_dist[0] / train_total) * 100
        train_1_pct = (train_dist[1] / train_total) * 100
        test_0_pct = (test_dist[0] / test_total) * 100
        test_1_pct = (test_dist[1] / test_total) * 100

        print(f"  Train Label Distribution: 0: {train_0_pct:.2f}%, 1: {train_1_pct:.2f}%")
        print(f"  Test Label Distribution:  0: {test_0_pct:.2f}%, 1: {test_1_pct:.2f}%")
    else:  # Multilabel case
        # Flatten multilabel targets and count occurrences for top 5 labels
        train_dist = Counter([label for labels in y_train for label in labels if label or label == 0])
        test_dist = Counter([label for labels in y_test for label in labels if label or label == 0])
        
        # Convert counts to percentages
        train_total = sum(train_dist.values())
        test_total = sum(test_dist.values())
        train_pct = {label: (count / train_total) * 100 for label, count in train_dist.items()}
        test_pct = {label: (count / test_total) * 100 for label, count in test_dist.items()}
        
        # Get top 5 labels by percentage
        top_5_train = sorted(train_pct.items(), key=lambda x: x[1], reverse=True)[:5]
        top_5_test = sorted(test_pct.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"  Top 5 Train Labels: {[(label, f'{pct:.2f}%') for label, pct in top_5_train]}")
        print(f"  Top 5 Test Labels:  {[(label, f'{pct:.2f}%') for label, pct in top_5_test]}")


def hamming_score(*args, **kwargs):
    # Define hamming_score as 1 - hamming_loss
    return 1 - hamming_loss(*args, **kwargs)



def get_features(log=True, eeg=True, face=True, sam=True, norm=True, non_norm=False):
    features = []

    if norm is False and non_norm is False:
        raise Exception("Either norm or non normalized features should be selected.")
    
    if log is False and eeg is False and face is False and sam is False:
        raise Exception("At least one feature source should be selected.")
    
    # log-based features:
    if log:
        features += [
            # 'pid',
            # 'eid',
            'rid',
            'srid',
            'rtid',
            'event',
            'dom_object',
            'finished_task',
        ]

        if non_norm:
            features += [
                'event_duration',
                'event_cum_duration',

                'episode_click_duration',
                'episode_change_duration',
                'episode_scroll_duration',
                'episode_tabchange_duration',
                'episode_total_duration',

                'episode_click_num',
                'episode_change_num',
                'episode_scroll_num',
                'episode_tabchange_num',
                'episode_keypress_num',
                'episode_total_events',

                'event_click_text_length',
                'event_click_pos_x',
                'event_click_pos_y',
                'event_change_text_length',
                'event_change_value_length',

                'event_keypress_num',
                'event_scroll_time',
                'event_scroll_num',

                'episode_tabchange_num_per_min',
                'episode_change_num_per_min',
                'episode_click_num_per_min',
                'episode_scroll_num_per_min',
                'episode_keypress_num_per_min',
                'episode_total_events_per_min',

                'xpath_depth',
                'xpath_div_count',
                'xpath_num_unique',

                'event_duration_max',
                'event_duration_min',
                'event_duration_last',
                'event_duration_first',
                'event_duration_second',
                'event_duration_avg',

                'episode_total_duration_prev_1',
                'episode_total_duration_prev_2',
                'episode_total_duration_prev_3',
                'episode_total_duration_next_1',
                'episode_total_duration_next_2',
                'episode_total_duration_next_3',
                'episode_total_events_prev_1',
                'episode_total_events_prev_2',
                'episode_total_events_prev_3',
                'episode_total_events_next_1',
                'episode_total_events_next_2',
                'episode_total_events_next_3',
            ]


        if norm:
            features += [
                'event_duration_norm',
                'event_cum_duration_norm',

                'episode_tabchange_duration_norm',
                'episode_change_duration_norm',
                'episode_click_duration_norm',
                'episode_scroll_duration_norm',
                'episode_total_duration_norm',

                'episode_tabchange_num_norm',
                'episode_change_num_norm',
                'episode_click_num_norm',
                'episode_scroll_num_norm',
                'episode_keypress_num_norm',
                'episode_total_events_norm',

                'event_click_text',
                'event_click_button',
                'event_change_text',
                'event_change_value',
                'event_change_value_sim_session',
                'event_change_value_sim_episode',
                'event_change_text_length_norm',
                'event_change_value_length_norm',
                'event_click_text_length_norm',
                'event_click_pos_x_norm',
                'event_click_pos_y_norm',

                'event_keypress_num_norm',
                'event_scroll_time_norm',
                'event_scroll_num_norm',

                'episode_tabchange_num_per_min_norm',
                'episode_change_num_per_min_norm',
                'episode_click_num_per_min_norm',
                'episode_scroll_num_per_min_norm',
                'episode_keypress_num_per_min_norm',
                'episode_total_events_per_min_norm',

                'xpath_depth_norm',
                'xpath_div_count_norm',
                'xpath_num_unique_norm',
                'dom_object_prev',
                'dom_object_next',

                'event_duration_norm_max',
                'event_duration_norm_min',
                'event_duration_norm_last',
                'event_duration_norm_first',
                'event_duration_norm_second',
                'event_duration_norm_avg',

                'episode_total_duration_norm_prev_1',
                'episode_total_duration_norm_prev_2',
                'episode_total_duration_norm_prev_3',
                'episode_total_duration_norm_next_1',
                'episode_total_duration_norm_next_2',
                'episode_total_duration_norm_next_3',
                'episode_total_events_norm_prev_1',
                'episode_total_events_norm_prev_2',
                'episode_total_events_norm_prev_3',
                'episode_total_events_norm_next_1',
                'episode_total_events_norm_next_2',
                'episode_total_events_norm_next_3'
            ]


    # EEG based features:
    if eeg:
        if non_norm:
            features += [
                'episode_bit_eeg_signal_min',
                'episode_bit_eeg_signal_max',
                'episode_bit_eeg_signal_avg',
                'episode_bit_eeg_signal_std',
                'episode_bit_eeg_features_ts_min',
                'episode_bit_eeg_features_ts_max',
                'episode_bit_eeg_features_ts_avg',
                'episode_bit_eeg_features_ts_std',
                'episode_bit_eeg_theta_min',
                'episode_bit_eeg_theta_max',
                'episode_bit_eeg_theta_avg',
                'episode_bit_eeg_theta_std',
                'episode_bit_eeg_alpha_low_min',
                'episode_bit_eeg_alpha_low_max',
                'episode_bit_eeg_alpha_low_avg',
                'episode_bit_eeg_alpha_low_std',
                'episode_bit_eeg_alpha_high_min',
                'episode_bit_eeg_alpha_high_max',
                'episode_bit_eeg_alpha_high_avg',
                'episode_bit_eeg_alpha_high_std',
                'episode_bit_eeg_beta_min',
                'episode_bit_eeg_beta_max',
                'episode_bit_eeg_beta_avg',
                'episode_bit_eeg_beta_std',
                'episode_bit_eeg_gamma_min',
                'episode_bit_eeg_gamma_max',
                'episode_bit_eeg_gamma_avg',
                'episode_bit_eeg_gamma_std',
                # 'episode_bit_eeg_lim_a',
                # 'episode_bit_eeg_lim_b',
                'episode_bit_bvp_signal_min',
                'episode_bit_bvp_signal_max',
                'episode_bit_bvp_signal_avg',
                'episode_bit_bvp_signal_std',
                'episode_bit_bvp_onsets_min',
                'episode_bit_bvp_onsets_max',
                'episode_bit_bvp_onsets_avg',
                'episode_bit_bvp_onsets_std',
                'episode_bit_bvp_hr_min',
                'episode_bit_bvp_hr_max',
                'episode_bit_bvp_hr_avg',
                'episode_bit_bvp_hr_std',
                # 'episode_bit_bvp_lim_a',
                # 'episode_bit_bvp_lim_b',
                'event_bit_eeg_signal_min',
                'event_bit_eeg_signal_max',
                'event_bit_eeg_signal_avg',
                'event_bit_eeg_signal_std',
                'event_bit_eeg_features_ts_min',
                'event_bit_eeg_features_ts_max',
                'event_bit_eeg_features_ts_avg',
                'event_bit_eeg_features_ts_std',
                'event_bit_eeg_theta_min',
                'event_bit_eeg_theta_max',
                'event_bit_eeg_theta_avg',
                'event_bit_eeg_theta_std',
                'event_bit_eeg_alpha_low_min',
                'event_bit_eeg_alpha_low_max',
                'event_bit_eeg_alpha_low_avg',
                'event_bit_eeg_alpha_low_std',
                'event_bit_eeg_alpha_high_min',
                'event_bit_eeg_alpha_high_max',
                'event_bit_eeg_alpha_high_avg',
                'event_bit_eeg_alpha_high_std',
                'event_bit_eeg_beta_min',
                'event_bit_eeg_beta_max',
                'event_bit_eeg_beta_avg',
                'event_bit_eeg_beta_std',
                'event_bit_eeg_gamma_min',
                'event_bit_eeg_gamma_max',
                'event_bit_eeg_gamma_avg',
                'event_bit_eeg_gamma_std',
                # 'event_bit_eeg_lim_a',
                # 'event_bit_eeg_lim_b',
            ]
        if norm:
            features += [
                'episode_bit_eeg_signal_min_norm',
                'episode_bit_eeg_signal_max_norm',
                'episode_bit_eeg_signal_avg_norm',
                'episode_bit_eeg_signal_std_norm',
                'episode_bit_eeg_features_ts_min_norm',
                'episode_bit_eeg_features_ts_max_norm',
                'episode_bit_eeg_features_ts_avg_norm',
                'episode_bit_eeg_features_ts_std_norm',
                'episode_bit_eeg_theta_min_norm',
                'episode_bit_eeg_theta_max_norm',
                'episode_bit_eeg_theta_avg_norm',
                'episode_bit_eeg_theta_std_norm',
                'episode_bit_eeg_alpha_low_min_norm',
                'episode_bit_eeg_alpha_low_max_norm',
                'episode_bit_eeg_alpha_low_avg_norm',
                'episode_bit_eeg_alpha_low_std_norm',
                'episode_bit_eeg_alpha_high_min_norm',
                'episode_bit_eeg_alpha_high_max_norm',
                'episode_bit_eeg_alpha_high_avg_norm',
                'episode_bit_eeg_alpha_high_std_norm',
                'episode_bit_eeg_beta_min_norm',
                'episode_bit_eeg_beta_max_norm',
                'episode_bit_eeg_beta_avg_norm',
                'episode_bit_eeg_beta_std_norm',
                'episode_bit_eeg_gamma_min_norm',
                'episode_bit_eeg_gamma_max_norm',
                'episode_bit_eeg_gamma_avg_norm',
                'episode_bit_eeg_gamma_std_norm',
                # 'episode_bit_eeg_lim_a_norm',
                # 'episode_bit_eeg_lim_b_norm',
                'episode_bit_bvp_signal_min_norm',
                'episode_bit_bvp_signal_max_norm',
                'episode_bit_bvp_signal_avg_norm',
                'episode_bit_bvp_signal_std_norm',
                'episode_bit_bvp_onsets_min_norm',
                'episode_bit_bvp_onsets_max_norm',
                'episode_bit_bvp_onsets_avg_norm',
                'episode_bit_bvp_onsets_std_norm',
                'episode_bit_bvp_hr_min_norm',
                'episode_bit_bvp_hr_max_norm',
                'episode_bit_bvp_hr_avg_norm',
                'episode_bit_bvp_hr_std_norm',
                # 'episode_bit_bvp_lim_a_norm',
                # 'episode_bit_bvp_lim_b_norm',
                'event_bit_eeg_signal_min_norm',
                'event_bit_eeg_signal_max_norm',
                'event_bit_eeg_signal_avg_norm',
                'event_bit_eeg_signal_std_norm',
                'event_bit_eeg_features_ts_min_norm',
                'event_bit_eeg_features_ts_max_norm',
                'event_bit_eeg_features_ts_avg_norm',
                'event_bit_eeg_features_ts_std_norm',
                'event_bit_eeg_theta_min_norm',
                'event_bit_eeg_theta_max_norm',
                'event_bit_eeg_theta_avg_norm',
                'event_bit_eeg_theta_std_norm',
                'event_bit_eeg_alpha_low_min_norm',
                'event_bit_eeg_alpha_low_max_norm',
                'event_bit_eeg_alpha_low_avg_norm',
                'event_bit_eeg_alpha_low_std_norm',
                'event_bit_eeg_alpha_high_min_norm',
                'event_bit_eeg_alpha_high_max_norm',
                'event_bit_eeg_alpha_high_avg_norm',
                'event_bit_eeg_alpha_high_std_norm',
                'event_bit_eeg_beta_min_norm',
                'event_bit_eeg_beta_max_norm',
                'event_bit_eeg_beta_avg_norm',
                'event_bit_eeg_beta_std_norm',
                'event_bit_eeg_gamma_min_norm',
                'event_bit_eeg_gamma_max_norm',
                'event_bit_eeg_gamma_avg_norm',
                'event_bit_eeg_gamma_std_norm',
                # 'event_bit_eeg_lim_a_norm',
                # 'event_bit_eeg_lim_b_norm',
            ]

    # face-based features:
    if face:

        if non_norm:
            features += [
                'episode_face_avg_0_anger_proba',
                'episode_face_avg_1_disgust_proba',
                'episode_face_avg_2_fear_proba',
                'episode_face_avg_3_enjoyment_proba',
                'episode_face_avg_4_contempt_proba',
                'episode_face_avg_5_sadness_proba',
                'episode_face_avg_6_surprise_proba',
                'episode_face_avg_most_likely',
                'episode_face_avg_entropy',
                'episode_face_avg_most_freq',
                # 'episode_face_lim_a',
                # 'episode_face_lim_b',
                'event_face_avg_0_anger_proba',
                'event_face_avg_1_disgust_proba',
                'event_face_avg_2_fear_proba',
                'event_face_avg_3_enjoyment_proba',
                'event_face_avg_4_contempt_proba',
                'event_face_avg_5_sadness_proba',
                'event_face_avg_6_surprise_proba',
                'event_face_avg_most_likely',
                'event_face_avg_entropy',
                'event_face_avg_most_freq',
                # 'event_face_lim_a',
                # 'event_face_lim_b',
            ]

        if norm:
            features += [
                'episode_face_avg_0_anger_proba_norm',
                'episode_face_avg_1_disgust_proba_norm',
                'episode_face_avg_2_fear_proba_norm',
                'episode_face_avg_3_enjoyment_proba_norm',
                'episode_face_avg_4_contempt_proba_norm',
                'episode_face_avg_5_sadness_proba_norm',
                'episode_face_avg_6_surprise_proba_norm',
                'episode_face_avg_most_likely_norm',
                'episode_face_avg_entropy_norm',
                'episode_face_avg_most_freq_norm',
                'event_face_avg_0_anger_proba_norm',
                'event_face_avg_1_disgust_proba_norm',
                'event_face_avg_2_fear_proba_norm',
                'event_face_avg_3_enjoyment_proba_norm',
                'event_face_avg_4_contempt_proba_norm',
                'event_face_avg_5_sadness_proba_norm',
                'event_face_avg_6_surprise_proba_norm',
                'event_face_avg_most_likely_norm',
                'event_face_avg_entropy_norm',
                'event_face_avg_most_freq_norm',
            ]


    # sam-based features:
    if sam:
        features += [
            'sam_valence',
            'sam_arousal',
        ]


    return features



if __name__ == '__main__':

    # Define level
    level = sys.argv[1] if len(sys.argv) > 1 else 'task'
    print('Running for level:', level)

    # Get feature flags from command-line arguments
    def str_to_bool(value):
        return value.lower() in {"true", "1", "yes"}

    log = str_to_bool(sys.argv[2]) if len(sys.argv) > 2 else True
    eeg = str_to_bool(sys.argv[3]) if len(sys.argv) > 3 else True
    face = str_to_bool(sys.argv[4]) if len(sys.argv) > 4 else True
    sam = str_to_bool(sys.argv[5]) if len(sys.argv) > 5 else True

    print('Running for level:', level)
    print(f"Features - log: {log}, eeg: {eeg}, face: {face}, sam: {sam}")

    # Set random seed
    random.seed(42)
    np.random.seed(42)

    # load dataset
    df_all = pd.read_pickle('dfs/df_all.pkl')

    # Define features (X) and target (y)
    columns_to_drop = [
        'network',
        'username',
        'task_id',
        'unix_time',
        'time',
        'url',
        'xpath',
        'labels_0',
        'labels_1',
        'labels_2',
        'labels_union',
        'labels_inter',
        'labels_valid',
        'y_binary',
        'y_task_multilabel',
        'y_action_multilabel',
    ]
    columns_to_keep = get_features(log=log, eeg=eeg, face=face, sam=sam, norm=True, non_norm=False)
    networks = ['SN_1', 'SN_2', 'SN_3']
    #networks = []

    # Define number of folds for cross-validation
    n_splits = 5

    # Define hyperparameters for different classifiers
    param_grid = {
        'RandomForest': {
            'clf': [MultiOutputClassifier(RandomForestClassifier(random_state=42, n_jobs=-1))],
            'clf__estimator__n_estimators': [50, 100, 200],
            'clf__estimator__max_depth': [None, 10, 20]  # Corrected
        },
        'LogisticRegression': {
            'clf': [OneVsRestClassifier(LogisticRegression(random_state=42, solver='sag', max_iter=1000, n_jobs=-1))],
            'clf__estimator__C': [0.1, 1.0, 10.0]
        },
        'KNeighbors': {
           'clf': [KNeighborsClassifier(n_jobs=4)],
           'clf__n_neighbors': [3, 5, 7]
        },
        'MLP': {
            'clf': [MLPClassifier(random_state=42, max_iter=1000)],
            'clf__hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__learning_rate': ['constant', 'adaptive']
        },
        'GradientBoosting': {
           'clf': [MultiOutputClassifier(GradientBoostingClassifier(random_state=42))],
           'clf__estimator__n_estimators': [50, 100],
           'clf__estimator__max_depth': [3, 5, 10],
        },
        'LinearSVC': {
            'clf': [OneVsRestClassifier(LinearSVC(random_state=42, max_iter=1000))],
            'clf__estimator__C': [0.1, 1.0, 10.0],
        },
        # 'ExtraTrees': {
        #     'clf': [MultiOutputClassifier(ExtraTreesClassifier(random_state=42, n_jobs=-1))],
        #     'clf__estimator__n_estimators': [50, 100, 200],
        #     'clf__estimator__max_depth': [None, 10, 20],
        # },
        # 'Ridge': {
        #     'clf': [OneVsRestClassifier(RidgeClassifier(random_state=42))],
        #     'clf__estimator__alpha': [0.1, 1.0, 10.0],
        # },
    }
    param_grid = {
        **param_grid,
        'BaselineMajority': {
            'clf': [DummyClassifier(strategy='most_frequent')],
        },
        'BaselineUniform': {
            'clf': [DummyClassifier(strategy='uniform')],
        },
        'BaselineConstant': {
            'clf': [DummyClassifier(strategy='constant', constant=np.zeros(4 if level == 'action' else 7))],
        },
    }


    for network in networks + ['ALL']:
        print(f"Network: {network}")
        
        # Define features (X) and target (y) for task-level
        if level == 'task':
            df = df_all[(df_all['eid'] == 0)].copy()  # task-level
            y = df['y_task_multilabel'] 
        else:
            df = df_all[(df_all['eid'] > 0)].copy()  # action-level
            y = df['y_action_multilabel']

        # Define X
        if len(columns_to_keep) > 0:
            X = df[columns_to_keep]  # Keep only selected columns
        else:
            X = df.drop(columns=columns_to_drop)  # Drop columns

        # convert [0] (none label) to a empty array
        y = y.map(lambda x: [] if len(x) == 1 and x[0] == 0 else x)

        # Convert `y_task_multilabel` to binary matrix using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        y_bin = mlb.fit_transform(y)
        
        # Convert categorical features to numeric codes
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Ensure no NaNs exist in the dataset
        X = X.fillna(0)
        X[['sam_valence', 'sam_arousal']] = X[['sam_valence', 'sam_arousal']].fillna(5)
        
        # Define metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro', zero_division=0),
            'recall': make_scorer(recall_score, average='macro', zero_division=0),
            'f1_micro': make_scorer(f1_score, average='micro', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
            'jaccard': make_scorer(jaccard_score, average='macro'),
            'hamming_score': make_scorer(hamming_score),
        }
        
        # Define cross-validation strategy
        if network != 'ALL': 
            cv = get_predefined_split(df, network=network, n_splits=n_splits)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Perform cross-validation and print dataset shapes
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            print(f"Fold {fold_idx}:")
            print(f"  Train shape: {X.iloc[train_idx].shape}")
            print(f"  Test shape: {X.iloc[test_idx].shape}")
            print_label_distribution(y.iloc[train_idx].to_numpy(), y.iloc[test_idx].to_numpy())

        # Convert to numpy
        X = X.to_numpy()
        
        # Loop through classifiers and hyperparameter grids
        for clf_name, grid in param_grid.items():
            print(f"\nClassifier: {clf_name}")

            current_param_grid = {k.replace('clf__', ''): v for k, v in grid.items() if k != 'clf'}
            print(current_param_grid)

            # Define grid search
            grid_search = GridSearchCV(
                estimator=grid['clf'][0],
                param_grid=current_param_grid,
                scoring=scoring,
                refit='f1_macro',
                cv=cv,
                return_train_score=True,
                n_jobs=-1,
            )

            if clf_name == 'LogisticRegression':
                # Scale data for LogisticRegression or other models that require scaling
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            
            # Fit the model
            grid_search.fit(X, y_bin)

            # Get the best parameters and the corresponding index
            best_params = grid_search.best_params_
            best_index = grid_search.best_index_
            
            # Extract results for the best parameters
            results = grid_search.cv_results_

            # Extract train and test metrics
            metrics = [
                ('Accuracy', 'accuracy'),
                ('Precision', 'precision'),
                ('Recall', 'recall'),
                ('F1 (macro)', 'f1_macro'),
                ('F1 (micro)', 'f1_micro'),
                ('F1 (weighted)', 'f1_weighted'),
                ('Jaccard', 'jaccard'),
                ('Hamming', 'hamming_score'),
            ]
            
            print("=" * 40)
            print(f"Best Parameters: {best_params}")
            print(f"{'Metric':<15} {'Train':<10} {'Test':<10}")
            print("-" * 40)
            
            for metric_name, metric_key in metrics:
                if f'mean_train_{metric_key}' in results.keys():
                    train_score = results[f'mean_train_{metric_key}'][best_index]
                    test_score = results[f'mean_test_{metric_key}'][best_index]
                    print(f"{metric_name:<15} {train_score:.4f}     {test_score:.4f}")
            
            print("=" * 40)
            print("")


