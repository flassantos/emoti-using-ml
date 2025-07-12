import os
import re

import h5py
import numpy as np
import pandas as pd
from scipy.signal import stft

# Mapping of smell labels to integers
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
action_smells_mapping = {
    'none': 0,
    'unnecessary_action': 1,
    'misleading_action': 2,
    'missing_action_feedback': 3,
    'undescriptive_element': 4
}


def str_to_list(x):
    try:
        return eval(x) if isinstance(x, str) else []
    except (ValueError, SyntaxError):
        return []

def unroll(list_of_lists, recursive=False):
    if not recursive:
        return [item for sublist in list_of_lists for item in sublist]
    else:
        while any(isinstance(i, list) for i in list_of_lists):
            list_of_lists = [item for sublist in list_of_lists for item in sublist]
        return list_of_lists

def list_of_labels_to_list_of_ids(x, mapping):
    if len(x) == 0 or (len(x) == 1 and x[0] == 'none'):
        return [mapping['none']]
    if 'none' in x and len(x) > 1:
        x.remove('none')
    return [mapping[s] for s in x]

def read_dataset(dirpath, keep_only_log_cols=True):
    """
    Reads the entire dataset from your folder structure, building a list of DataFrames.
    """
    dfs = []
    for network in os.listdir(dirpath):
        network_path = os.path.join(dirpath, network)
        if not os.path.isdir(network_path):
            continue
        for username in os.listdir(network_path):
            user_path = os.path.join(network_path, username)
            if not os.path.isdir(user_path):
                continue
            fname = os.path.join(user_path, 'data.csv')
            if os.path.exists(fname):
                df = pd.read_csv(fname, sep='\t')
                # Keep only the log-related columns
                if keep_only_log_cols:
                    log_cols = []
                    for x in df.columns:
                        if 'bit_' in x or 'face_' in x:
                            continue
                        log_cols.append(x)
                    df = df[log_cols]
                # create rtid feature
                if 'task_id' in df.columns:
                    df['rtid'] = df['task_id'] / df['task_id'].max() if df['task_id'].max() != 0 else 0
                df['network'] = network
                df['username'] = username
                dfs.append(df)
    if len(dfs) == 0:
        raise ValueError(f"No data.csv files found under {dirpath}")
    return dfs

def generate_labels(df):
    """
    Add y_binary, y_task_multilabel, y_action_multilabel columns to df.
    """
    # Binary label (0 or 1)
    df['y_binary'] = df['labels_union'].apply(
        lambda x: 0 if (str_to_list(x) == ['none'] or len(str_to_list(x)) == 0) else 1
    ).astype(int)

    # Task-level smells
    sel_task = (df['eid'] == 0)
    df.loc[sel_task, 'y_task_multilabel'] = df.loc[sel_task, 'labels_union'].apply(
        lambda x: list_of_labels_to_list_of_ids(str_to_list(x), task_smells_mapping)
    )

    # Action-level smells
    sel_action = (df['eid'] > 0)
    df.loc[sel_action, 'y_action_multilabel'] = df.loc[sel_action, 'labels_union'].apply(
        lambda x: list_of_labels_to_list_of_ids(str_to_list(x), action_smells_mapping)
    )
    return df


def generate_episode_level_global_duration_features(df, col_name):
    """
    E.g. event_duration -> event_duration_max, event_duration_min, etc.
    Groups by [username, network, pid].
    """
    def gen_features(df_g, col):
        max_val = df_g[col].max()
        min_val = df_g[col].min()
        last_val = df_g[col].iloc[-1]
        first_val = df_g[col].iloc[0]
        second_val = df_g[col].iloc[1] if len(df_g[col]) > 1 else 0
        avg_val = df_g[col].mean()

        # We'll create a dictionary of new columns, then assign them all at once
        size = len(df_g)
        df_g = df_g.assign(**{
            f'{col}_max':    [max_val]   * size,
            f'{col}_min':    [min_val]   * size,
            f'{col}_last':   [last_val]  * size,
            f'{col}_first':  [first_val] * size,
            f'{col}_second': [second_val]* size,
            f'{col}_avg':    [avg_val]   * size,
        })
        return df_g

    df = df.groupby(['username', 'network', 'pid']).apply(lambda x: gen_features(x, col_name))
    df = df.reset_index(drop=True)
    return df


def generate_episode_level_local_duration_features(df, col_name='episode_total_duration', k_range=3):
    """
    For 'eid' == 0 only, shift the column to get prev_k and next_k.
    """
    sel = df['eid'] == 0
    for k in range(1, k_range + 1):
        df.loc[sel, f'{col_name}_prev_{k}'] = df.loc[sel, col_name].shift(k)
        df.loc[sel, f'{col_name}_next_{k}'] = df.loc[sel, col_name].shift(-k)
    df = df.fillna(0)
    return df

def get_prepared_dataframe(dirpath, normalize_cols=True, keep_only_log_cols=True, keep_only_norm_cols=True):
    """
    Reads and processes the entire dataset, returning a DataFrame 'df_all' with:
      - All your engineered columns
      - y_binary, y_task_multilabel, y_action_multilabel

    Arguments:
        - dirpath: the path to your /UsabilitySmellsDataset/ folder
        - normalize_cols: if True, we normalize all columns (z-score, min-max) grouped by [network, username]
        - keep_only_log_cols: if True, we only keep the columns that are in the log files
                              (e.g. 'event_duration', 'event_click_pos_x')
        - keep_only_norm_cols: if True, we only keep the normalized columns
    """
    # 1) Read CSV from all subfolders
    dfs = read_dataset(dirpath, keep_only_log_cols=keep_only_log_cols)
    df_all = pd.concat(dfs, ignore_index=True)

    # 2) Generate the label columns
    df_all = generate_labels(df_all)

    # 3) Example: Generate "per min" features
    # (Make sure these columns exist in your data before referencing them)
    safe_div = lambda num, den: num / den if den != 0 else 0
    df_all['episode_tabchange_num_per_min'] = df_all.apply(
        lambda r: safe_div(r['episode_tabchange_num'], (r['episode_total_duration'] / 60)), axis=1
    )
    df_all['episode_change_num_per_min'] = df_all.apply(
        lambda r: safe_div(r['episode_change_num'], (r['episode_total_duration'] / 60)), axis=1
    )
    df_all['episode_click_num_per_min'] = df_all.apply(
        lambda r: safe_div(r['episode_click_num'], (r['episode_total_duration'] / 60)), axis=1
    )
    df_all['episode_scroll_num_per_min'] = df_all.apply(
        lambda r: safe_div(r['episode_scroll_num'], (r['episode_total_duration'] / 60)), axis=1
    )
    df_all['episode_keypress_num_per_min'] = df_all.apply(
        lambda r: safe_div(r['episode_keypress_num'], (r['episode_total_duration'] / 60)), axis=1
    )
    df_all['episode_total_events_per_min'] = df_all.apply(
        lambda r: safe_div(r['episode_total_events'], (r['episode_total_duration'] / 60)), axis=1
    )

    # 4) Generate HTML features
    df_all['xpath_depth'] = df_all['xpath'].apply(lambda x: str(x).count('>'))
    df_all['xpath_div_count'] = df_all['xpath'].apply(lambda x: str(x).count('div'))
    df_all['xpath_num_unique'] = df_all['xpath'].apply(lambda x: len(set(re.findall(r'\b\w+\b', str(x)))))
    # df_all['dom_object_prev'] = df_all['dom_object'].shift(1).fillna('none')
    # df_all['dom_object_next'] = df_all['dom_object'].shift(-1).fillna('none')
    # df_all['event_prev'] = df_all['event'].shift(1).fillna('none')
    # df_all['event_next'] = df_all['event'].shift(-1).fillna('none')

    # 5) Z-score normalization (grouped by [network, username])
    zscore_columns = [
        'event_duration','event_cum_duration','event_scroll_time','event_scroll_num','event_keypress_num',
        'episode_tabchange_duration','episode_change_duration','episode_click_duration','episode_scroll_duration',
        'episode_total_duration','episode_tabchange_num','episode_change_num','episode_click_num','episode_scroll_num',
        'episode_keypress_num','episode_total_events','episode_tabchange_num_per_min','episode_change_num_per_min',
        'episode_click_num_per_min','episode_scroll_num_per_min','episode_keypress_num_per_min',
        'episode_total_events_per_min','xpath_depth','xpath_div_count','xpath_num_unique',
        'event_change_text_length','event_change_value_length','event_click_text_length'
    ]
    # face columns
    face_columns = [c for c in df_all.columns if ('face_avg' in c or 'face_entropy' in c)]
    zscore_columns += face_columns

    # --- Z-score normalization (grouped) ---
    if normalize_cols:
        zscore_newcols = {}
        for col in zscore_columns:
            if col not in df_all.columns:
                continue
            group_mean = df_all.groupby(['network', 'username'])[col].transform('mean')
            group_std = df_all.groupby(['network', 'username'])[col].transform('std').replace(0, 1)
            zscore_newcols[f'{col}_norm'] = (df_all[col] - group_mean) / group_std

        # Assign all these new columns in one shot
        df_all = df_all.assign(**zscore_newcols)

        # Keep only the normalized columns
        if keep_only_norm_cols:
            # drop the previous columns in zscore_columns
            df_all.drop(columns=[c for c in zscore_columns if c in df_all.columns],
                        inplace=True, errors='ignore')

    # 6) Min-max normalization for click coords, EEG/BVP columns
    # --- Min-max normalization (grouped) ---
    if normalize_cols:
        minmax_columns = ['event_click_pos_x', 'event_click_pos_y']
        eeg_bvp_cols = [c for c in df_all.columns if ('bit_eeg' in c or 'bit_bvp' in c)]
        minmax_columns += eeg_bvp_cols

        minmax_newcols = {}
        for col in minmax_columns:
            if col not in df_all.columns:
                continue
            grouped = df_all.groupby(['network', 'username'])[col]
            min_vals = grouped.transform('min')
            max_vals = grouped.transform('max')
            minmax_newcols[f'{col}_norm'] = (df_all[col] - min_vals) / (max_vals - min_vals + 1e-8)

        df_all = df_all.assign(**minmax_newcols)

        # Keep only the normalized columns
        if keep_only_norm_cols:
            # drop the previous columns in minmax_columns
            df_all.drop(columns=[c for c in minmax_columns if c in df_all.columns],
                        inplace=True, errors='ignore')

    # 7) Global and local duration features
    if 'event_duration' in df_all.columns:
        df_all = generate_episode_level_global_duration_features(df_all, 'event_duration')
        df_all = generate_episode_level_local_duration_features(df_all, 'episode_total_duration')
        df_all = generate_episode_level_local_duration_features(df_all, 'episode_total_events')

    if normalize_cols:
        df_all = generate_episode_level_global_duration_features(df_all, 'event_duration_norm')
        df_all = generate_episode_level_local_duration_features(df_all, 'episode_total_duration_norm')
        df_all = generate_episode_level_local_duration_features(df_all, 'episode_total_events_norm')

        if keep_only_norm_cols and 'event_duration' in df_all.columns:
            df_all.drop(columns=[c for c in df_all.columns if 'event_duration' in c and 'norm' not in c],
                        inplace=True, errors='ignore')
            df_all.drop(columns=[c for c in df_all.columns if 'episode_total_duration' in c and 'norm' not in c],
                        inplace=True, errors='ignore')
            df_all.drop(columns=[c for c in df_all.columns if 'episode_total_events' in c and 'norm' not in c],
                        inplace=True, errors='ignore')

    # 8) Transform the 'dom_object' and 'event' columns to ints
    # events = ['click', 'change', 'tabchange', 'scroll', 'keypress', 'none']
    # dom_objects = ['input', 'button', 'div', 'p', 'h1', 'img', 'a', 'li', 'span', 'textarea', 'body', 'ul',
    #                'select', 'form', 'i', 'label', 'none']
    # df_all['event'] = df_all['event'].apply(lambda x: events.index(x) if x in events else -1)
    # df_all['event_prev'] = df_all['event_prev'].apply(lambda x: events.index(x) if x in events else -1)
    # df_all['event_next'] = df_all['event_next'].apply(lambda x: events.index(x) if x in events else -1)
    # df_all['dom_object'] = df_all['dom_object'].apply(lambda x: dom_objects.index(x) if x in dom_objects else -1)
    # df_all['dom_object_prev'] = df_all['dom_object_prev'].apply(lambda x: dom_objects.index(x) if x in dom_objects else -1)
    # df_all['dom_object_next'] = df_all['dom_object_next'].apply(lambda x: dom_objects.index(x) if x in dom_objects else -1)

    # 9) SAM columns
    if 'sam_valence' in df_all.columns:
        df_all['sam_valence_norm'] = (df_all['sam_valence'] - 1) / 8
    if 'sam_arousal' in df_all.columns:
        df_all['sam_arousal_norm'] = (df_all['sam_arousal'] - 1) / 8

    # Convert 'event' column to one-hot encoding
    df_all = pd.get_dummies(df_all, columns=['event'], prefix='event', dtype=int)

    # Convert 'dom_object' column to target encoding
    from sklearn.preprocessing import TargetEncoder
    target_encoder = TargetEncoder(smooth='auto', target_type='binary')
    x = df_all['dom_object'].to_numpy()[:, None]
    y = df_all['y_binary'].to_numpy()
    df_all['dom_object'] = target_encoder.fit_transform(x, y).squeeze(-1)

    # Convert 'finished_task' to float
    df_all['finished_task'] = df_all['finished_task'].astype(float)

    return df_all


def load_eeg_bvp(base_dir, network, username, level=None):
    """
    Attempt to open bit.h5 and extract the EEG/BVP timeseries for (pid, eid).
    This function is a placeholder; adapt to your .h5 layout.
    """
    bit_path = os.path.join(base_dir, network, username, "bit.h5")
    if not os.path.exists(bit_path):
        return None
    try:
        loaded_data = []
        with h5py.File(bit_path, 'r') as file:
            for key in file.keys():
                group = file[key]
                item = {
                    'interval': group.attrs['interval'],
                    'level': group.attrs['level'],  # 'tabchange' if self.level=='task' else 'event'
                    'eeg_signal': np.array(group['eeg_signal']),
                    'bvp_signal': np.array(group['bvp_signal'])
                }
                if level == 'task' and item['level'] == 'tabchange':
                    loaded_data.append(item)
                elif level == 'action' and item['level'] == 'event':
                    loaded_data.append(item)
                elif level is None:
                    loaded_data.append(item)
        return loaded_data
    except Exception as e:
        print(f"Error reading {bit_path}: {e}")
        return None


def load_face(base_dir, network, username, level=None):
    """
    Attempt to open face.h5 and extract the face emotion probabilities for (pid, eid).
    This function is also a placeholder; adapt to your .h5 layout.
    """
    face_path = os.path.join(base_dir, network, username, "face.h5")
    if not os.path.exists(face_path):
        return None
    try:
        loaded_data = []
        with h5py.File(face_path, 'r') as file:
            for key in file.keys():
                group = file[key]
                item = {
                    'interval': group.attrs['interval'],
                    'level': group.attrs['level'],  # 'tabchange' if self.level=='task' else 'event'
                    'emotions': np.array(group['emotions']),
                }
                if level == 'task' and item['level'] == 'tabchange':
                    loaded_data.append(item)
                elif level == 'action' and item['level'] == 'event':
                    loaded_data.append(item)
                elif level is None:
                    loaded_data.append(item)
        return loaded_data
    except Exception as e:
        print(f"Error reading {face_path}: {e}")
        return None


def get_eeg_data(dirpath, level='task'):
    """
    Load EEG/BVP data for all networks and usernames
    """
    eeg_bvp_dict = {}
    for network in os.listdir(dirpath):
        network_path = os.path.join(dirpath, network)
        if not os.path.isdir(network_path):
            continue
        for username in os.listdir(network_path):
            user_path = os.path.join(network_path, username)
            if not os.path.isdir(user_path):
                continue
            eeg_bvp_data = load_eeg_bvp(dirpath, network, username, level=level)
            if eeg_bvp_data is not None:
                if network not in eeg_bvp_dict:
                    eeg_bvp_dict[network] = {}
                eeg_bvp_dict[network][username] = eeg_bvp_data
    return eeg_bvp_dict


def get_face_data(dirpath, level='task'):
    """
    Load face data for all networks and usernames
    """
    face_dict = {}
    for network in os.listdir(dirpath):
        network_path = os.path.join(dirpath, network)
        if not os.path.isdir(network_path):
            continue
        for username in os.listdir(network_path):
            user_path = os.path.join(network_path, username)
            if not os.path.isdir(user_path):
                continue
            face_data = load_face(dirpath, network, username, level=level)
            if face_data is not None:
                if network not in face_dict:
                    face_dict[network] = {}
                face_dict[network][username] = face_data
    return face_dict


def get_standard_csv_cols(log=True, sam=True, norm=True, non_norm=False):
    cols = []

    if norm is False and non_norm is False:
        raise Exception("Either norm or non normalized features should be selected.")

    # log-based features:
    if log:
        cols += [
            # 'pid',
            # 'eid',
            'rid',
            'srid',
            'rtid',

            # 'event',
            'event_click',
            'event_change',
            'event_tabchange',
            'event_scroll',
            'event_keypress',

            'dom_object',
            'finished_task',
        ]

        if non_norm:
            cols += [
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
            ]

        if norm:
            cols += [
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

                # 'event_click_text',
                # 'event_click_button',
                # 'event_change_text',
                # 'event_change_value',
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

                'event_duration_norm_max',
                'event_duration_norm_min',
                'event_duration_norm_last',
                'event_duration_norm_first',
                'event_duration_norm_second',
                'event_duration_norm_avg',
            ]

    # sam-based features:
    if sam:
        if norm:
            cols += [
                'sam_valence_norm',
                'sam_arousal_norm',
            ]
        else:
            cols += [
                'sam_valence',
                'sam_arousal',
            ]

    return cols
