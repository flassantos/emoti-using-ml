from collections import Counter

import torch
import numpy as np
from torch.utils.data import Dataset

from dataset.eeg_features import get_eeg_features, compute_eeg_bvp_spectrograms, extract_combined_eeg_features, \
    compute_emotion_features_from_eeg
from dataset.face_features import chunk_face_emotions_stacked, extract_face_features
from dataset.utils import get_standard_csv_cols, unroll


class UsabilitySmellsSubset(Dataset):
    """
    A thin wrapper that references a parent UsabilitySmellsDataset
    but only exposes a subset of its indices.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]

    def get_label_distribution(self):
        """
        Returns the distribution of labels in this subset.
        """
        c = Counter()
        for idx in self.indices:
            labels = self.dataset[idx]['label']
            for i in range(len(labels)):
                if len(labels.shape) == 1 or labels.shape[-1] <= 1:
                    y = labels[i].squeeze().item()
                else:
                    y = tuple(torch.nonzero(labels[i]).squeeze(-1).tolist())
                c[y] += 1
        total = sum(c.values())
        dist = {k: v / total for k, v in c.items()}
        return dist, total

    def get_feature_distribution(self):
        """
        Returns the distribution of features in this subset.
        For each feature, return a tuple with (mean, std)
        """
        features = {}
        for idx in self.indices:
            log = self.dataset[idx]['log']
            columns = self.dataset[idx]['columns']
            for i, col in enumerate(columns):
                if col not in features:
                    features[col] = [0, 0]
                features[col][0] += log[:, i].mean().item()
                features[col][1] += log[:, i].std().item()
        dist = {}
        for k, v in features.items():
            dist[k] = (v[0] / len(self.indices), v[1] / len(self.indices))
        return dist

    def get_input_lengths_distribution(self):
        """
        Returns the distribution of input lengths in this subset.
        Returns, min, mean, std, max
        """
        lengths = []
        for idx in self.indices:
            lengths.append(len(self.dataset[idx]['log']))
        return min(lengths), np.mean(lengths), np.std(lengths), max(lengths)

class UsabilitySmellsDataset(Dataset):
    """
    A multi-modal dataset for (task-level or action-level) classification,
    supporting both binary or multi-label setups, with memory caching for faster access.
    """
    def __init__(
        self,
        log_df,
        eeg_bvp_data,
        face_data,
        label_type='binary',  # 'binary' or 'multilabel'
        level='task',         # 'task' (eid=0) or 'action' (eid>0)
        csv_feature_cols=None,
        mlb=None,
        force_align=False,
        full_history=False,
        use_spectrograms=False,
        eeg_bvp_kwargs=None,
        face_kwargs=None,
        inner_cat=False,
        preload_cache=True,   # Enable caching by default
        num_workers=4,        # Number of workers for parallel processing
    ):
        """
        log_df: a list (or DataFrame) of records from df_all
        eeg_bvp_data: a dict of EEG/BVP data (from bit.h5)
        face_data: a dict of face data (from face.h5)
        label_type: 'binary' -> use y_binary, 'multilabel' -> use y_task_multilabel / y_action_multilabel
        level: 'task' => interpret y_task_multilabel, 'action' => y_action_multilabel
        csv_feature_cols: list of columns for numeric CSV features. If None, auto-select
        mlb: a MultiLabelBinarizer (required if label_type='multilabel')
        force_align: if True, align the EEG/BVP/Face data with the log data
        full_history: if True, use all events (not just the selected ones according to `level`).
                      include also a bool variable indicating whether the eid==0 (tabchange).
        use_spectrograms: if True, use EEG waves instead of spectrograms
        eeg_bvp_kwargs: additional kwargs for compute_eeg_bvp_spectrograms
        face_kwargs: additional kwargs for chunk_face_emotions_stacked
        inner_cat: if True, concatenate EEG/BVP features from different intervals
        preload_cache: if True, precompute and cache all data during initialization
        num_workers: number of workers for parallel processing
        """
        super().__init__()
        # transform log_df to a dict for faster access, where the key is the (network, username)
        self.log_data = {}
        self.idx_to_network_username = {}
        log_df['network_username'] = log_df['network'] + '-' + log_df['username']
        for i, net_user in enumerate(log_df['network_username'].unique().tolist()):
            net, user = net_user.split('-')
            self.log_data[(net, user)] = log_df[log_df['network_username'] == net_user].copy().reset_index(drop=True)
        self.eeg_bvp_data = eeg_bvp_data
        self.face_data = face_data
        self.idx_to_network_username = {i: k for i, k in enumerate(self.log_data.keys())}
        self.network_usernames_to_idx = {v: k for k, v in self.idx_to_network_username.items()}

        # remove (network, username) pairs that do not have EEG/BVP/Face data
        keep_only_with_eeg_or_face = False
        if keep_only_with_eeg_or_face:
            to_remove = []
            for net_user in self.log_data.keys():
                network, username = net_user
                has_eeg = network in self.eeg_bvp_data and username in self.eeg_bvp_data[network]
                has_face = network in self.face_data and username in self.face_data[network]
                if not has_eeg or not has_face:
                    to_remove.append((net_user, self.network_usernames_to_idx[net_user]))
            for net_user, idx in sorted(to_remove, key=lambda x: x[1], reverse=True):
                del self.log_data[net_user]
            self.idx_to_network_username = {i: k for i, k in enumerate(self.log_data.keys())}
            self.network_usernames_to_idx = {v: k for k, v in self.idx_to_network_username.items()}

        self.label_type = label_type
        self.level = level
        self.csv_feature_cols = csv_feature_cols
        if self.csv_feature_cols is None:
            self.csv_feature_cols = get_standard_csv_cols()
        self.mlb = mlb
        self.force_align = force_align
        self.full_history = full_history
        if self.full_history:
            # add a column to indicate whether the event is a tabchange
            self.csv_feature_cols.append('is_tabchange')
        self.use_spectrograms = use_spectrograms
        self.inner_cat = inner_cat
        self.eeg_bvp_kwargs = dict(
            fs_eeg=1000,
            fs_bvp=1000,
            window='hann',
            nperseg_eeg=500,
            noverlap_eeg=250,
            nperseg_bvp=500,
            noverlap_bvp=250,
            return_db=True  # consider log-scale
        )
        if eeg_bvp_kwargs:
            self.eeg_bvp_kwargs.update(eeg_bvp_kwargs)
        self.face_kwargs = dict(
            window_size=1000,
            overlap=250
        )
        if face_kwargs:
            self.face_kwargs.update(face_kwargs)
        if label_type == 'multilabel':
            assert self.mlb is not None, "MultiLabelBinarizer must be provided for multi-label"

        # Initialize cache
        self.cache = {}
        self.num_workers = num_workers

        # Preload all data into cache if requested
        if preload_cache:
            self._build_cache()

    def _build_cache(self):
        """
        Precompute and cache all data items
        """
        print(f"Building cache for {len(self)} users...")
        for idx in range(len(self)):
            network, username = self.idx_to_network_username[idx]
            print(f"Processing user {idx + 1}/{len(self)}: {network} {username}")
            self.cache[idx] = self._process_single_item(idx)
        print(f"Cache built. {len(self.cache)} items cached.")

    @property
    def csv_dim(self):
        # this user has both eeg and face data
        hardcoded_idx = self.network_usernames_to_idx[('SN_2', 'L04')]
        return self[hardcoded_idx]['log'].shape[-1]

    @property
    def eeg_dim(self):
        # this user has both eeg and face data
        hardcoded_idx = self.network_usernames_to_idx[('SN_2', 'L04')]
        return self[hardcoded_idx]['eeg'][0].shape[-1]

    @property
    def bvp_dim(self):
        # this user has both eeg and face data
        hardcoded_idx = self.network_usernames_to_idx[('SN_2', 'L04')]
        return self[hardcoded_idx]['bvp'][0].shape[-1]

    @property
    def face_dim(self):
        # this user has both eeg and face data
        hardcoded_idx = self.network_usernames_to_idx[('SN_2', 'L04')]
        return self[hardcoded_idx]['face'][0].shape[-1]

    @property
    def num_labels(self):
        if self.mlb is None:
            return 1
        return self[0]['label'].shape[-1]

    def __len__(self):
        return len(self.log_data)

    def __getitem__(self, idx):
        """
        Get a cached item if available, otherwise process and cache it
        """
        if idx in self.cache:
            return self.cache[idx]

        # Process the item if not in cache
        item = self._process_single_item(idx)

        # Cache the item for future use
        self.cache[idx] = item

        return item

    def _process_single_item(self, idx):
        # Recover network and username from the index
        network, username = self.idx_to_network_username[idx]

        # Get the log data for this user
        log_df = self.log_data[(network, username)]


        if not self.full_history:
            # Filter out the rows that are not aligned with EEG/BVP/Face data
            log_mask = log_df['eid'] == 0 if self.level == 'task' else log_df['eid'] > 0
            log_df = log_df[log_mask].copy().reset_index(drop=True)
            # Save the mask for tabchange events
            level_mask = torch.tensor((log_df['eid'] == 0).astype(int).tolist()).long()
            # Get the indices that are aligned with EEG/BVP/Face data
            align_indices = [i for i, x in enumerate(log_mask.tolist()) if x]
        else:
            assert self.force_align is False, "Cannot force align when using full history"
            # add a column to indicate whether the event is a tabchange
            log_df['is_tabchange'] = (log_df['eid'] == 0).astype(float)
            log_df = log_df.copy().reset_index(drop=True)
            # save the mask for tabchange events
            level_mask = torch.tensor((log_df['eid'] == 0).astype(int).tolist()).long()
            # all rows are aligned
            align_indices = list(range(len(log_df)))

        # Extract the pid and eid for debugging
        pid = log_df['pid'].tolist()
        eid = log_df['eid'].tolist()

        # ------------------------------------------------------------------
        # 1) Extract CSV-based features
        # ------------------------------------------------------------------
        log_tensor, log_columns = self._build_log_tensor(log_df)  # shape of [N, num_csv_features]

        # ------------------------------------------------------------------
        # 2) Extract label
        # ------------------------------------------------------------------
        label_tensor = self._build_label_tensor(log_df)  # shape of [N, 1] or [N, K]

        # ------------------------------------------------------------------
        # 3) Load EEG/BVP data from bit.h5 if it exists
        # ------------------------------------------------------------------
        # eeg_bvp_data[network][username] is a list of T items,
        # item['eeg_signal'] has shape [L,] where L is the length of the EEG signal for that interval
        # the item is the concatenation of EEG and BVP signals from successive intervals
        # ps: L is different for each item
        if not self.use_spectrograms:
            eeg_features = None
            bvp_features = None
            if network in self.eeg_bvp_data and username in self.eeg_bvp_data[network]:
                eeg_features, bvp_features = [], []
                eeg_signals, bvp_signals = [], []
                indices = align_indices if self.force_align else range(len(self.eeg_bvp_data[network][username]))
                for i in indices:
                    # corner case: the EEG/BVP data is shorter than the log data
                    if i >= len(self.eeg_bvp_data[network][username]):
                        continue
                    item = self.eeg_bvp_data[network][username][i]
                    if self.inner_cat:
                        try:
                            e = eeg_signals + np.array(item['eeg_signal']).tolist()
                            eeg_features_, _ = compute_emotion_features_from_eeg(
                                np.array(e),
                                sampling_rate=1000.0,
                                size=1.0,
                                overlap=0.5,
                                z_normalize=True,
                                n_jobs=8
                            )
                            eeg_features.append(eeg_features_)
                            eeg_signals = []

                            b = bvp_signals + np.array(item['bvp_signal']).tolist()
                            bvp_features_, _ = compute_emotion_features_from_eeg(
                                np.array(b),
                                sampling_rate=1000.0,
                                size=1.0,
                                overlap=0.5,
                                z_normalize=True,
                                n_jobs=8
                            )
                            bvp_features.append(bvp_features_)
                            bvp_signals = []
                        except Exception as e:
                            eeg_signals.extend(np.array(item['eeg_signal']).tolist())
                            bvp_signals.extend(np.array(item['bvp_signal']).tolist())
                            # print(f"Error. Skipping eeg {network}-{username} {i} ({len(item['eeg_signal'])})...")
                    else:
                        eeg_signals.append(item['eeg_signal'])
                        bvp_signals.append(item['bvp_signal'])
                if self.inner_cat:
                    eeg_features = np.array(unroll(eeg_features))
                    bvp_features = np.array(unroll(bvp_features))
                else:
                    eeg_signals = np.array(unroll(eeg_signals))
                    bvp_signals = np.array(unroll(bvp_signals))
                    eeg_features, _ = compute_emotion_features_from_eeg(
                        eeg_signals,
                        sampling_rate=1000.0,
                        size=0.5,
                        overlap=0.25,
                        z_normalize=True,
                        n_jobs=8
                    )
                    bvp_features, _ = compute_emotion_features_from_eeg(
                        bvp_signals,
                        sampling_rate=1000.0,
                        size=0.5,
                        overlap=0.25,
                        z_normalize=True,
                        n_jobs=8
                    )
        else:
            eeg_features = None
            bvp_features = None
            if network in self.eeg_bvp_data and username in self.eeg_bvp_data[network]:
                eeg_features, bvp_features = [], []
                indices = align_indices if self.force_align else range(len(self.eeg_bvp_data[network][username]))
                for i in indices:
                    # corner case: the EEG/BVP data is shorter than the log data
                    if i >= len(self.eeg_bvp_data[network][username]):
                        continue
                    item = self.eeg_bvp_data[network][username][i]
                    (eeg_f, eeg_t, eeg_spectro), (bvp_f, bvp_t, bvp_spectro) = compute_eeg_bvp_spectrograms(
                        item['eeg_signal'],
                        item['bvp_signal'],
                        **self.eeg_bvp_kwargs
                    )
                    eeg_features.append(eeg_spectro)  # shape [freq_eeg, time_eeg]
                    bvp_features.append(bvp_spectro)  # shape [freq_bvp, time_bvp]

        # ------------------------------------------------------------------
        # 4) Load face data from face.h5 if it exists
        # ------------------------------------------------------------------
        if not self.use_spectrograms:
            face_features = None
            if network in self.face_data and username in self.face_data[network]:
                face_features = []
                emotions = []
                indices = align_indices if self.force_align else range(len(self.face_data[network][username]))
                for i in indices:
                    # corner case: the face data is shorter than the log data
                    if i >= len(self.face_data[network][username]):
                        continue
                    item = self.face_data[network][username][i]
                    if self.inner_cat:
                        try:
                            e = emotions + np.array(item['emotions']).tolist()
                            face_features_, _, _ = extract_face_features(
                                np.array(e),
                                emotions_names=['anger', 'disgust', 'fear', 'enjoyment', 'contempt', 'sadness', 'surprise'],
                                sampling_rate=1000,
                                window_size=1.0,
                                overlap=0.5,
                                normalize='zscore',
                                n_jobs=8
                            )
                            emotions = []
                            face_features.append(face_features_)
                        except Exception as e:
                            emotions.extend(np.array(item['emotions']).tolist())
                            # print(f"Error. Skipping face {network}-{username} {i} ({len(item['emotions'])})...")
                    else:
                        emotions.append(item['emotions'])

                if self.inner_cat:
                    face_features = np.array(unroll(face_features))
                else:
                    emotions = np.array(unroll(emotions))
                    face_features, _, _ = extract_face_features(
                        emotions,
                        emotions_names=['anger', 'disgust', 'fear', 'enjoyment', 'contempt', 'sadness', 'surprise'],
                        sampling_rate=1000,
                        window_size=0.5,
                        overlap=0.25,
                        normalize='zscore',
                        n_jobs=8
                    )
        else:
            face_features = None
            if network in self.face_data and username in self.face_data[network]:
                face_features = []
                indices = align_indices if self.force_align else range(len(self.face_data[network][username]))
                for i in indices:
                    # corner case: the face data is shorter than the log data
                    if i >= len(self.face_data[network][username]):
                        continue
                    item = self.face_data[network][username][i]
                    chunked_feats = chunk_face_emotions_stacked(
                        item['emotions'],
                        **self.face_kwargs
                    )  # shape [chunks, 16]
                    face_features.append(chunked_feats)

        return {
            'label': label_tensor,  # shape [N, 1] or [N, K]
            'log': log_tensor,      # shape [N, num_csv_features]
            'eeg': eeg_features,    # list of length N, each shape (freq_i, time_i)
            'bvp': bvp_features,    # list of length N, each shape (freq_i, time_i)
            'face': face_features,  # list of length N, each shape [chunks, 16]
            'network': network,     # str, for debugging
            'username': username,   # str, for debugging
            'pid': pid,             # list of length N, for debugging
            'eid': eid,             # list of length N, for debugging
            'columns': log_columns,
            'level_mask': level_mask,
        }

    def _build_log_tensor(self, df):
        """
        Creates a torch.Tensor of numeric features from `df` either:
         - from `csv_feature_cols`, if provided, OR
         - automatically from either _norm or non-_norm columns, depending on self.use_norm_cols
        """
        if self.csv_feature_cols is not None:
            # If user explicitly provided columns, we just use them as-is
            good_cols = []
            for col in self.csv_feature_cols:
                if col not in df.columns:
                    # print(f"Warning: {col} not found in row")
                    continue
                if not any([dtype in str(df[col].dtype) for dtype in ('int', 'float', 'bool')]):
                    # print(f"Warning: {col} is not numeric, dropping")
                    continue
                good_cols.append(col)
            csv_features = df[good_cols].astype(float).values

        else:
            # Automatic approach: gather numeric columns that are not label columns,
            ignore_cols = {
                'network', 'username', 'pid', 'eid', 'task_id',
                'unix_time', 'time', 'url', 'xpath',
                'y_binary', 'y_task_multilabel', 'y_action_multilabel',
                'labels_0', 'labels_1', 'labels_2', 'labels_union', 'labels_inter', 'labels_valid',
                'event_click_text', 'event_click_button', 'event_change_text', 'event_change_value',
            }
            good_cols = []
            for col in df.columns.tolist():
                if col in ignore_cols:
                    # print(f"Warning: {col} is being ignored")
                    continue
                if not any([dtype in str(df[col].dtype) for dtype in ('int', 'float', 'bool')]):
                    # print(f"Warning: {col} is not numeric, dropping")
                    continue
                good_cols.append(col)
            csv_features = df[good_cols].values
        # Convert to float32
        csv_features = np.array(csv_features, dtype=np.float32)
        return torch.tensor(csv_features, dtype=torch.float32), good_cols

    def _build_label_tensor(self, df):
        """
        Converts the df's label into a torch.Tensor, either binary or multi-label.
        """
        if self.label_type == 'binary':
            label_arr = df['y_binary'].tolist()  # 0 or 1
        else:
            col_name = 'y_task_multilabel' if self.level == 'task' else 'y_action_multilabel'
            label_list = df[col_name].map(
                lambda y: [] if not isinstance(y, (list, tuple)) or (len(y) == 1 and y[0] == 0) else y
            ).tolist()
            label_arr = self.mlb.transform(label_list)
        return torch.tensor(label_arr, dtype=torch.long)


def resize_spectrogram(spectro, target_freq=128, target_time=128):
    # spectro: shape [freq, time]
    # we can treat it like [1,1,freq,time], then call interpolate or adaptive avg pool
    spectro_4d = spectro.unsqueeze(0).unsqueeze(0)  # => [1,1,freq,time]
    # option 1: use F.adaptive_avg_pool2d
    # => shape [1,1,target_freq,target_time]
    out_4d = torch.nn.functional.adaptive_avg_pool2d(spectro_4d, (target_freq, target_time))
    return out_4d.squeeze(0).squeeze(0)  # => [target_freq, target_time]


def hierarchical_collate_fn(batch, use_spectrograms=False):
    """
    Expects each item in `batch` is a dict from UsabilitySmellsDataset.__getitem__:
      {
        'label': shape [N, 1 or K],
        'log': shape [N, csv_dim],
        'eeg': list of length N, each shape [freq_eeg_i, time_eeg_i],
        'bvp': list of length N, each shape [freq_bvp_i, time_bvp_i],
        'face': list of length N, each shape [chunks_i, face_dim],
        'network': str,
        'username': str,
        'pid': list length N,
        'eid': list length N
      }

    This collate:
      - Summarizes #intervals per sample => 'lengths'
      - Aggregates all intervals into a single "mega-batch" of intervals => shape [total_intervals, ...]
      - Zero-pads EEG/BVP spectrograms to [max_freq, max_time]
      - Zero-pads face arrays to [max_face_len, face_dim]
      - Stacks CSV and labels
      - Returns:
        {
          'batch_eeg': [total_intervals, max_fe, max_te],
          'batch_bvp': [total_intervals, max_fb, max_tb],
          'batch_face': [total_intervals, max_face_len, face_dim],
          'batch_csv': [total_intervals, csv_dim],
          'batch_labels': [total_intervals, num_labels],
          'owner_idx': list of length total_intervals,
          'lengths': list of length B (each #intervals for that sample),
        }
    """
    B = len(batch)  # number of samples in this batch

    if B == 0:
        # e.g. maybe an empty batch or no intervals
        return {
            'batch_eeg': torch.zeros(0),
            'batch_bvp': torch.zeros(0),
            'batch_face': torch.zeros(0),
            'batch_csv': torch.zeros(0),
            'batch_labels': torch.zeros(0),
            'owner_idx': [],
            'lengths': [],
        }

    max_t_log = max([len(b['log']) for b in batch])
    max_t_labels = max([ len(b['label']) for b in batch])
    max_t_eeg = max([len(b['eeg']) if b['eeg'] is not None else 0 for b in batch])
    max_t_bcp = max([len(b['bvp']) if b['bvp'] is not None else 0 for b in batch])
    max_t_face = max([len(b['face']) if b['face'] is not None else 0 for b in batch])

    # Zero-pad the log data
    batch_log = torch.zeros(B, max_t_log, batch[0]['log'].shape[-1])
    lengths = []
    for i, b in enumerate(batch):
        lengths.append(len(b['log']))
        batch_log[i, :len(b['log'])] = torch.tensor(b['log']) if not isinstance(b['log'], torch.Tensor) else b['log'].clone()
    lengths = torch.tensor(lengths, dtype=torch.long)

    # Pad the labels with -100
    ignore_index = -100
    if len(batch[0]['label'].shape) > 1:
        batch_labels = torch.full((B, max_t_labels, batch[0]['label'].shape[-1]), ignore_index, dtype=torch.int)
    else:
        batch_labels = torch.full((B, max_t_labels), ignore_index, dtype=torch.int)
    for i, b in enumerate(batch):
        batch_labels[i, :len(b['label'])] = torch.tensor(b['label']) if not isinstance(b['label'], torch.Tensor) else b['label'].clone()

    if not use_spectrograms:
        max_t_eeg = max([b['eeg'].shape[0] if b['eeg'] is not None else 0 for b in batch])
        feats_eeg = max([b['eeg'].shape[1] if b['eeg'] is not None else 0 for b in batch])
        batch_eeg = torch.zeros(B, max_t_eeg, feats_eeg)
        mask_eeg = torch.zeros(B, dtype=torch.bool)
        lenghts_eeg = []
        for i, b in enumerate(batch):
            if b['eeg'] is None:
                lenghts_eeg.append(0)
                continue
            lenghts_eeg.append(len(b['eeg']))
            mask_eeg[i] = True
            batch_eeg[i, :len(b['eeg'])] = torch.tensor(b['eeg'])
        lenghts_eeg = torch.tensor(lenghts_eeg, dtype=torch.long)

        max_t_bvp = max([b['bvp'].shape[0] if b['bvp'] is not None else 0 for b in batch])
        feats_bvp = max([b['bvp'].shape[1] if b['bvp'] is not None else 0 for b in batch])
        batch_bvp = torch.zeros(B, max_t_bvp, feats_bvp)
        mask_bvp = torch.zeros(B, dtype=torch.bool)
        lenghts_bvp = []
        for i, b in enumerate(batch):
            if b['bvp'] is None:
                lenghts_bvp.append(0)
                continue
            lenghts_bvp.append(len(b['bvp']))
            mask_bvp[i] = True
            batch_bvp[i, :len(b['bvp'])] = torch.tensor(b['bvp'])
        lenghts_bvp = torch.tensor(lenghts_bvp, dtype=torch.long)

        max_t_face = max([b['face'].shape[0] if b['face'] is not None else 0 for b in batch])
        feats_face = max([b['face'].shape[1] if b['face'] is not None else 0 for b in batch])
        batch_face = torch.zeros(B, max_t_face, feats_face)
        mask_face = torch.zeros(B, dtype=torch.bool)
        lenghts_face = []
        for i, b in enumerate(batch):
            if b['face'] is None:
                lenghts_face.append(0)
                continue
            lenghts_face.append(len(b['face']))
            mask_face[i] = True
            batch_face[i, :len(b['face'])] = torch.tensor(b['face'])
        lenghts_face = torch.tensor(lenghts_face, dtype=torch.long)

    else:
        # Zero-pad the EEG spectrograms
        max_freq_eeg = 128
        max_time_eeg = 128
        batch_eeg = torch.zeros(B, max_t_eeg, max_freq_eeg, max_time_eeg)
        mask_eeg = torch.zeros(B, dtype=torch.bool)
        lenghts_eeg = []
        for i, b in enumerate(batch):
            if b['eeg'] is None:
                lenghts_eeg.append(0)
                continue
            lenghts_eeg.append(len(b['eeg']))
            mask_eeg[i] = True
            for j, sxx in enumerate(b['eeg']):
                sxx = torch.tensor(sxx)
                if sxx.shape[0] == 0 or sxx.shape[1] == 0:
                    continue
                batch_eeg[i, j] = resize_spectrogram(sxx, target_freq=max_freq_eeg, target_time=max_time_eeg)
        lenghts_eeg = torch.tensor(lenghts_eeg, dtype=torch.long)

        # Zero-pad the BVP spectrograms
        max_freq_bvp = 128
        max_time_bvp = 128
        batch_bvp = torch.zeros(B, max_t_bcp, max_freq_bvp, max_time_bvp)
        mask_bvp = torch.zeros(B, dtype=torch.bool)
        lenghts_bvp = []
        for i, b in enumerate(batch):
            if b['bvp'] is None:
                lenghts_bvp.append(0)
                continue
            lenghts_bvp.append(len(b['bvp']))
            mask_bvp[i] = True
            for j, sxx in enumerate(b['bvp']):
                sxx = torch.tensor(sxx)
                if sxx.shape[0] == 0 or sxx.shape[1] == 0:
                    continue
                batch_bvp[i, j] = resize_spectrogram(sxx, target_freq=max_freq_bvp, target_time=max_time_bvp)
        lenghts_bvp = torch.tensor(lenghts_bvp, dtype=torch.long)

        # Zero-pad the face data
        max_face_chunks = 128
        face_dim = 16  # hard-coded
        batch_face = torch.zeros(B, max_t_face, max_face_chunks, face_dim)
        mask_face = torch.zeros(B, dtype=torch.bool)
        lenghts_face = []
        for i, b in enumerate(batch):
            if b['face'] is None:
                lenghts_face.append(0)
                continue
            lenghts_face.append(len(b['face']))
            mask_face[i] = True
            for j, face_feats in enumerate(b['face']):
                sxx = torch.tensor(face_feats)
                if sxx.shape[0] == 0 or sxx.shape[1] == 0:
                    continue
                batch_face[i, j] = resize_spectrogram(sxx, target_freq=max_face_chunks, target_time=face_dim)
        lenghts_face = torch.tensor(lenghts_face, dtype=torch.long)


    # pad level_mask with -1
    max_t_level_mask = max([len(b['level_mask']) if b['level_mask'] is not None else 0 for b in batch])
    batch_level_mask = torch.full((B, max_t_level_mask), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        if b['level_mask'] is not None:
            batch_level_mask[i, :len(b['level_mask'])] = b['level_mask']

    return {
        'batch_log': batch_log,        # [B, max_t_log, csv_dim]
        'batch_labels': batch_labels,  # [B, max_t_labels, num_labels]
        'batch_eeg': batch_eeg,        # [B, max_t_eeg, max_freq_eeg, max_time_eeg]
        'batch_bvp': batch_bvp,        # [B, max_t_bcp, max_freq_bvp, max_time_bvp]
        'batch_face': batch_face,      # [B, max_t_face, max_face_chunks, face_dim]
        'batch_level_mask': batch_level_mask,  # [B, max_t_level_mask]
        'lengths': lengths,            # [B], each is number of intervals for that sample
        'lengths_eeg': lenghts_eeg,    # [B], each is number of EEG intervals for that sample
        'lengths_bvp': lenghts_bvp,    # [B], each is number of BVP intervals for that sample
        'lengths_face': lenghts_face,  # [B], each is number of face intervals for that sample
        'mask_eeg': mask_eeg,          # [B], True if EEG exists
        'mask_bvp': mask_bvp,          # [B], True if BVP exists
        'mask_face': mask_face,        # [B], True if face exists
        'network': [b['network'] for b in batch],
        'username': [b['username'] for b in batch],
        'pid': [b['pid'] for b in batch],
        'eid': [b['eid'] for b in batch],
        'columns': batch[0]['columns'],
    }
