from functools import partial

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.dataset import UsabilitySmellsDataset, hierarchical_collate_fn, UsabilitySmellsSubset
from dataset.utils import get_prepared_dataframe, get_eeg_data, get_face_data
from sklearn.preprocessing import MultiLabelBinarizer


class UsabilitySmellsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base_dir,
        level='task',
        label_type='binary',
        batch_size=8,
        num_workers=0,
        csv_feature_cols=None,
        normalize_cols=True,
        keep_only_log_cols=True,
        keep_only_norm_cols=True,
        train_ratio=0.8,
        test_ratio=0.1,
        shuffle=True,
        target_network=None,
        target_idxs=None,
        force_align=False,
        use_test_as_val=True,
        full_history=False,
        use_spectrograms=False,
        eeg_bvp_kwargs=None,
        face_kwargs=None,
        inner_cat=False,
        filter_by=None,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.level = level
        self.label_type = label_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.csv_feature_cols = csv_feature_cols
        self.normalize_cols = normalize_cols
        self.keep_only_log_cols = keep_only_log_cols
        self.keep_only_norm_cols = keep_only_norm_cols
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.target_network = target_network
        self.target_idxs = target_idxs
        self.force_align = force_align
        self.full_history = full_history
        self.use_spectrograms = use_spectrograms
        self.eeg_bvp_kwargs = eeg_bvp_kwargs
        self.face_kwargs = face_kwargs
        self.use_test_as_val = use_test_as_val
        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.indices = None
        self.built = False
        self.inner_cat = inner_cat
        self.filter_by = filter_by

    def setup(self, stage=None):
        # Check if already setup
        if self.built:
            print("Data module already set up. Skipping...")
            return

        print("Setting up the data module...")

        # Get the log data
        log_df = get_prepared_dataframe(
            self.base_dir,
            normalize_cols=self.normalize_cols,
            keep_only_log_cols=self.keep_only_log_cols,
            keep_only_norm_cols=self.keep_only_norm_cols
        )

        # Load EEG/BVP data (use level='action' for finer granularity)
        eeg_bvp_data = get_eeg_data(self.base_dir, level='action')

        # Load Face data (use level='action' for finer granularity)
        face_data = get_face_data(self.base_dir, level='action')

        # If filter_by is provided, filter the log_df, eeg_bvp_data, and face_data
        if self.filter_by is not None:
            network = self.filter_by[0]
            username = self.filter_by[1]
            # Filter log_df
            if network in log_df['network'].unique():
                log_df = log_df[(log_df['network'] == network)]
                log_df = log_df.reset_index(drop=True)
                if username is not None and username in log_df['username'].unique():
                    log_df = log_df[(log_df['username'] == username)]
                    log_df = log_df.reset_index(drop=True)
            # Filter EEG/BVP and Face data
            if network in eeg_bvp_data.keys():
                eeg_bvp_data = {network: eeg_bvp_data[network]}
                if username is not None and username in eeg_bvp_data[network].keys():
                    eeg_bvp_data[network] = {username: eeg_bvp_data[network][username]}
            if network in face_data.keys():
                face_data = {network: face_data[network]}
                if username is not None and username in face_data[network].keys():
                    face_data[network] = {username: face_data[network][username]}

        # If we're multi-label, build the MultiLabelBinarizer
        mlb = None
        if self.label_type == 'multilabel':
            col_name = 'y_task_multilabel' if self.level == 'task' else 'y_action_multilabel'
            all_label_lists = log_df[col_name].tolist()
            # ignore non-lists entries (action-level labels when level='task' or vice versa)
            all_label_lists = [x for x in all_label_lists if isinstance(x, list)]
            mlb = MultiLabelBinarizer()
            mlb.fit(all_label_lists)

        # Create a UsabilitySmellsDataset with all data
        self.dataset = UsabilitySmellsDataset(
            log_df=log_df,
            eeg_bvp_data=eeg_bvp_data,
            face_data=face_data,
            label_type=self.label_type,
            level=self.level,
            csv_feature_cols=self.csv_feature_cols,
            mlb=mlb,
            force_align=self.force_align,
            full_history=self.full_history,
            use_spectrograms=self.use_spectrograms,
            eeg_bvp_kwargs=self.eeg_bvp_kwargs,
            face_kwargs=self.face_kwargs,
            inner_cat=self.inner_cat
        )

        # Perform the train/val/test split
        n = len(self.dataset)
        if self.target_network is None:
            self.indices = np.arange(n)
            if self.shuffle:
                self.indices = np.random.permutation(self.indices)
        else:
            self.indices = [i for i, k in self.dataset.idx_to_network_username.items() if k[0] == self.target_network]
            if self.shuffle:
                self.indices = np.random.permutation(self.indices)

        if self.target_network is None and self.target_idxs is None:
            train_end = int(self.train_ratio * n)
            train_idx = self.indices[:train_end]
            test_idx = self.indices[train_end:]

        elif self.target_idxs is not None:
            # use the target indices directly as test set (useful for stratified sampling)
            if self.target_network == 'ALL':
                target_rows = [k for k, v in self.dataset.idx_to_network_username.items() if v in self.target_idxs]
            else:
                target_rows = [k for k, v in self.dataset.idx_to_network_username.items() if v[0] == self.target_network and v[1] in self.target_idxs]
            train_idx = [i for i in range(n) if i not in target_rows]
            test_idx = target_rows.copy()

        elif self.target_network is not None:
            # if target_network is not None, make sure only the indices from that network are used for test
            other_indices = [i for i, k in self.dataset.idx_to_network_username.items() if k[0] != self.target_network]
            n = len(self.indices)
            train_end = int(self.train_ratio * n)
            train_idx = other_indices + self.indices[:train_end]
            test_idx = self.indices[train_end:]
        else:
            raise ValueError("Invalid train/test split configuration")

        if self.use_test_as_val:
            self.train_ds = UsabilitySmellsSubset(self.dataset, train_idx)
            self.test_ds = UsabilitySmellsSubset(self.dataset, test_idx)
            self.val_ds = UsabilitySmellsSubset(self.dataset, test_idx)  # Use the test set as validation set
        else:
            n = len(train_idx)
            train_end = int(0.9 * n)  # 90% of the training set
            train_idx, val_idx = train_idx[:train_end], train_idx[train_end:]
            self.train_ds = UsabilitySmellsSubset(self.dataset, train_idx)
            self.val_ds = UsabilitySmellsSubset(self.dataset, val_idx)
            self.test_ds = UsabilitySmellsSubset(self.dataset, test_idx)

        self.built = True

    def train_dataloader(self):
        collate_fn = hierarchical_collate_fn
        if self.use_spectrograms:
            collate_fn = partial(hierarchical_collate_fn, use_spectrograms=True)
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        collate_fn = hierarchical_collate_fn
        if self.use_spectrograms:
            collate_fn = partial(hierarchical_collate_fn, use_spectrograms=True)
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        collate_fn = hierarchical_collate_fn
        if self.use_spectrograms:
            collate_fn = partial(hierarchical_collate_fn, use_spectrograms=True)
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
