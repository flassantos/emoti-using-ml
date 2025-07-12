import argparse
import os
from collections import defaultdict
from pprint import pprint
from weakref import finalize

# needs to be imported before pytorch_lightning
import comet_ml

import pytorch_lightning as pl
from lightning_fabric import seed_everything
from lightning_fabric.loggers import CSVLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger, WandbLogger
import torch
from torch.nn.functional import dropout

from dataset.data_module import UsabilitySmellsDataModule
from models.model import MultiModalHierarchicalModel
from models.model_full_history import FullHistoryMultiModalHierarchicalModel

eeg_bvp_kwargs = dict(
    fs_eeg=1000,  # 1000 Hz
    fs_bvp=1000,  # 1000 Hz
    window='hann',
    nperseg_eeg=500,  # window size
    noverlap_eeg=250,  # overlap
    nperseg_bvp=500,
    noverlap_bvp=250,
    return_db=True  # consider log-scale
)
face_kwargs = dict(
    window_size=1000,
    overlap=250
)

target_idxs = {
    'SN_1': {
        0: ['P30', 'P16', 'P25', 'P18', 'P09', 'P10', 'P31'],
        1: ['P26', 'P13', 'P01', 'P05', 'P17', 'P06', 'P14'],
        2: ['P12', 'P24', 'P02', 'P03', 'P27', 'P04'],
        3: ['P22', 'P28', 'P23', 'P19', 'P32', 'P21'],
        4: ['P08', 'P11', 'P15', 'P29', 'P20', 'P07'],
    },
    'SN_2': {
        0: ['L01', 'L06', 'L21', 'L27', 'L14', 'L08'],
        1: ['L22', 'L11', 'L13', 'L30', 'L25', 'L23'],
        2: ['L17', 'L04', 'L02', 'L18', 'L09', 'L07'],
        3: ['L24', 'L05', 'L03', 'L20', 'L12', 'L19'],
        4: ['L26', 'L15', 'L16', 'L29', 'L28', 'L10'],
    },
    'SN_3': {
        0: ['S06', 'S08'],
        1: ['S01', 'S02'],
        2: ['S03', 'S05'],
        3: ['S04'],
        4: ['S07'],
    },
    'ALL': {
        0: [
            ('SN_1', 'P30'), ('SN_1', 'P16'), ('SN_1', 'P25'), ('SN_1', 'P18'), ('SN_1', 'P09'), ('SN_1', 'P10'),
            ('SN_1', 'P31'), ('SN_2', 'L01'), ('SN_2', 'L06'), ('SN_2', 'L21'), ('SN_2', 'L27'), ('SN_2', 'L14'),
            ('SN_2', 'L08'), ('SN_3', 'S06'), ('SN_3', 'S08')
        ],
        1: [
            ('SN_1', 'P26'), ('SN_1', 'P13'), ('SN_1', 'P01'), ('SN_1', 'P05'), ('SN_1', 'P17'), ('SN_1', 'P06'),
            ('SN_1', 'P14'), ('SN_2', 'L22'), ('SN_2', 'L11'), ('SN_2', 'L13'), ('SN_2', 'L30'), ('SN_2', 'L25'),
            ('SN_2', 'L23'), ('SN_3', 'S01'), ('SN_3', 'S02')
        ],
        2: [
            ('SN_1', 'P12'), ('SN_1', 'P24'), ('SN_1', 'P02'), ('SN_1', 'P03'), ('SN_1', 'P27'), ('SN_1', 'P04'),
            ('SN_2', 'L17'), ('SN_2', 'L04'), ('SN_2', 'L02'), ('SN_2', 'L18'), ('SN_2', 'L09'), ('SN_2', 'L07'),
            ('SN_3', 'S03'), ('SN_3', 'S05')
        ],
        3: [
            ('SN_1', 'P22'), ('SN_1', 'P28'), ('SN_1', 'P23'), ('SN_1', 'P19'), ('SN_1', 'P32'), ('SN_1', 'P21'),
            ('SN_2', 'L24'), ('SN_2', 'L05'), ('SN_2', 'L03'), ('SN_2', 'L20'), ('SN_2', 'L12'), ('SN_2', 'L19'),
            ('SN_3', 'S04')
        ],
        4: [
            ('SN_1', 'P08'), ('SN_1', 'P11'), ('SN_1', 'P15'), ('SN_1', 'P29'), ('SN_1', 'P20'), ('SN_1', 'P07'),
            ('SN_2', 'L26'), ('SN_2', 'L15'), ('SN_2', 'L16'), ('SN_2', 'L29'), ('SN_2', 'L28'), ('SN_2', 'L10'),
            ('SN_3', 'S07')
        ]
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Modal Model")

    # -- general stuff
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_final', action='store_true', help='Train a final model on ALL data (train+val+test)')
    parser.add_argument('--ablate', action='store_true', help='Ablate features')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save final models')

    # -- data module stuff
    parser.add_argument('--base_dir', type=str, default='UsabilitySmellsDataset', help='Path to your dataset folder')
    parser.add_argument('--level', type=str, default='task', choices=['task','action'], help='Whether to do task-level or action-level classification')
    parser.add_argument('--label_type', type=str, default='binary', choices=['binary','multilabel'], help='Binary or multi-label classification')
    parser.add_argument('--force_align', action='store_true', help='Whether to force align data')
    parser.add_argument('--full_history', action='store_true', help='Whether to use full history')
    parser.add_argument('--use_spectrograms', action='store_true', help='Whether to use spectrograms')
    parser.add_argument('--inner_cat', action='store_true', help='Whether to use concat eeg/face in the inner loop')

    # -- training hyperparams
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulate gradients')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','adamw','sgd'], help='Which optimizer to use')
    parser.add_argument('--scheduler', type=str, default=None, help='Which scheduler to use')
    parser.add_argument('--minor_class_weight', type=float, default=1.0, help='Weight for the minority class')
    parser.add_argument('--patience', type=int, default=-1, help='Early stopping patience (-1 = no early stopping)')

    # -- model dims
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for LSTMs, etc.')
    parser.add_argument('--kernel_size', type=int, default=7, help='Kernel size for CNNs; 0 for no CNN')
    parser.add_argument('--use_eeg_bvp', action='store_true', help='Whether to use EEG/BVP data')
    parser.add_argument('--use_face', action='store_true', help='Whether to use face data')
    parser.add_argument('--zero_eeg_bvp', action='store_true', help='Whether to zero out EEG/BVP features')
    parser.add_argument('--zero_face', action='store_true', help='Whether to zero out face features')
    parser.add_argument('--zero_sam', action='store_true', help='Whether to zero out SAM features')

    # -- data splits
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--target_network', type=str, default='ALL', choices=['SN_1','SN_2','SN_3', 'ALL'], help='If you want a specific network for test or CV, e.g. SN_1')

    # -- logger
    parser.add_argument('--logger', type=str, default='comet', choices=['none','comet','wandb','csv'], help='Which logger to use.')
    parser.add_argument('--project_name', type=str, default='emoti_using', help='Logger project name')
    parser.add_argument('--run_name', type=str, default='default_run', help='Logger run name')

    # -- cross-validation
    parser.add_argument('--k_folds', type=int, default=1, help='If >1, do cross-validation')

    args = parser.parse_args()

    # fix patience
    if args.patience == -1:
        args.patience = None
    return args


def get_logger(args):
    """Create and return a logger based on args.logger."""
    if args.logger == 'comet':
        logger = CometLogger(
            api_key='D5wDMnTugEFEqm2aibwQ088In',
            save_dir='comet_logs',
            project_name=args.project_name,
            workspace="mtreviso"
        )
    elif args.logger == 'wandb':
        logger = WandbLogger(
            project=args.project_name,
            name=args.run_name
        )
    elif args.logger == 'csv':
        logger = CSVLogger('csv_logs', name=args.run_name)
    else:
        logger = None
    return logger


def cross_validate(args, k_folds=5):
    """
    Example cross-validation approach. If you have user-level folds, define them or pass from data_module.
    We'll do a naive approach: repeatedly call data_module setup with different folds, train/test each time.
    """
    global target_idxs

    # build logger
    logger = get_logger(args)

    all_results = []
    for fold_idx in range(k_folds):
        print(f"=== Fold {fold_idx+1}/{k_folds} ===")
        # Possibly define a special fold split. For example,
        # data_module_class can take `fold_idx=fold_idx, total_folds=k` to do the right train/test
        dm = UsabilitySmellsDataModule(
            base_dir=args.base_dir,
            level=args.level,
            label_type=args.label_type,
            batch_size=args.batch_size,
            # for train/test split
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            # for target network
            target_network=args.target_network,
            target_idxs=target_idxs[args.target_network][fold_idx] if args.target_network in target_idxs else None,
            # for log data
            csv_feature_cols=None,
            normalize_cols=True,
            keep_only_log_cols=True,
            keep_only_norm_cols=True,
            # for force align and full history
            force_align=args.force_align,
            full_history=args.full_history,
            # for eeg/bvp and face spectrogram
            use_spectrograms=args.use_spectrograms,
            eeg_bvp_kwargs=eeg_bvp_kwargs,
            face_kwargs=face_kwargs,
            inner_cat=args.inner_cat,
        )

        # setup the data module
        dm.setup()

        # print label distribution for train, val, test
        print("Train label distribution:")
        dist, total = dm.train_ds.get_label_distribution()
        print('Total labels:', total)
        pprint(dist)
        print("Test label distribution:")
        dist, total = dm.test_ds.get_label_distribution()
        print('Total labels:', total)
        pprint(dist)

        # change the logger prefix
        logger._prefix = f"fold_{fold_idx + 1}"

        # build the model
        model_cls = FullHistoryMultiModalHierarchicalModel if args.full_history else MultiModalHierarchicalModel
        model = model_cls(
            csv_dim=dm.dataset.csv_dim,
            eeg_dim=dm.dataset.eeg_dim,
            bvp_dim=dm.dataset.bvp_dim,
            face_dim=dm.dataset.face_dim,
            hidden_dim=args.hidden_dim,
            kernel_size=args.kernel_size,
            num_labels=dm.dataset.num_labels,
            level=args.level,
            use_eeg_bvp=args.use_eeg_bvp,
            use_face=args.use_face,
            dropout=args.dropout,
            minor_class_weight=args.minor_class_weight,
            optimizer=args.optimizer,
            lr=args.lr,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            scheduler=args.scheduler,
            use_spectrograms=args.use_spectrograms,
        )

        # define trainer
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=logger,
            accelerator='auto',
            devices=1,
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.accumulate_grad_batches,
            use_distributed_sampler=False,
            callbacks=[
                EarlyStopping(monitor='val_f1_macro', mode='max', patience=args.patience),
                ModelCheckpoint(monitor="val_f1_macro", mode='max')
            ],
        )
        # print trainer info
        pprint(trainer.__dict__)

        trainer.fit(model, dm)
        result = trainer.test(model, dm, ckpt_path='best')

        # store result
        all_results.append(result[0])

    print("Cross Validation Results:")
    for i, r in enumerate(all_results):
        print(f"Fold {i+1} => {r}")

    # Average the results
    logger._prefix = f"avg"
    avg_results = {}
    for r in all_results:
        for k, v in r.items():
            avg_results[k] = avg_results.get(k, 0) + v
    for k in avg_results:
        avg_results[k] /= k_folds
    print("Average Results:", avg_results)

    # Log the average results
    logger.log_metrics(avg_results)


def cross_validate_and_ablate(args, k_folds=5):
    """
    Example cross-validation approach. If you have user-level folds, define them or pass from data_module.
    We'll do a naive approach: repeatedly call data_module setup with different folds, train/test each time.
    """
    global target_idxs

    # define feature combinations to ablate
    feature_combinations = [
        (True, True, True),    # EEG, Face, SAM
        (True, True, False),   # No SAM
        (True, False, True),   # No Face
        (False, True, True),   # No LOG
        (True, False, False),  # No Face, No SAM
        (False, False, False), # No EEG, No Face, No SAM
    ]

    # build logger
    logger = get_logger(args)

    all_results = defaultdict(list)
    for fold_idx in range(k_folds):
        print(f"=== Fold {fold_idx+1}/{k_folds} ===")
        # Possibly define a special fold split. For example,
        # data_module_class can take `fold_idx=fold_idx, total_folds=k` to do the right train/test
        dm = UsabilitySmellsDataModule(
            base_dir=args.base_dir,
            level=args.level,
            label_type=args.label_type,
            batch_size=args.batch_size,
            # for train/test split
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            # for target network
            target_network=args.target_network,
            target_idxs=target_idxs[args.target_network][fold_idx] if args.target_network in target_idxs else None,
            # for log data
            csv_feature_cols=None,
            normalize_cols=True,
            keep_only_log_cols=True,
            keep_only_norm_cols=True,
            # for force align and full history
            force_align=args.force_align,
            full_history=args.full_history,
            # for eeg/bvp and face spectrogram
            use_spectrograms=args.use_spectrograms,
            eeg_bvp_kwargs=eeg_bvp_kwargs,
            face_kwargs=face_kwargs,
            inner_cat=args.inner_cat,
        )

        # setup the data module
        dm.setup()

        # print label distribution for train, val, test
        print("Train label distribution:")
        dist, total = dm.train_ds.get_label_distribution()
        print('Total labels:', total)
        pprint(dist)
        print("Test label distribution:")
        dist, total = dm.test_ds.get_label_distribution()
        print('Total labels:', total)
        pprint(dist)

        # change the logger prefix
        logger._prefix = f"fold_{fold_idx + 1}"

        # build the model
        model_cls = FullHistoryMultiModalHierarchicalModel if args.full_history else MultiModalHierarchicalModel
        model = model_cls(
            csv_dim=dm.dataset.csv_dim,
            eeg_dim=dm.dataset.eeg_dim,
            bvp_dim=dm.dataset.bvp_dim,
            face_dim=dm.dataset.face_dim,
            hidden_dim=args.hidden_dim,
            kernel_size=args.kernel_size,
            num_labels=dm.dataset.num_labels,
            level=args.level,
            use_eeg_bvp=args.use_eeg_bvp,
            use_face=args.use_face,
            dropout=args.dropout,
            minor_class_weight=args.minor_class_weight,
            optimizer=args.optimizer,
            lr=args.lr,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            scheduler=args.scheduler,
            use_spectrograms=args.use_spectrograms,
        )

        # define trainer
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=logger,
            accelerator='auto',
            devices=1,
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.accumulate_grad_batches,
            use_distributed_sampler=False,
            callbacks=[
                EarlyStopping(monitor='val_f1_macro', mode='max', patience=args.patience),
                ModelCheckpoint(monitor="val_f1_macro", mode='max')
            ],
        )
        # print trainer info
        pprint(trainer.__dict__)

        # train model
        trainer.fit(model, dm)

        # ablate features
        for has_eeg_bvp, has_face, has_sam in feature_combinations:
            key = f"EEG={has_eeg_bvp}, Face={has_face}, SAM={has_sam}"
            print(f"Ablating features: ", key)
            model.zero_eeg_bvp = not has_eeg_bvp
            model.zero_face = not has_face
            model.zero_sam = not has_sam

            # Test the model
            result = trainer.test(model, dm, ckpt_path='best')
            # store result
            all_results[key].append(result[0])

    print("Cross Validation Results:")
    for key, results in all_results.items():
        print(f"Feature Combination: {key}")
        for i, r in enumerate(results):
            print(f"Fold {i+1} => {r}")

        print('')

        # Average the results
        logger._prefix = f"avg"
        avg_results = {}
        for r in results:
            for k, v in r.items():
                avg_results[key + "-" + k] = avg_results.get(key + "-" + k, 0) + v
        for k in avg_results:
            avg_results[k] /= k_folds
        print("Average Results:", avg_results)

        # Log the average results
        logger.log_metrics(avg_results)


def train_final_model(args):
    """Train a model on ALL data (train+test+val combined) and save it."""
    print("=== Training FINAL MODEL on ALL data ===")

    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Setup data module with train_ratio=1.0 to use all data for training
    dm = UsabilitySmellsDataModule(
        base_dir=args.base_dir,
        level=args.level,
        label_type=args.label_type,
        batch_size=args.batch_size,
        # Setting train_ratio=1.0 to use all data for training
        train_ratio=1.0,
        test_ratio=0.0,
        # No target network for final model
        target_network=None,
        target_idxs=None,
        # for log data
        csv_feature_cols=None,
        normalize_cols=True,
        keep_only_log_cols=True,
        keep_only_norm_cols=True,
        # for force align and full history
        force_align=args.force_align,
        full_history=args.full_history,
        # for eeg/bvp and face spectrogram
        use_spectrograms=args.use_spectrograms,
        eeg_bvp_kwargs=eeg_bvp_kwargs,
        face_kwargs=face_kwargs,
        inner_cat=args.inner_cat,
    )
    dm.setup()

    # print data information
    print("Final model training data distribution:")
    dist, total = dm.train_ds.get_label_distribution()
    print('Total labels:', total)
    pprint(dist)

    # Build the model
    model_cls = FullHistoryMultiModalHierarchicalModel if args.full_history else MultiModalHierarchicalModel
    model = model_cls(
        csv_dim=dm.dataset.csv_dim,
        eeg_dim=dm.dataset.eeg_dim,
        bvp_dim=dm.dataset.bvp_dim,
        face_dim=dm.dataset.face_dim,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        num_labels=dm.dataset.num_labels,
        level=args.level,
        use_eeg_bvp=args.use_eeg_bvp,
        use_face=args.use_face,
        zero_sam=args.zero_sam,
        zero_eeg_bvp=args.zero_eeg_bvp,
        zero_face=args.zero_face,
        dropout=args.dropout,
        minor_class_weight=args.minor_class_weight,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        scheduler=args.scheduler,
        use_spectrograms=args.use_spectrograms,
    )

    # Create logger for final model
    logger = get_logger(args)
    if logger:
        logger._prefix = "final_model"

    # Create a checkpoint callback that always saves the latest model
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=f"{args.level}_{args.label_type}_eeg{int(args.use_eeg_bvp)}_face{int(args.use_face)}",
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )

    # Create trainer with no validation
    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        accelerator="auto",
        devices=1,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback],
    )

    # Train model
    trainer.fit(model, dm)

    # Get the path to the final model checkpoint
    final_model_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    print(f"Final model saved to: {final_model_path}")

    return final_model_path


def single_run(args):
    # 1) build data module
    dm = UsabilitySmellsDataModule(
        base_dir=args.base_dir,
        level=args.level,
        label_type=args.label_type,
        batch_size=args.batch_size,
        # for train/test split
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        # for target network
        target_network=args.target_network if args.target_network != 'ALL' else None,  # set to None for k_folds=1
        target_idxs=None,
        # for log data
        csv_feature_cols=None,
        normalize_cols=True,
        keep_only_log_cols=True,
        keep_only_norm_cols=True,
        # for force align and full history
        force_align=args.force_align,
        full_history=args.full_history,
        # for eeg/bvp and face spectrogram
        use_spectrograms=args.use_spectrograms,
        eeg_bvp_kwargs=eeg_bvp_kwargs,
        face_kwargs=face_kwargs,
        inner_cat=args.inner_cat,
    )
    dm.setup()

    # print label distribution for train, val, test
    print("Train label distribution:")
    dist, total = dm.train_ds.get_label_distribution()
    print('Total labels:', total)
    pprint(dist)

    print("Train feature distribution:")
    dist = dm.train_ds.get_feature_distribution()
    for k, (mean, std) in dist.items():
        print('{:.36s}: {:.2f} +/- {:.2f}'.format(k, mean, std))

    print("Train lengths")
    min_l, mean_l, std_l, max_l = dm.train_ds.get_input_lengths_distribution()
    print('Min: {}, Mean: {:.2f}, Std: {:.2f}, Max: {}'.format(min_l, mean_l, std_l, max_l))

    print("Test label distribution:")
    dist, total = dm.test_ds.get_label_distribution()
    print('Total labels:', total)
    pprint(dist)

    print("Test feature distribution:")
    dist = dm.test_ds.get_feature_distribution()
    for k, (mean, std) in dist.items():
        print('{:.36s}: {:.2f} +/- {:.2f}'.format(k, mean, std))

    print("Test lengths")
    min_l, mean_l, std_l, max_l = dm.test_ds.get_input_lengths_distribution()
    print('Min: {}, Mean: {:.2f}, Std: {:.2f}, Max: {}'.format(min_l, mean_l, std_l, max_l))

    # 2) build the model
    model_cls = FullHistoryMultiModalHierarchicalModel if args.full_history else MultiModalHierarchicalModel
    model = model_cls(
        csv_dim=dm.dataset.csv_dim,
        eeg_dim=dm.dataset.eeg_dim,
        bvp_dim=dm.dataset.bvp_dim,
        face_dim=dm.dataset.face_dim,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        num_labels=dm.dataset.num_labels,
        level=args.level,
        use_eeg_bvp=args.use_eeg_bvp,
        use_face=args.use_face,
        zero_sam=args.zero_sam,
        zero_eeg_bvp=args.zero_eeg_bvp,
        zero_face=args.zero_face,
        dropout=args.dropout,
        minor_class_weight=args.minor_class_weight,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        scheduler=args.scheduler,
        use_spectrograms=args.use_spectrograms,
    )

    # 3) create logger
    logger = get_logger(args)

    # 4) create trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        accelerator="auto",
        devices=1,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        use_distributed_sampler=False,
        callbacks=[
            EarlyStopping(monitor='val_f1_macro', mode='max', patience=args.patience),
            ModelCheckpoint(monitor="val_f1_macro", mode='max')
        ],
    )
    pprint(trainer.__dict__)

    # 5) fit
    trainer.fit(model, dm)

    # 6) test
    result = trainer.test(model, dm, ckpt_path='best')
    print("Test Results:")
    pprint(result[0])


def main():
    args = parse_args()

    # set seed
    seed_everything(args.seed)

    # Check if we should train the final model on all data
    if args.train_final:
        final_model_path = train_final_model(args)
        print(f"Final model training completed. Model saved to: {final_model_path}")
        return

    # Check if we should run an ablation study
    if args.ablate:
        print(f"Running ablation study with {args.k_folds}-fold cross-validation")
        cross_validate_and_ablate(args, k_folds=args.k_folds)
        return

    # if k_folds>1 => do cross validation
    if args.k_folds > 1:
        print(f"Running {args.k_folds}-fold cross-validation")
        cross_validate(args, k_folds=args.k_folds)
        return

    # else do a single run for debugging
    print("Running single run")
    single_run(args)


if __name__ == "__main__":
    main()
