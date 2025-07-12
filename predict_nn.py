import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from dataset.data_module import UsabilitySmellsDataModule
from models.model import MultiModalHierarchicalModel
from models.model_full_history import FullHistoryMultiModalHierarchicalModel

# Same kwargs as in the training script
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


def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions from a trained model')

    # Model loading params
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model checkpoint')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')

    # Dataset params (should match those used for training)
    parser.add_argument('--base_dir', type=str, default='UsabilitySmellsDataset',
                        help='Path to your dataset folder')
    parser.add_argument('--level', type=str, default='task',
                        choices=['task', 'action'], help='Whether to do task-level or action-level classification')
    parser.add_argument('--label_type', type=str, default='binary',
                        choices=['binary', 'multilabel'], help='Binary or multi-label classification')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--force_align', action='store_true', help='Whether to force align data')
    parser.add_argument('--full_history', action='store_true', help='Whether to use full history')
    parser.add_argument('--use_spectrograms', action='store_true', help='Whether to use spectrograms')
    parser.add_argument('--inner_cat', action='store_true', help='Whether to use concat eeg/face in the inner loop')

    # Prediction specific params
    parser.add_argument('--use_eeg_bvp', action='store_true', help='Whether to use EEG/BVP data')
    parser.add_argument('--use_face', action='store_true', help='Whether to use face data')
    parser.add_argument('--predict_on', choices=['train', 'test', 'all'], default='all',
                        help='Which dataset to generate predictions for')
    parser.add_argument('--output_probabilities', action='store_true',
                        help='Whether to output probabilities in addition to class predictions')
    parser.add_argument('--output_filename', type=str, default='predictions.json',
                        help='Name of the output file')

    args = parser.parse_args()

    # If filename doesn't have .json extension, add it
    if not args.output_filename.endswith('.json'):
        args.output_filename += '.json'

    return args


def get_predictions(model, dataloader, device, num_labels=1, output_probabilities=False):
    """Generate predictions for a dataloader"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            # Move batch to device and get network/username info
            networks = batch['network']
            usernames = batch['username']
            pids = batch['pid']
            eids = batch['eid']

            # Generate predictions
            logits = model(batch)  # [B, T, num_classes]
            B, T = logits.shape[0], logits.shape[1]

            # Process predictions based on label type
            if num_labels == 1:  # Binary classification
                # Shape: [B, T, 2] (log probabilities)
                log_probs = logits.view(B, T, 2)
                probs = torch.exp(log_probs)  # Convert log probs to probabilities
                pred_classes = torch.argmax(log_probs, dim=-1)  # [B, T]

                # Get mask for valid predictions
                labels = batch['batch_labels'].to(device)
                mask = (labels != -100).view(B, T)

                for b in range(B):
                    for t in range(T):
                        if mask[b, t]:
                            item = {
                                'network': networks[b],
                                'username': usernames[b],
                                'pid': pids[b][t] if isinstance(pids[b], list) else pids[b],
                                'eid': eids[b][t] if isinstance(eids[b], list) else eids[b],
                                'prediction': int(pred_classes[b, t].item())
                            }

                            if output_probabilities:
                                item['probability'] = float(probs[b, t, 1].item())  # Prob of positive class

                            predictions.append(item)

    return predictions


def load_model(model_path, device, use_eeg_bvp=True, use_face=True, full_history=False):
    """Load a trained model from a checkpoint"""
    print(f"Loading model from {model_path}")

    # Use the appropriate model class
    model_cls = FullHistoryMultiModalHierarchicalModel if full_history else MultiModalHierarchicalModel

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract hyperparameters from checkpoint
    hparams = checkpoint['hyper_parameters']

    # Override EEG/BVP and Face usage if specified
    hparams['use_eeg_bvp'] = use_eeg_bvp
    hparams['use_face'] = use_face

    # Create model and load state
    model = model_cls(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded with parameters:")
    print(f"  - level: {hparams['level']}")
    print(f"  - num_labels: {hparams['num_labels']}")
    print(f"  - use_eeg_bvp: {hparams['use_eeg_bvp']}")
    print(f"  - use_face: {hparams['use_face']}")

    return model, hparams


def main():
    args = parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)

    # Load model
    model, hparams = load_model(
        args.model_path,
        device,
        use_eeg_bvp=args.use_eeg_bvp,
        use_face=args.use_face,
        full_history=args.full_history
    )

    # Setup data module
    dm = UsabilitySmellsDataModule(
        base_dir=args.base_dir,
        level=args.level,
        label_type=args.label_type,
        batch_size=args.batch_size,
        train_ratio=1.0,  # Default split
        test_ratio=0.0,  # Default split
        target_network=None,
        target_idxs=None,
        csv_feature_cols=None,
        normalize_cols=True,
        keep_only_log_cols=True,
        keep_only_norm_cols=True,
        force_align=args.force_align,
        full_history=args.full_history,
        use_spectrograms=args.use_spectrograms,
        eeg_bvp_kwargs=eeg_bvp_kwargs,
        face_kwargs=face_kwargs,
        inner_cat=args.inner_cat,
    )
    dm.setup()

    # Generate predictions based on the specified dataset

    print("Generating predictions...")
    all_predictions = get_predictions(
        model,
        dm.train_dataloader(),
        device,
        num_labels=hparams['num_labels'],
        output_probabilities=args.output_probabilities
    )

    # Save predictions to JSON file
    output_path = output_dir / args.output_filename
    print(f"Saving {len(all_predictions)} predictions to {output_path}")

    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    print("Done!")
