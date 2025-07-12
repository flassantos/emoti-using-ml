import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, average_precision_score,
    jaccard_score, hamming_loss
)


# https://github.com/itakurah/Focal-loss-PyTorch/blob/main/focal_loss.py
def multi_label_focal_loss(inputs, targets, alpha=None, gamma=2.0, reduction='mean'):
    """ Focal loss for multi-label classification. """
    probs = torch.sigmoid(inputs)

    # Compute binary cross entropy
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    # Compute focal weight
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    # Apply alpha if provided
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        bce_loss = alpha_t * bce_loss

    # Apply focal loss weight
    loss = focal_weight * bce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def asymmetric_multi_label_focal_loss(logits, targets, gamma_pos=0, gamma_neg=4, clip=1e-4, eps=1e-8, reduction='mean'):
    """ Asymmetric multi-label focal loss. """
    # Calculate probabilities
    probas = torch.sigmoid(logits)

    # Optionally clip probabilities to prevent small gradients for negatives
    if clip is not None and clip > 0:
        probas = torch.clamp(probas, min=clip, max=1 - clip)

    # Calculate losses for positive and negative targets
    pos_loss = targets * torch.log(probas + eps)
    neg_loss = (1 - targets) * torch.log(1 - probas + eps)

    # Asymmetric modulating factors
    modulator = (1 - probas) ** gamma_pos * targets + (probas) ** gamma_neg * (1 - targets)

    # Final loss calculation
    loss = - modulator * (pos_loss + neg_loss)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


class DummyLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, epoch=None):
        pass

# Custom optimizer/scheduler
def create_optimizer_scheduler(model, args):
    """Example function to create the optimizer & scheduler based on user args."""
    # Collect model parameters
    params = model.parameters()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=args.lr, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    else:  # sgd
        optimizer = torch.optim.SGD(
            params, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9
        )

    # Example: step LR every epoch
    if args.scheduler is None:
        # use a dummy scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=1)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print('Optimizer:', optimizer)
    print('Scheduler:', scheduler)

    # Return them in a dict for Pytorch Lightning
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss',  # optional
        } if scheduler is not None else None
    }

def lens_to_mask(lengths, max_len=None):
    """
    lengths: [B] with actual lengths
    max_len: optional maximum length, to pad the mask
    returns: mask [B, max_len]
    """
    B = lengths.size(0)
    max_len = max_len or lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(B, -1) < lengths.unsqueeze(1)
    return mask


def safe_lstm_aplpy(lstm_fn, x, lengths):
    # mask out zero-length sequences => replace with 1
    zero_mask = lengths == 0
    pos_lengths = lengths.masked_fill(zero_mask, 1)  # avoid 0

    # pack
    lengths_cpu = pos_lengths.cpu()  # needed
    packed_in = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
    packed_out, (h, c) = lstm_fn(packed_in)

    # un-pack
    lstm_out, lens_unpacked = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

    # mask out invalid steps
    lstm_out = lstm_out.masked_fill(zero_mask.unsqueeze(-1).unsqueeze(-1), 0)
    return h, lstm_out


class SimpleAttention(nn.Module):
    """
    A single-head attention that takes Q=final_hidden, K=V=the entire LSTM outputs.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = float(hidden_dim) ** 0.5

    def forward(self, lstm_out, final_hidden, mask=None):
        """
        lstm_out: [B, T, H]
        final_hidden: [B, H] or [B, T, H]
        mask: optional [B, T] mask

        Returns: context: [B, T, H] or maybe [B, T, H].
        We'll do a *per-sequence* approach:
          - Q = final_hidden (shape [B, H])
          - K,V = all steps => shape [B, T, H]
        We'll produce an attended context for *each step*.
        But typically, an attention is a single vector.
        Let’s keep it simple: we produce a single [B, H] by weighting all steps.
        """
        B, T, H = lstm_out.shape
        if final_hidden.dim() == 2:
            # Q => [B, 1, H]
            q = final_hidden.unsqueeze(1)  # [B,1,H]
        else:
            q = final_hidden
        # K => [B, T, H], dot => [B,1,T]
        attn_scores = torch.bmm(q, lstm_out.transpose(1, 2)) / self.scale  # => [B,T,T]
        # optional mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)  # => [B,T,T]
        # Weighted sum
        context = torch.bmm(attn_weights, lstm_out)  # => [B,1,H]
        if final_hidden.dim() == 2:
            context = context.squeeze(1)  # => [B,1,H] => [B,H]
        return context  # or [B,H]


class Conv1DContextualizer(nn.Module):
    """
    A simple 1D convolutional contextualizer.

    We'll apply a 1D convolution to the input sequence.
    """

    def __init__(self, input_dim, hidden_dim=64, kernel_size=7, dropout=0.1):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        Returns: [B, T, hidden_dim]
        """
        out = self.conv1d(x.permute(0,2,1)).permute(0,2,1)
        out = self.tanh(out)
        out = self.dropout(out)
        return out


class LSTMWithAttention(nn.Module):
    """
    A submodule that:
      - pack+padded => LSTM => we keep all outputs => final hidden => self-attention => out
      - returns a sequence representation [B, T, H] (optional) or
        a single vector [B,T,H], or just the final attentional context. We can adapt.
    """

    def __init__(
            self,
            input_dim,
            hidden_dim=64,
            dropout=0.1,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
            quadratic_attn=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)
        self.attn = SimpleAttention(hidden_dim)
        self.quadratic_attn = quadratic_attn

    def forward(self, x, lengths, mask=None):
        """
        x: [B, T, input_dim], possibly padded
        lengths: [B] actual lengths
        Returns:
          - lstm_out: [B, T, H] after un-pack
          - context: [B, H], an attention-based summary
        """
        # linear => [B, T, H]
        x = self.linear(x)
        x = torch.relu(x)
        x = self.dropout(x)
        # pack + LSTM + un-pack
        h, lstm_out = safe_lstm_aplpy(self.lstm, x, lengths)
        # bidirectional => sum the two directions
        if self.bidirectional:
            lstm_out = lstm_out[:, :, :self.hidden_dim] + lstm_out[:, :, self.hidden_dim:]
        # final_hidden => h[-1,...] => shape [1,B,H] => [B,H]
        final_hidden = h[-1]  # => [B,H]
        # do self-attention => out.shape [B,H]
        if self.quadratic_attn:
            out = self.attn(lstm_out, lstm_out, mask=mask)
            out = out.mean(dim=1)  # [B,T,H] => [B,H]
        else:
            out = self.attn(lstm_out, final_hidden, mask=mask)
        return out, lstm_out


class EEGBVP_CNN(nn.Module):
    """
    CNN specialized for EEG/BVP spectrograms of shape [128 x 128].
    """
    def __init__(self, in_channels=2, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),        # optional batch norm
            nn.ReLU(),
            nn.MaxPool2d(2),           # => freq/time halved => [64 x 64]
            nn.Dropout2d(dropout),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),           # => [32 x 32]
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),           # => [16 x 16]
            nn.Dropout2d(dropout)
        )
        self.adapool = nn.AdaptiveAvgPool2d((8,8))       # => [8 x 8]
        self.fc = nn.Linear(64*8*8, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: [N, 1, 128, 128]
        returns [N, hidden_dim]
        """
        out = self.conv(x)          # => [N,64,16,16], then pooled => [N,64,8,8]
        out = self.adapool(out)
        out = out.flatten(1)
        out = self.fc(out)          # => [N, hidden_dim]
        out = torch.relu(out)
        out = self.dropout(out)
        return out


class Face_CNN(nn.Module):
    """
    CNN specialized for Face input of shape [128 x 16].
    E.g. freq=128, time=16, or chunk=128, face_dim=16.
    Adjust kernel sizes to your data.
    """
    def __init__(self, in_channels=1, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout2d(dropout),

            nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout2d(dropout),
        )
        # Suppose final shape is now [32, ...], adapt pool to (4,4) or (8,2) etc.
        self.adapool = nn.AdaptiveAvgPool2d((4,4))
        self.fc = nn.Linear(32*4*4, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: [N,1,128,16]
        returns [N, hidden_dim]
        """
        out = self.conv(x)
        out = self.adapool(out)
        out = out.flatten(1)
        out = self.fc(out)
        out = torch.relu(out)
        out = self.dropout(out)
        return out


class CrossAttention(nn.Module):
    """
    For each time step in query_seq, we produce an attention over the entire key_seq.
    We'll do a single-head approach for simplicity.
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.scale = float(hidden_dim)**0.5
        self.dropout = nn.Dropout(dropout)
        # Optionally define linear transforms Q, K, V. For simplicity, let's do direct.
        # self.Wq = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query_seq, key_seq, mask=None, return_attn_weights=False):
        """
        query_seq: [B, Tq, H]
        key_seq: [B, Tk, H]
        mask: optional [B, Tk] mask
        return_attn_weights: if True, return the attention weights

        We'll produce cross_attn_out => shape [B, Tq, H],
        where each q in [Tq] attends to [Tk].
        """
        # B, Tq, H = query_seq.shape
        # Tk = key_seq.shape[1]
        query_seq = self.linear_q(query_seq)  # [B, Tq, H]
        key_seq = self.linear_k(key_seq)      # [B, Tk, H]

        # Q => [B,Tq,H], K=> [B,Tk,H], => dot => [B,Tq,Tk]
        # we do a batch matmul:
        # (B,Tq,H) x (B,H,Tk) => (B,Tq,Tk)
        scores = torch.bmm(query_seq, key_seq.transpose(1,2)) / self.scale

        # apply mask for key_lens
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)  # => [B,Tq,Tk]
        attn_weights = self.dropout(attn_weights)

        # Weighted sum => out => [B,Tq,H]
        # attn_weights => [B,Tq,Tk], key_seq => [B,Tk,H]
        out = torch.bmm(attn_weights, key_seq)  # => [B,Tq,H]

        if return_attn_weights:
            return out, attn_weights  # [B,Tq,H], [B,Tq,Tk]

        return out  # shape [B,Tq,H]


def get_cols_index_that_start_with_prefix(cols, prefix):
    """Get indices of columns that start with a specific prefix."""
    indices = []
    for i, col in enumerate(cols):
        if col.startswith(prefix):
            indices.append(i)
    return indices

class MultiModalHierarchicalModel(pl.LightningModule):
    def __init__(self,
            csv_dim=32,
            eeg_dim=128,
            bvp_dim=128,
            face_dim=16,
            hidden_dim=64,
            kernel_size=7,
            num_labels=1,  # 1 for binary, or K for multi-label
            level='task',
            use_eeg_bvp=True,
            use_face=True,
            zero_sam=False,
            zero_eeg_bvp=False,
            zero_face=False,
            dropout=0.1,
            minor_class_weight=5.0,
            log_dynamic_temperature=False,
            log_num_lstm_layers=1,
            log_quadratic_attn=False,
            use_spectrograms=False,
            # optimizer/scheduler args
            optimizer='adam',
            lr=1e-3,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.999,
            scheduler=None,
            **kwargs
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['kwargs', 'test_dataloader_tmp', 'val_dataloader_tmp'])

        self.num_labels = num_labels
        self.level = level
        self.hidden_dim = hidden_dim
        self.csv_dim = csv_dim
        self.eeg_dim = eeg_dim
        self.bvp_dim = bvp_dim
        self.face_dim = face_dim
        self.use_eeg_bvp = use_eeg_bvp
        self.use_face = use_face
        self.zero_sam = zero_sam
        self.zero_eeg_bvp = zero_eeg_bvp
        self.zero_face = zero_face
        self.kernel_size = kernel_size
        self.minor_class_weight = minor_class_weight
        self.log_dynamic_temperature = log_dynamic_temperature
        self.log_num_lstm_layers = log_num_lstm_layers
        self.log_quadratic_attn = log_quadratic_attn
        self.use_spectrograms = use_spectrograms
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.scheduler = scheduler
        self.dropout = nn.Dropout(dropout)

        # LOG branch => LSTMWithAttention
        if kernel_size > 0:
            self.log_cnn1d = Conv1DContextualizer(csv_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, dropout=dropout)
            self.log_lstm = LSTMWithAttention(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_layers=log_num_lstm_layers,
                quadratic_attn=log_quadratic_attn
            )
        else:
            self.log_cnn1d = None
            self.log_lstm = LSTMWithAttention(
                input_dim=csv_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_layers=log_num_lstm_layers,
                quadratic_attn=log_quadratic_attn
            )

        # EEG/BVP => 2D CNN => LSTMWithAttention
        if self.use_eeg_bvp:
            # bvp_dim == eeg_dim
            if self.use_spectrograms:
                self.eeg_bvp_cnn = EEGBVP_CNN(in_channels=2, hidden_dim=hidden_dim, dropout=dropout)
                self.eeg_bvp_lstm = LSTMWithAttention(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
                )
            else:
                self.eeg_bvp_cnn = lambda x: x
                self.eeg_bvp_lstm = LSTMWithAttention(
                    input_dim=eeg_dim + bvp_dim, hidden_dim=hidden_dim, dropout=dropout,
                    num_layers=1, bidirectional=True
                )
            # Cross-attention from log->EEG
            self.cross_attn_eeg = CrossAttention(hidden_dim, dropout=dropout)

        # Face => 2D CNN => LSTMWithAttention
        if self.use_face:
            if self.use_spectrograms:
                self.face_cnn = Face_CNN(in_channels=1, hidden_dim=hidden_dim, dropout=dropout)
                self.face_lstm = LSTMWithAttention(input_dim=hidden_dim, hidden_dim=hidden_dim)
            else:
                self.face_cnn = lambda x: x
                self.face_lstm = LSTMWithAttention(
                    input_dim=face_dim, hidden_dim=hidden_dim, dropout=dropout,
                    num_layers=1, bidirectional=True
                )

            # Cross-attention from log->Face
            self.cross_attn_face = CrossAttention(hidden_dim, dropout=dropout)

        # Gating alpha for the final combo
        self.gate_logits = nn.Linear(hidden_dim, 3)

        # final classifier => produce [B, T, num_labels]
        if self.num_labels == 1:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels)
            )

        # For testing, we'll store predictions and labels
        self.train_preds = []
        self.train_labels = []
        self.test_preds = []
        self.test_labels = []
        self.attn_weights_eeg = []
        self.attn_weights_face = []
        self.gate_alpha = []

    def configure_optimizers(self):
        conf = create_optimizer_scheduler(self, self.hparams)
        return conf

    def forward(self, batch):
        """
        batch:
          'batch_log': [B, T, csv_dim]
          'batch_labels': [B, T, num_labels or 1]
          'batch_eeg': [B, T, freq, time]
          'batch_bvp': maybe you unify or skip?
          'batch_face': [B, T, face_freq, face_time]
          'lengths': [B], number of intervals
          'lengths_eeg', 'lengths_bvp', 'lengths_face': [B] lengths
          'mask_eeg', 'mask_bvp', 'mask_face': [B] booleans
        returns: logits => [B, T, num_labels]
        """
        device = self.device
        batch_log = batch['batch_log'].to(device)        # shape [B, T, csv_dim]
        batch_labels = batch['batch_labels'].to(device)  # shape [B, T, num_labels]
        lengths = batch['lengths'].to(device)            # shape [B]
        lengths_eeg = batch.get('lengths_eeg', lengths)
        lengths_bvp = batch.get('lengths_bvp', lengths)
        lengths_face = batch.get('lengths_face', lengths)
        mask = lens_to_mask(lengths)
        B, T, csv_dim = batch_log.shape

        # For ablation studies
        if not self.training and self.zero_sam:
            # If not using SAM, set the dimensions of the batch_log to 0
            # Just to simulate the behavior of not using SAM
            batch_cols = batch.get('columns', [])
            sam_idxs = get_cols_index_that_start_with_prefix(batch_cols, 'sam_')
            if len(sam_idxs) > 0:
                # If there are sam columns, set them to 0
                batch_log[:, :, sam_idxs] = 0

        # 1) log => LSTM+attn => [B,T,H]
        # pack => log_lstm => out
        if self.log_cnn1d is not None:
            log_conv_out = self.log_cnn1d(batch_log)
            log_h, log_lstm_out = self.log_lstm(log_conv_out, lengths, mask=mask)
        else:
            log_h, log_lstm_out = self.log_lstm(batch_log, lengths, mask=mask)

        eeg_lstm_out = None
        if self.use_eeg_bvp and torch.sum(lengths_eeg).item() > 0:
            # 2) EEG => 2D CNN => shape [B*T,1,freq,time], LSTM => [B,T,H]
            # Actually we have batch_eeg shape [B, T, freq, time]
            # We'll flatten => [B*T, freq,time], pass CNN => [B*T, hidden], reshape => [B,T, hidden], pack => LSTM
            batch_eeg = batch['batch_eeg'].to(device)
            batch_bvp = batch['batch_bvp'].to(device)
            assert lengths_eeg == lengths_bvp, "Lengths of EEG and BVP should match"

            if self.use_spectrograms:
                # stack EEG and BVP
                # [B, T, 2, freq, time]
                B, T_eeg, freq_eeg, time_eeg = batch_eeg.size()
                batch_eeg_bvp = torch.stack([batch_eeg, batch_bvp], dim=2)
                # [B*T, 2, freq, time]
                eeg_flat = batch_eeg_bvp.view(-1, 2, freq_eeg, time_eeg)
                eeg_emb_2d = self.eeg_bvp_cnn(eeg_flat)           # => [B*T_eeg, hidden]
                eeg_emb_2d = eeg_emb_2d.view(B, T_eeg, self.hidden_dim)

                # LSTM
                attn_mask_eeg = lens_to_mask(lengths_eeg)
                eeg_h, eeg_lstm_out = self.eeg_bvp_lstm(eeg_emb_2d, lengths_eeg, mask=attn_mask_eeg) # => [B,T,H]

            else:
                B, T_eeg, F_eeg = batch_eeg.size()
                B, T_bvp, F_bvp = batch_bvp.size()
                assert T_eeg == T_bvp, "Lengths of EEG and BVP should match"

                # [B, T, F_eeg+F_bvp]
                eeg_bvp_flat = torch.cat([batch_eeg, batch_bvp], dim=-1)
                attn_mask_eeg = lens_to_mask(lengths_eeg)
                eeg_h, eeg_lstm_out = self.eeg_bvp_lstm(eeg_bvp_flat, lengths_eeg, mask=attn_mask_eeg)


        face_lstm_out = None
        if self.use_face and torch.sum(lengths_face).item() > 0:
            # 3) Face => 2D CNN => LSTM
            batch_face = batch['batch_face'].to(device)
            lengths_face = batch.get('lengths_face', lengths)

            if self.use_spectrograms:
                # shape => [B,T,face_freq,face_time]
                T_face = batch_face.size(1)
                face_freq = batch_face.size(2)
                face_time = batch_face.size(3)
                face_flat = batch_face.view(-1, face_freq, face_time).unsqueeze(1) # => [B*T,1,face_freq,face_time]
                face_emb_2d = self.face_cnn(face_flat)                              # => [B*T, hidden]
                face_emb_2d = face_emb_2d.view(B, T_face, self.hidden_dim)

                attn_mask_face = lens_to_mask(lengths_face)
                face_h, face_lstm_out = self.face_lstm(face_emb_2d, lengths_face, mask=attn_mask_face) # => [B,T,H]

            else:
                # shape => [B,T,face_dim]
                T_face = batch_face.size(1)
                F_face = batch_face.size(2)
                attn_mask_face = lens_to_mask(lengths_face)
                face_h, face_lstm_out = self.face_lstm(batch_face, lengths_face, mask=attn_mask_face)

        # 4) cross-attn: from log_lstm_out => query, EEG => key
        # shape => log_lstm_out [B,T,H], eeg_lstm_out [B,T,H]
        # => cross_attn_eeg(log_lstm_out, eeg_lstm_out) => shape [B,T,H]
        log_eeg_attn = 0
        log_face_attn = 0
        if self.use_eeg_bvp and torch.sum(lengths_eeg).item() > 0:
            log_eeg_attn, log_eeg_attn_weights  = self.cross_attn_eeg(log_lstm_out, eeg_lstm_out, attn_mask_eeg,
                                                                      return_attn_weights=True)
            self.attn_weights_eeg = log_eeg_attn_weights.detach().cpu().numpy()  # [B,T,T]
        if self.use_face and torch.sum(lengths_face).item() > 0:
            log_face_attn, log_eeg_attn_weights = self.cross_attn_face(log_lstm_out, face_lstm_out, attn_mask_face,
                                                                       return_attn_weights=True)
            self.attn_weights_face = log_eeg_attn_weights.detach().cpu().numpy()  # [B,T,T]

        # 5) gating with mask
        gate_logits = self.gate_logits(log_lstm_out.mean(dim=1))  # shape [B,3]

        if self.use_eeg_bvp:
            mask_eeg = batch.get('mask_eeg', torch.zeros(B, dtype=torch.bool, device=device))  # if not present => all False
            gate_logits[:, 1] = gate_logits[:, 1].masked_fill(~mask_eeg, float('-inf'))
        else:
            gate_logits[:, 1] = float('-inf')

        if self.use_face:
            mask_face = batch.get('mask_face', torch.zeros(B, dtype=torch.bool, device=device))
            gate_logits[:, 2] = gate_logits[:, 2].masked_fill(~mask_face, float('-inf'))
        else:
            gate_logits[:, 2] = float('-inf')

        # For ablation studies
        if not self.training and self.zero_eeg_bvp:
            gate_logits[:, 1] = float('-inf')
        if not self.training and self.zero_face:
            gate_logits[:, 2] = float('-inf')

        # linearly increase log temperature from 0.2 to 1.0 in first 10 epochs
        # if self.log_dynamic_temperature and not self.use_eeg_bvp and not self.use_face:
        #     log_temperature = min(0.2 + max(0, self.current_epoch - 1) / 10.0, 1.0)
        # else:
        #     log_temperature = 1.0
        #
        # temperatures = [log_temperature, 1.0, 1.0]
        # temperatures = torch.tensor(temperatures, device=device)
        # gate_logits = gate_logits / temperatures.unsqueeze(0)
        alpha = torch.softmax(gate_logits, dim=1)  # shape [B,3]
        self.gate_alpha = alpha.detach().cpu().numpy()

        rep_log = alpha[:, 0].unsqueeze(-1).unsqueeze(-1) * log_lstm_out  # shape [B,T,H]
        rep_eeg = alpha[:, 1].unsqueeze(-1).unsqueeze(-1) * log_eeg_attn
        rep_face= alpha[:, 2].unsqueeze(-1).unsqueeze(-1) * log_face_attn
        final_seq = rep_log + rep_eeg + rep_face  # => [B,T,H]

        final_seq = self.dropout(final_seq)

        # 6) classifier => => [B,T,num_labels]
        logits = self.classifier(final_seq)  # => [B,T,num_labels]
        return logits

    def training_step(self, batch, batch_idx):
        """
        In binary case -> shape of 'logits' is [B, T, 2].
        In multi-label case -> shape of 'logits' is [B, T, num_labels].
        """
        logits = self.forward(batch)  # shape => [B, T, 2] or [B, T, num_labels]
        labels = batch['batch_labels'].to(self.device)
        B, T = labels.shape[0], labels.shape[1]

        # flatten
        if self.num_labels == 1:
            # => 2-class multi-class shape => [B,T,2]
            # label => [B,T]
            # We'll do NLLLoss with ignore_index = -100
            logits = logits.view(B * T, 2)  # => [B*T,2]
            labels = labels.view(B * T)  # => [B*T], each in {0,1 or -100}

            mask = (labels != -100)
            valid_logits = logits[mask]
            valid_labels = labels[mask]
            if valid_labels.numel() == 0:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                class_weights = torch.tensor([1.0, self.minor_class_weight], device=self.device)
                loss = torch.nn.functional.nll_loss(valid_logits, valid_labels.long(),
                                                    ignore_index=-100, weight=class_weights)
        else:
            # => multi-label => shape [B,T,num_labels], raw logits
            # label => [B,T,num_labels]
            logits = logits.view(B * T, self.num_labels)  # => [B*T, num_labels]
            labels = labels.view(B * T, self.num_labels)  # => [B*T, num_labels]

            # mask out -100
            mask = (labels != -100).all(dim=-1)
            valid_logits = logits[mask]
            valid_labels = labels[mask]
            if valid_labels.numel() == 0:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                loss = multi_label_focal_loss(valid_logits, valid_labels.float(), gamma=self.minor_class_weight)
                # loss = torch.nn.functional.binary_cross_entropy_with_logits(valid_logits, valid_labels.float())

        self.log("train_loss", loss)

        # Store for metrics
        self.train_preds.append(valid_logits.detach().cpu().numpy())
        self.train_labels.append(valid_labels.detach().cpu().numpy())

        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval_test_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_test_step(batch, batch_idx, "test")

    def _shared_eval_test_step(self, batch, batch_idx, prefix):
        logits = self.forward(batch)  # [B,T,2] or [B,T,num_labels]
        labels = batch['batch_labels'].to(self.device)
        B, T = labels.shape[0], labels.shape[1]

        if self.num_labels == 1:
            # binary => [B,T,2]
            logits = logits.view(B * T, 2)
            labels = labels.view(B * T)
            mask = (labels != -100)
            valid_logits = logits[mask]
            valid_labels = labels[mask]

            if valid_labels.numel() == 0:
                loss = torch.tensor(0.0, device=self.device)
            else:
                class_weights = torch.tensor([1.0, self.minor_class_weight], device=self.device)
                loss = torch.nn.functional.nll_loss(valid_logits, valid_labels.long(),
                                                    ignore_index=-100, weight=class_weights)

        else:
            # multi-label => shape [B,T,num_labels], raw logits
            logits = logits.view(B * T, self.num_labels)
            labels = labels.view(B * T, self.num_labels)
            mask = (labels != -100).all(dim=-1)
            valid_logits = logits[mask]
            valid_labels = labels[mask]
            if valid_labels.numel() == 0:
                loss = torch.tensor(0.0, device=self.device)
            else:
                loss = multi_label_focal_loss(valid_logits, valid_labels.float(), gamma=self.minor_class_weight)
                # loss = torch.nn.functional.binary_cross_entropy_with_logits(valid_logits, valid_labels.float())

        self.log(f"{prefix}_loss", loss)

        # Store for metrics
        self.test_preds.append(valid_logits.detach().cpu().numpy())
        self.test_labels.append(valid_labels.detach().cpu().numpy())

        return loss

    def on_train_epoch_end(self):
        self._shared_eval_test_epoch_end("train")

    def on_validation_epoch_end(self):
        self._shared_eval_test_epoch_end("val")

    def on_test_epoch_end(self):
        self._shared_eval_test_epoch_end("test")

    def _shared_eval_test_epoch_end(self, prefix):

        if prefix == 'train':
            all_preds = np.concatenate(self.train_preds, axis=0)
            all_labels = np.concatenate(self.train_labels, axis=0)
        else:
            all_preds = np.concatenate(self.test_preds, axis=0)  # shape => [N, 2] or [N, num_labels]
            all_labels = np.concatenate(self.test_labels, axis=0)  # shape => [N] or [N, num_labels]

        metrics = {}
        if self.num_labels == 1:
            # binary => shape [N,2], log-probs
            # => pred_bin = argmax, pred_proba = exp(...)[...,1]
            log_probs = torch.tensor(all_preds)  # [N,2]
            pred_proba = log_probs.exp()[:, 1].numpy()  # shape [N]
            pred_bin = all_preds.argmax(axis=-1)  # shape [N]
            labels_bin = all_labels  # shape [N]

            metrics['accuracy'] = accuracy_score(labels_bin, pred_bin)
            metrics['precision'] = precision_score(labels_bin, pred_bin, zero_division=0)
            metrics['recall'] = recall_score(labels_bin, pred_bin, zero_division=0)
            metrics['f1'] = f1_score(labels_bin, pred_bin, zero_division=0)
            metrics['balanced_accuracy'] = balanced_accuracy_score(labels_bin, pred_bin)
            metrics['precision_macro'] = precision_score(labels_bin, pred_bin, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(labels_bin, pred_bin, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(labels_bin, pred_bin, average='macro', zero_division=0)
            metrics['auprc'] = average_precision_score(labels_bin, pred_proba, average='macro')
            metrics['jaccard'] = jaccard_score(labels_bin, pred_bin, average='samples' if self.num_labels > 1 else 'binary', zero_division=0)
            metrics['hamming_score'] = 1 - hamming_loss(labels_bin, pred_bin)

        else:
            # multi-label => shape [N, num_labels], raw logits
            # => do sigmoid => threshold 0.5 => [N, num_labels] in {0,1}
            logits_tensor = torch.tensor(all_preds)  # [N,num_labels]
            probs = torch.sigmoid(logits_tensor)  # [N,num_labels]
            pred_bin = (probs >= 0.5).int().numpy()  # shape [N,num_labels]
            pred_proba = probs.numpy()  # shape [N,num_labels]
            labels_bin = all_labels  # shape [N,num_labels]

            metrics['accuracy'] = (labels_bin == pred_bin).all(-1).astype(float).mean()
            metrics['precision_macro'] = precision_score(labels_bin, pred_bin, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(labels_bin, pred_bin, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(labels_bin, pred_bin, average='macro', zero_division=0)
            metrics['f1_micro'] = f1_score(labels_bin, pred_bin, average='micro', zero_division=0)
            metrics['f1_weighted'] = f1_score(labels_bin, pred_bin, average='weighted', zero_division=0)
            metrics['jaccard'] = jaccard_score(labels_bin, pred_bin, average='macro', zero_division=0)
            metrics['hamming_score'] = 1 - hamming_loss(labels_bin, pred_bin)

        # log them
        for k, v in metrics.items():
            self.log(f"{prefix}_{k}", float(v))
        print(f"{prefix} metrics: {metrics}")

        # Clear buffers
        if prefix == 'train':
            self.train_preds.clear()
            self.train_labels.clear()
        else:
            self.test_preds.clear()
            self.test_labels.clear()

        return metrics
