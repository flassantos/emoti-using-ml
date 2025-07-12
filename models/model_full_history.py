import numpy as np
import torch

from models.model import MultiModalHierarchicalModel, lens_to_mask, multi_label_focal_loss


def select_and_pad_tensor(data: torch.Tensor, mask: torch.Tensor, level: str, fill_value: float = 0) -> torch.Tensor:
    """
    Select timesteps from 'data' based on 'mask' and 'level', then pad sequences to equal length.

    Args:
        data (torch.Tensor): Input tensor of shape [batch, time] or [batch, time, feature].
        mask (torch.Tensor): Mask tensor of shape [batch, time] (dtype: long).
        level (str): Either "task" or "action".
                     For "task", selects timesteps where mask == 1.
                     For "action", selects timesteps where mask == 0.
        fill_value (float): Padding value to use.

    Returns:
        torch.Tensor: Padded tensor with selected timesteps.
                      - If data is 2D, output shape: [batch, max_selected_length].
                      - If data is 3D, output shape: [batch, max_selected_length, feature].
    """

    if level not in ("task", "action"):
        raise ValueError("Invalid level. Must be 'task' or 'action'.")

    # If batch size is 1, we can just do masked select and return the result.
    if data.size(0) == 1:
        val = 1 if level == "task" else 0
        sel_data = data[mask == val]
        return sel_data.unsqueeze(0)  # Add batch dimension back

    selected_list = []

    # Iterate over each batch
    for i in range(data.size(0)):
        if level == "task":
            indices = (mask[i] == 1).nonzero(as_tuple=False).squeeze(-1)
        else:  # level == "action"
            indices = (mask[i] == 0).nonzero(as_tuple=False).squeeze(-1)

        # Select the timesteps for the current batch.
        # For 2D data, data[i] is shape [time] -> selected: [num_selected]
        # For 3D data, data[i] is shape [time, feature] -> selected: [num_selected, feature]
        selected = data[i][indices]
        selected_list.append(selected)

    # Find the maximum number of selected timesteps across batches
    max_len = max(seq.size(0) for seq in selected_list)

    padded_list = []
    for seq in selected_list:
        pad_len = max_len - seq.size(0)
        if seq.dim() == 1:  # 2D input: seq is [time]
            pad_tensor = torch.full((pad_len,), fill_value, dtype=seq.dtype, device=seq.device)
            padded_seq = torch.cat([seq, pad_tensor], dim=0)
        elif seq.dim() == 2:  # 3D input: seq is [time, feature]
            pad_tensor = torch.full((pad_len, seq.size(1)), fill_value, dtype=seq.dtype, device=seq.device)
            padded_seq = torch.cat([seq, pad_tensor], dim=0)
        padded_list.append(padded_seq)

    # Stack padded sequences back into a single tensor.
    padded_tensor = torch.stack(padded_list, dim=0)
    return padded_tensor


def get_cols_index_that_start_with_prefix(cols, prefix):
    """Get indices of columns that start with a specific prefix."""
    indices = []
    for i, col in enumerate(cols):
        if col.startswith(prefix):
            indices.append(i)
    return indices


class FullHistoryMultiModalHierarchicalModel(MultiModalHierarchicalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        batch_level_mask = batch['batch_level_mask'].to(device)  # shape [B, T]
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

        # 5) Gather only valid logits for each modality
        # log_lstm_out = select_and_pad_tensor(log_lstm_out, batch_level_mask, level=self.level, fill_value=0)
        # if self.use_eeg_bvp and torch.sum(lengths_eeg).item() > 0:
        #     log_eeg_attn = select_and_pad_tensor(log_eeg_attn, batch_level_mask, level=self.level, fill_value=0)
        # if self.use_face and torch.sum(lengths_face).item() > 0:
        #     log_face_attn = select_and_pad_tensor(log_face_attn, batch_level_mask, level=self.level, fill_value=0)

        # 6) gating with mask
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
        self.gate_alpha = alpha.detach().cpu().numpy()  # [B,3]

        rep_log = alpha[:, 0].unsqueeze(-1).unsqueeze(-1) * log_lstm_out  # shape [B,T,H]
        rep_eeg = alpha[:, 1].unsqueeze(-1).unsqueeze(-1) * log_eeg_attn
        rep_face= alpha[:, 2].unsqueeze(-1).unsqueeze(-1) * log_face_attn
        final_seq = rep_log + rep_eeg + rep_face  # => [B,T,H]

        final_seq = self.dropout(final_seq)

        # 7) classifier => => [B,T,num_labels]
        logits = self.classifier(final_seq)  # => [B,T,num_labels]
        return logits

    def training_step(self, batch, batch_idx):
        """
        In binary case -> shape of 'logits' is [B, T, 2].
        In multi-label case -> shape of 'logits' is [B, T, num_labels].
        """
        logits = self.forward(batch)  # shape => [B, T, 2] or [B, T, num_labels]
        labels = batch['batch_labels'].to(logits.device)
        batch_level_mask = batch['batch_level_mask'].to(logits.device)  # shape [B, T]

        # labels = select_and_pad_tensor(labels, batch_level_mask, level=self.level, fill_value=-100).long()
        if self.level == 'task':
            mask = (batch_level_mask == 1).unsqueeze(-1) if self.num_labels > 1 else (batch_level_mask == 1)
            labels = labels.masked_fill(~mask, -100)
        else:
            mask = (batch_level_mask == 0).unsqueeze(-1) if self.num_labels > 1 else (batch_level_mask == 0)
            labels = labels.masked_fill(~mask, -100)

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


    def _shared_eval_test_step(self, batch, batch_idx, prefix):
        logits = self.forward(batch)  # [B,T,2] or [B,T,num_labels]
        labels = batch['batch_labels'].to(self.device)
        batch_level_mask = batch['batch_level_mask'].to(logits.device)  # shape [B, T]

        # labels = select_and_pad_tensor(labels, batch_level_mask, level=self.level, fill_value=-100).long()
        if self.level == 'task':
            mask = (batch_level_mask == 1).unsqueeze(-1) if self.num_labels > 1 else (batch_level_mask == 1)
            labels = labels.masked_fill(~mask, -100)
        else:
            mask = (batch_level_mask == 0).unsqueeze(-1) if self.num_labels > 1 else (batch_level_mask == 0)
            labels = labels.masked_fill(~mask, -100)

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
