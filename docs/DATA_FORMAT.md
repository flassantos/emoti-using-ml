# Data Format Specification

## Overview

The framework supports multimodal data from multiple sources:
- **Interaction Logs**: User behavior and interface interaction data
- **EEG/BVP Signals**: Neurophysiological measurements
- **Facial Expressions**: Emotion recognition from facial analysis
- **SAM Scores**: Self-Assessment Manikin ratings for valence and arousal


## Dataset Structure

### Primary Dataset (df_all.pkl)

The main dataset is a pandas DataFrame with the following structure:

```python
import pandas as pd
df = pd.read_pickle('dfs/df_all.pkl')
print(df.columns.tolist())
```

#### Core Columns

| Column | Type | Description |
|--------|------|-------------|
| `network` | str | Network identifier (SN_1, SN_2, SN_3) |
| `username` | str | Anonymized user identifier |
| `pid` | int | Process/page identifier |
| `eid` | int | Event identifier (0 for task-level, >0 for action-level) |
| `task_id` | int | Task identifier |
| `unix_time` | int | Unix timestamp |
| `time` | str | Human-readable timestamp |
| `url` | str | Page URL |

#### Label Columns

| Column | Type | Description |
|--------|------|-------------|
| `y_binary` | int | Binary usability label (0/1) |
| `y_task_multilabel` | list | Task-level multilabel (list of integers) |
| `y_action_multilabel` | list | Action-level multilabel (list of integers) |
| `labels_0`, `labels_1`, `labels_2` | various | Original label annotations |
| `labels_union`, `labels_inter` | list | Label aggregations |
| `labels_valid` | bool | Label validity flag |

#### Interaction Log Features

**Basic Event Data:**
- `rid`, `srid`, `rtid`: Various identifiers
- `event`: Event type (click, scroll, keypress, etc.)
- `dom_object`: DOM element type
- `finished_task`: Task completion status
- `xpath`: Element XPath
- `xpath_depth`, `xpath_div_count`, `xpath_num_unique`: XPath statistics

**Temporal Features:**
- `event_duration*`: Event timing measurements
- `event_cum_duration*`: Cumulative timing
- `episode_*_duration*`: Episode-level timing aggregations
- `episode_*_num*`: Episode-level event counts

**Interaction Patterns:**
- `event_click_*`: Click-related features (position, text length)
- `event_change_*`: Form change features
- `event_scroll_*`: Scrolling behavior
- `event_keypress_*`: Keyboard interaction

**Context Features:**
- `episode_total_*_prev_*`, `episode_total_*_next_*`: Historical context
- `dom_object_prev`, `dom_object_next`: Element context

#### EEG/BVP Features

All EEG features follow the pattern: `[episode|event]_bit_eeg_[signal_type]_[statistic]_[norm]`

**Signal Types:**
- `signal`: Raw EEG signal
- `features_ts`: Time series features
- `theta`, `alpha_low`, `alpha_high`, `beta`, `gamma`: Frequency bands

**Statistics:**
- `min`, `max`, `avg`, `std`: Statistical measures

**BVP Features:**
- `episode_bit_bvp_signal_*`: BVP signal statistics
- `episode_bit_bvp_onsets_*`: Heart rate onset detection
- `episode_bit_bvp_hr_*`: Heart rate measurements

#### Facial Expression Features

Pattern: `[episode|event]_face_avg_[emotion]_[norm]`

**Emotion Categories:**
- `0_anger_proba`: Anger probability
- `1_disgust_proba`: Disgust probability
- `2_fear_proba`: Fear probability
- `3_enjoyment_proba`: Enjoyment probability
- `4_contempt_proba`: Contempt probability
- `5_sadness_proba`: Sadness probability
- `6_surprise_proba`: Surprise probability

**Derived Features:**
- `most_likely`: Most probable emotion
- `entropy`: Emotion probability entropy
- `most_freq`: Most frequent emotion

#### SAM (Self-Assessment Manikin) Features

- `sam_valence`: Valence rating (1-9 scale)
- `sam_arousal`: Arousal rating (1-9 scale)

### Feature Normalization

Features come in two variants:
- **Raw features**: Original measurements
- **Normalized features**: Suffix `_norm`, normalized within user/session

Example:
```python
# Raw feature
'event_duration'
# Normalized feature  
'event_duration_norm'
```


More information can be found in the AMUSED repo: https://github.com/flassantos/amused/


## Neural Network Dataset Structure

For PyTorch Lightning models, data is organized in the `UsabilitySmellsDataset/` directory.



## Data Loading Examples

### Scikit-learn Models

```python
import pandas as pd
from run_grid_search_binary import get_features

# Load processed dataset
df = pd.read_pickle('dfs/df_all.pkl')

# Filter for task-level data
task_df = df[df['eid'] == 0]

# Get specific feature sets
log_features = get_features(log=True, eeg=False, face=False, sam=False)
eeg_features = get_features(log=False, eeg=True, face=False, sam=False)
all_features = get_features(log=True, eeg=True, face=True, sam=True)

# Extract features and labels
X = task_df[all_features]
y = task_df['y_binary']
```

### Neural Network Models

```python
from dataset.data_module import UsabilitySmellsDataModule

# Create data module
dm = UsabilitySmellsDataModule(
    base_dir='UsabilitySmellsDataset/',
    level='task',
    label_type='binary',
    batch_size=8,
    use_spectrograms=True
)

# Setup and get data loaders
dm.setup()
train_loader = dm.train_dataloader()

# Batch structure
for batch in train_loader:
    print("Log data:", batch['batch_log'].shape)      # [B, T, csv_dim]
    print("EEG data:", batch['batch_eeg'].shape)      # [B, T, freq, time] 
    print("Face data:", batch['batch_face'].shape)    # [B, T, face_freq, face_time]
    print("Labels:", batch['batch_labels'].shape)     # [B, T, num_labels]
    break
```

## Missing Data Handling

### Default Values
- **SAM features**: Missing values filled with `5` (neutral)
- **Other numeric features**: Missing values filled with `0`
- **Categorical features**: Converted to numeric codes, missing as `-1`

### Data Availability Masks
Neural network models use masks to handle missing modalities:
```python
batch['mask_eeg']   # Boolean mask for EEG availability
batch['mask_face']  # Boolean mask for face data availability
```

## Label Encoding

### Binary Classification
- `0`: No usability issue
- `1`: Usability issue detected

### Multilabel Classification
Labels are lists of integers representing different usability issue types:
```python
# Example multilabel encodings
[]         # No issues
[3]        # One issue
[1, 3, 5]  # Multiple issue types
```

### Level-specific Labels
- **Task-level** (`eid == 0`): High-level usability assessments
- **Action-level** (`eid > 0`): Granular interaction-level labels

## Data Preprocessing Pipeline

1. **Feature Engineering**: Temporal aggregations, context windows
2. **Normalization**: Per-user/session normalization for temporal features
3. **Alignment**: Temporal alignment between modalities
4. **Segmentation**: Task/episode segmentation for hierarchical modeling
5. **Quality Control**: Missing data imputation and outlier handling

## Custom Data Integration

To add new data sources:

1. **Extend feature extraction** in `dataset/utils.py`
2. **Update data loading** in `dataset/dataset.py`
3. **Modify model inputs** in `models/model.py`
4. **Add preprocessing** in preparation scripts

Example:
```python
def get_custom_features():
    return [
        'custom_feature_1',
        'custom_feature_2_norm',
        # ...
    ]
```