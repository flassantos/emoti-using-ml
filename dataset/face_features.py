import numpy as np
from scipy.stats import entropy, zscore
from joblib import Parallel, delayed
import multiprocessing
from functools import partial


def normalize_features(features, method='zscore'):
    """
    Normalize features using different methods

    Parameters:
    -----------
    features : array
        Features to normalize
    method : str
        Normalization method (zscore, minmax, robust)

    Returns:
    --------
    normalized_features : array
        Normalized features
    """
    # This function should be implemented based on your existing code
    # I'm providing a placeholder implementation
    if method == 'zscore':
        # Custom robust z-scoring to avoid precision loss warnings
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)

        # Replace very small standard deviations with a minimum threshold
        eps = 1e-8
        stds[stds < eps] = 1.0  # Use 1.0 for features with near-zero variance

        # Apply z-score normalization manually
        return (features - means) / stds

    elif method == 'minmax':
        mins = np.min(features, axis=0)
        maxs = np.max(features, axis=0)
        denominator = maxs - mins
        denominator[denominator < 1e-8] = 1.0  # Avoid division by near-zero
        return (features - mins) / denominator

    elif method == 'robust':
        # Using percentiles for robust scaling
        q25 = np.percentile(features, 25, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0  # Avoid division by near-zero
        return (features - np.median(features, axis=0)) / iqr

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def _process_window(window_data, window_idx, step_samples, window_samples, sampling_rate,
                    emotions_names, valence_map, arousal_map, positive_emotions,
                    negative_emotions, high_arousal, low_arousal):
    """
    Process a single window of face emotion data to extract features

    Parameters:
    -----------
    window_data : array
        Emotion data for this window
    window_idx : int
        Index of the window
    step_samples, window_samples, sampling_rate : int
        Window parameters
    emotions_names, valence_map, arousal_map : dict/list
        Emotion mappings
    positive_emotions, negative_emotions, high_arousal, low_arousal : list
        Emotion category lists

    Returns:
    --------
    window_features : array
        Features extracted from this window
    window_time : float
        Center time of the window
    """
    n_emotions = len(emotions_names)

    # Define window boundaries
    start_idx = window_idx * step_samples
    end_idx = min(start_idx + window_samples, window_data.shape[0])
    window_time = (start_idx + end_idx) / (2 * sampling_rate)  # window center time

    # Initialize feature array - calculate size based on inputs
    n_features = (
            n_emotions +  # Mean probabilities for each emotion
            n_emotions +  # Standard deviation for each emotion
            2 +  # Valence, arousal
            4 +  # Quadrants
            1 +  # Entropy
            1 +  # Dominant confidence
            n_emotions +  # Temporal: trend (slope) for each emotion
            3  # Key emotion ratios: positive/negative, high/low arousal, primary/blend
    )
    features = np.zeros(n_features)

    # Get window data
    window = window_data[start_idx:end_idx, :]

    feature_idx = 0

    # 1. Mean emotion probabilities over window - vectorized
    emotion_means = np.mean(window, axis=0)
    features[feature_idx:feature_idx + n_emotions] = emotion_means
    feature_idx += n_emotions

    # 2. Standard deviation of emotion probabilities - vectorized
    emotion_stds = np.std(window, axis=0)
    features[feature_idx:feature_idx + n_emotions] = emotion_stds
    feature_idx += n_emotions

    # 3. Calculate valence and arousal (mean across window)
    # Vectorized calculation
    valence = np.sum([emotion_means[j] * valence_map[emotion]
                      for j, emotion in enumerate(emotions_names)])
    arousal = np.sum([emotion_means[j] * arousal_map[emotion]
                      for j, emotion in enumerate(emotions_names)])

    features[feature_idx] = valence
    feature_idx += 1
    features[feature_idx] = arousal
    feature_idx += 1

    # 4. Quadrants (based on mean valence/arousal)
    features[feature_idx] = 1 if valence > 0 and arousal > 0 else 0  # High valence, high arousal
    feature_idx += 1
    features[feature_idx] = 1 if valence > 0 and arousal <= 0 else 0  # High valence, low arousal
    feature_idx += 1
    features[feature_idx] = 1 if valence <= 0 and arousal > 0 else 0  # Low valence, high arousal
    feature_idx += 1
    features[feature_idx] = 1 if valence <= 0 and arousal <= 0 else 0  # Low valence, low arousal
    feature_idx += 1

    # 5. Entropy of mean emotion distribution
    features[feature_idx] = entropy(emotion_means)
    feature_idx += 1

    # 6. Dominant emotion confidence
    features[feature_idx] = np.max(emotion_means)
    feature_idx += 1

    # 7. Trends (slopes) for each emotion over the window
    if end_idx - start_idx > 1:
        x = np.arange(end_idx - start_idx)
        # Preallocate slopes array
        slopes = np.zeros(n_emotions)

        # Calculate trends for all emotions
        for j in range(n_emotions):
            y = window[:, j]
            if np.std(y) > 1e-6:  # Check for variation
                slopes[j] = np.polyfit(x, y, 1)[0]  # Only store the slope

        features[feature_idx:feature_idx + n_emotions] = slopes
    # else slopes stay 0
    feature_idx += n_emotions

    # 8. Key emotion ratios - vectorized where possible

    # Positive vs negative emotions ratio
    pos_indices = [j for j, emotion in enumerate(emotions_names) if emotion in positive_emotions]
    neg_indices = [j for j, emotion in enumerate(emotions_names) if emotion in negative_emotions]

    pos_sum = np.sum(emotion_means[pos_indices]) if pos_indices else 0
    neg_sum = np.sum(emotion_means[neg_indices]) if neg_indices else 0
    pos_neg_ratio = pos_sum / max(1e-6, neg_sum)  # Avoid division by zero
    features[feature_idx] = pos_neg_ratio
    feature_idx += 1

    # High vs low arousal ratio
    high_indices = [j for j, emotion in enumerate(emotions_names) if emotion in high_arousal]
    low_indices = [j for j, emotion in enumerate(emotions_names) if emotion in low_arousal]

    high_sum = np.sum(emotion_means[high_indices]) if high_indices else 0
    low_sum = np.sum(emotion_means[low_indices]) if low_indices else 0
    high_low_ratio = high_sum / max(1e-6, low_sum)  # Avoid division by zero
    features[feature_idx] = high_low_ratio
    feature_idx += 1

    # Primary emotion vs blended emotions ratio
    primary = np.max(emotion_means)
    secondary = np.sum(emotion_means) - primary
    primary_blend_ratio = primary / max(1e-6, secondary)  # Avoid division by zero
    features[feature_idx] = primary_blend_ratio

    return features, window_time


def extract_face_features(face_emotions_flat, emotions_names, sampling_rate=1000, window_size=1.0, overlap=0.5,
                          normalize=None, n_jobs=None):
    """
    Extract comprehensive features from facial emotion data using a sliding window approach

    Parameters:
    -----------
    face_emotions_flat : array, shape [n_samples, n_emotions]
        Array of emotion probabilities at 1000 fps
    emotions_names : list
        Names of the emotions corresponding to columns in face_emotions_flat
    sampling_rate : int, default=1000
        Sampling rate in frames per second
    window_size : float, default=1.0
        Window size in seconds
    overlap : float, default=0.5
        Window overlap proportion (0 to 1)
    normalize : str (zscore, minmax, robust), default=None
        Normalization method to apply
    n_jobs : int, default=None
        Number of jobs for parallel processing:
        None: one less than the number of cores
        -1: all cores
        Any positive number: that specific number of cores

    Returns:
    --------
    features : array shape [n_windows, n_features]
        Extracted features for each window
    feature_names : list
        Names of the extracted features
    window_times : array shape [n_windows]
        Center time of each window in seconds
    """
    n_samples, n_emotions = face_emotions_flat.shape

    # Calculate window parameters in samples
    window_samples = int(window_size * sampling_rate)
    step_samples = int(window_samples * (1 - overlap))

    # Calculate number of windows
    n_windows = max(1, (n_samples - window_samples) // step_samples + 1)

    # Set up feature names
    feature_names = []
    # 1. Mean emotion probabilities
    feature_names.extend([f"{emotion}_mean" for emotion in emotions_names])
    # 2. Standard deviation of emotions
    feature_names.extend([f"{emotion}_std" for emotion in emotions_names])
    # 3. Valence, arousal
    feature_names.extend(["valence", "arousal"])
    # 4. Quadrants
    feature_names.extend([
        "quadrant_excitedhappy",
        "quadrant_contentrelaxed",
        "quadrant_angryafraid",
        "quadrant_sadbored"
    ])
    # 5. Entropy
    feature_names.append("emotion_entropy")
    # 6. Dominant confidence
    feature_names.append("dominant_confidence")
    # 7. Trends
    feature_names.extend([f"{emotion}_trend" for emotion in emotions_names])
    # 8. Key ratios
    feature_names.extend([
        "positive_negative_ratio",
        "high_low_arousal_ratio",
        "primary_blend_ratio"
    ])

    # Determine number of features
    n_features_per_window = len(feature_names)

    # 1. Valence-Arousal mappings - precompute
    valence_map = {
        'anger': -0.8, 'disgust': -0.6, 'fear': -0.7, 'enjoyment': 0.8,
        'contempt': -0.5, 'sadness': -0.7, 'surprise': 0.1
    }

    arousal_map = {
        'anger': 0.7, 'disgust': 0.3, 'fear': 0.7, 'enjoyment': 0.5,
        'contempt': 0.3, 'sadness': -0.3, 'surprise': 0.7
    }

    # Create lists for positive and negative emotions
    positive_emotions = ['enjoyment', 'surprise']
    negative_emotions = ['anger', 'disgust', 'fear', 'contempt', 'sadness']

    # Create lists for high and low arousal emotions
    high_arousal = ['anger', 'fear', 'surprise']
    low_arousal = ['disgust', 'enjoyment', 'contempt', 'sadness']

    # Set up parallel processing
    if n_jobs is None:
        # Default: use all cores except one (leave one free for system processes)
        num_cores = max(1, multiprocessing.cpu_count() - 1)
    elif n_jobs == -1:
        # Use all available cores
        num_cores = multiprocessing.cpu_count()
    else:
        # Use the specified number of cores
        num_cores = max(1, int(n_jobs))

    # Create a partial function with all the constant parameters
    process_window_partial = partial(
        _process_window,
        face_emotions_flat,
        step_samples=step_samples,
        window_samples=window_samples,
        sampling_rate=sampling_rate,
        emotions_names=emotions_names,
        valence_map=valence_map,
        arousal_map=arousal_map,
        positive_emotions=positive_emotions,
        negative_emotions=negative_emotions,
        high_arousal=high_arousal,
        low_arousal=low_arousal
    )

    # Process windows in parallel
    results = Parallel(n_jobs=num_cores)(
        delayed(process_window_partial)(i) for i in range(n_windows)
    )

    # Unpack results
    features = np.zeros((n_windows, n_features_per_window))
    window_times = np.zeros(n_windows)

    for i, (window_features, window_time) in enumerate(results):
        features[i] = window_features
        window_times[i] = window_time

    # Apply normalization if requested
    if normalize is not None:
        features = normalize_features(features, method=normalize)

    return features, feature_names, window_times


def chunk_face_emotions(
    emotion_probs,
    window_size=64,
    overlap=32,
    agg_mode='mean'
):
    """
    Chunk the time dimension of face 'emotion_probs' to reduce length.
    Each chunk is of size `window_size` with an overlap of `overlap`,
    and we compute a summary (e.g., mean) within each chunk.

    Args:
        emotion_probs (ndarray): shape (T, 7), each row is a probability distribution
                                 over 7 emotions, summing to 1 along axis=-1.
        window_size (int): number of timesteps in each chunk/window
        overlap (int): number of overlapping timesteps between consecutive chunks
        agg_mode (str): which statistic to compute in each chunk.
                        Options might include 'mean', 'max', 'argmax', 'entropy', etc.

    Returns:
        chunked (ndarray): shape (N_chunks, 7) if agg_mode in {mean, max, etc.}
                           or shape (N_chunks,) if you do something like 'argmax' or 'entropy'
                           that collapses the emotion dimension.
    """
    T, L = emotion_probs.shape
    assert L == 7, "Expected shape (T, 7) for face emotions"

    step = window_size - overlap
    if step <= 0:
        raise ValueError(f"window_size={window_size} must be > overlap={overlap}")

    # We'll store a list of chunk-level stats
    chunks = []

    start = 0
    while start < T:
        end = min(start + window_size, T)
        chunk = emotion_probs[start:end]  # shape (chunk_len, 7)

        if agg_mode == 'mean':
            stat = chunk.mean(axis=0)  # shape (7,)
        elif agg_mode == 'max':
            stat = chunk.max(axis=0)   # shape (7,)
        elif agg_mode == 'argmax':
            # Could do argmax at each time => shape (chunk_len,)
            # then maybe the mode of those argmaxes or an average index?
            argmaxes = np.argmax(chunk, axis=-1)  # shape (chunk_len,)
            # e.g. the most frequent emotion in the chunk:
            stat = np.bincount(argmaxes, minlength=7).argmax()
            # shape is just () - a single integer. We'll store as [stat] for consistent shape
            stat = np.array([stat], dtype=np.float32)
        elif agg_mode == 'entropy':
            # chunk has shape (chunk_len, 7), each row is a probability distribution
            # Per-timestep entropies => shape (chunk_len,)
            # H(p) = - sum_i p_i log(p_i)
            small_eps = 1e-8
            chunk_entropy = -np.sum(chunk * np.log(chunk + small_eps), axis=-1)
            # Now we can average that across the chunk
            stat = np.array([chunk_entropy.mean()], dtype=np.float32)
        else:
            raise ValueError(f"Unknown agg_mode: {agg_mode}")

        chunks.append(stat)
        start += step  # move the window

        if end == T:
            break

    # Stack
    chunked = np.stack(chunks, axis=0)
    # e.g. shape (N_chunks, 7) if 'mean' or 'max', or shape (N_chunks, 1) if 'entropy',
    # or (N_chunks,) if you flattened it further

    return chunked


def chunk_face_emotions_stacked(
    emotion_probs,
    window_size=64,
    overlap=32
):
    """
    Chunks the time dimension (T) of face 'emotion_probs' (shape (T, 7)) into smaller windows,
    and for each chunk, computes four aggregator stats:
      1) mean probabilities (7-dim)
      2) max probabilities (7-dim)
      3) average entropy (1-dim)
      4) most-frequent (mode) emotion in that window (1-dim)
    Then concatenates all of them to produce a 16-dim feature vector per chunk.

    Args:
        emotion_probs (ndarray): shape (T, 7), each row is a probability distribution
                                 over 7 emotions, summing to 1 along axis=-1.
        window_size (int): number of timesteps in each chunk/window.
        overlap (int): number of overlapping timesteps between consecutive chunks.

    Returns:
        features (ndarray): shape (N_chunks, 16). The chunk-level features stacked,
                            where each row is [mean(7), max(7), avg_entropy(1), mode_emotion(1)].
    """
    # in case we get an empty array, return an empty array with shape (1, 16)
    if emotion_probs.size == 0:
        return np.zeros((1, 16), dtype=np.float32)

    T, L = emotion_probs.shape
    assert L == 7, "Expected shape (T, 7) for face emotions."

    step = window_size - overlap
    if step <= 0:
        raise ValueError(f"window_size={window_size} must be > overlap={overlap}")

    features = []
    start = 0

    while start < T:
        end = min(start + window_size, T)
        chunk = emotion_probs[start:end]  # shape (chunk_len, 7)

        # 1) Mean of each emotion
        mean_7 = chunk.mean(axis=0)  # shape [7]

        # 2) Max of each emotion
        max_7 = chunk.max(axis=0)    # shape [7]

        # 3) Average entropy across the chunk
        # Entropy H(p) = -sum_i [ p_i * log(p_i) ], for each row
        small_eps = 1e-8
        chunk_entropy = -np.sum(chunk * np.log(chunk + small_eps), axis=-1)  # shape [chunk_len]
        avg_ent = np.array([chunk_entropy.mean()], dtype=np.float32)         # shape [1]

        # 4) Most-frequent (mode) emotion
        #   Argmax each row -> shape (chunk_len,)
        #   Then find the emotion index that occurs most often
        argmaxes = np.argmax(chunk, axis=-1)             # shape [chunk_len]
        freq = np.bincount(argmaxes, minlength=7)        # shape [7]
        mode_emotion = freq.argmax()                     # single int
        mode_emotion = np.array([mode_emotion], dtype=np.float32)  # shape [1]

        # Concatenate => shape [7 + 7 + 1 + 1 = 16]
        feat_16 = np.concatenate([mean_7, max_7, avg_ent, mode_emotion], axis=0)
        features.append(feat_16)

        start += step
        if end == T:
            break

    # Stack all chunk-level features => shape (N_chunks, 16)
    features = np.stack(features, axis=0)
    return features
