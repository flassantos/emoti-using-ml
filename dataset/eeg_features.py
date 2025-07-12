import numpy as np
from scipy.stats import zscore
from scipy.signal import stft, welch
from joblib import Parallel, delayed
import multiprocessing
from functools import partial

from biosppy.signals import tools as st
from biosppy.signals.eeg import get_power_features


def get_eeg_features(signal=None, sampling_rate=1000.0, size=0.25, overlap=0.5):
    """Process raw EEG signals and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw EEG signal matrix; each column is one EEG channel.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : float, optional
        Window size (seconds).
    overlap : float, optional
        Window overlap (0 to 1).

    Returns
    -------
    features_ts : array
        Features time axis reference (seconds).
    theta : array
        Average power in the 4 to 8 Hz frequency band; each column is one EEG
        channel.
    alpha_low : array
        Average power in the 8 to 10 Hz frequency band; each column is one EEG
        channel.
    alpha_high : array
        Average power in the 10 to 13 Hz frequency band; each column is one EEG
        channel.
    beta : array
        Average power in the 13 to 25 Hz frequency band; each column is one EEG
        channel.
    gamma : array
        Average power in the 25 to 40 Hz frequency band; each column is one EEG
        channel.
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)
    signal = np.reshape(signal, (signal.shape[0], -1))

    sampling_rate = float(sampling_rate)
    nch = signal.shape[1]

    # high pass filter
    b, a = st.get_filter(
        ftype="butter",
        band="highpass",
        order=8,
        frequency=4,
        sampling_rate=sampling_rate,
    )

    aux, _ = st._filter_signal(b, a, signal=signal, check_phase=True, axis=0)

    # low pass filter
    b, a = st.get_filter(
        ftype="butter",
        band="lowpass",
        order=16,
        frequency=40,
        sampling_rate=sampling_rate,
    )

    filtered, _ = st._filter_signal(b, a, signal=aux, check_phase=True, axis=0)

    # band power features
    out = get_power_features(
        signal=filtered,
        sampling_rate=sampling_rate,
        size=size,
        overlap=overlap
    )
    ts_feat = out["ts"]
    theta = out["theta"]
    alpha_low = out["alpha_low"]
    alpha_high = out["alpha_high"]
    beta = out["beta"]
    gamma = out["gamma"]

    return ts_feat, theta, alpha_low, alpha_high, beta, gamma


def extract_combined_eeg_features(eeg_signal, sampling_rate=1000.0, size=1.0, overlap=0.5, n_jobs=None):
    """
    Extract a comprehensive set of EEG features for emotion recognition,
    using biosppy for band extraction and adding additional features.

    Parameters
    ----------
    eeg_signal : array
        Raw EEG signal.
    sampling_rate : float, optional
        Sampling frequency (Hz).
    size : float, optional
        Window size (seconds).
    overlap : float, optional
        Window overlap (0 to 1).

    Returns
    -------
    combined_features : array
        Matrix of extracted features with shape [n_windows, n_features].
        Each row represents a time window, each column a feature.
    feature_names : list
        Names of the extracted features.
    """
    # Reshape signal to 2D if needed (assuming single channel)
    if eeg_signal.ndim == 1:
        eeg_signal = eeg_signal.reshape(-1, 1)

    # Extract frequency bands using biosppy's function
    ts_feat, theta, alpha_low, alpha_high, beta, gamma = get_eeg_features(
        signal=eeg_signal,
        sampling_rate=sampling_rate,
        size=size,
        overlap=overlap
    )

    # Number of windows
    n_windows = len(ts_feat)

    # Window size in samples
    win_samples = int(size * sampling_rate)
    # Step size in samples
    step_samples = int(win_samples * (1 - overlap))

    # Pre-allocate feature arrays
    spectral_features = np.zeros((n_windows, 5))
    temporal_features = np.zeros((n_windows, 5))
    emotion_ratios = np.zeros((n_windows, 3))

    # Calculate window start indices in a vectorized way
    start_indices = np.minimum(
        (ts_feat * sampling_rate).astype(int),
        eeg_signal.shape[0] - 1
    )

    # ---------- PARALLEL PROCESSING APPROACH ----------
    # Set the number of jobs for parallel processing
    if n_jobs is None:
        # Default: use all cores except one (leave one free for system processes)
        num_cores = max(1, multiprocessing.cpu_count() - 1)
    elif n_jobs == -1:
        # Use all available cores
        num_cores = multiprocessing.cpu_count()
    else:
        # Use the specified number of cores
        num_cores = max(1, int(n_jobs))

    # Define a worker function that processes a single window
    def process_window(i, start_idx, signal, win_samples, sampling_rate, theta, alpha_low, alpha_high, beta, gamma):
        end_idx = min(start_idx + win_samples, signal.shape[0])

        # Skip if window is too small
        if end_idx - start_idx < win_samples // 2:
            return (i, None, None, None)

        # Extract window - use first channel
        window = signal[start_idx:end_idx, 0]

        # ----- SPECTRAL FEATURES -----
        # Calculate PSD using Welch's method
        nperseg = min(len(window), 256)
        f, Pxx = welch(window, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg // 2)

        sum_Pxx = np.sum(Pxx)
        if sum_Pxx > 0:
            norm_psd = Pxx / sum_Pxx
            log_vals = np.log2(norm_psd + 1e-10)
            spectral_entropy = -np.sum(norm_psd * log_vals)
            spectral_centroid = np.sum(f * norm_psd)

            if spectral_centroid > 0:
                f_minus_centroid = f - spectral_centroid
                spectral_spread = np.sqrt(np.sum(f_minus_centroid ** 2 * norm_psd))

                if spectral_spread > 0:
                    spectral_skewness = np.sum((f_minus_centroid ** 3) * norm_psd) / (spectral_spread ** 3)
                else:
                    spectral_skewness = 0
            else:
                spectral_spread = spectral_skewness = 0
        else:
            spectral_entropy = spectral_centroid = spectral_spread = spectral_skewness = 0

        # Dominant frequency
        dominant_freq = f[np.argmax(Pxx)] if len(Pxx) > 0 else 0

        spectral_feats = [
            spectral_entropy,
            spectral_centroid,
            spectral_spread,
            spectral_skewness,
            dominant_freq
        ]

        # ----- TEMPORAL FEATURES -----
        # Pre-compute differences
        diff_window = np.diff(window)

        # Basic stats
        baseline = np.mean(window)
        activity = np.var(window)

        if activity > 0:
            var_diff = np.var(diff_window)
            mobility = np.sqrt(var_diff / activity)

            if var_diff > 0:
                complexity = np.sqrt(np.var(np.diff(diff_window)) / var_diff)
            else:
                complexity = 0
        else:
            mobility = complexity = 0

        # Zero crossing rate
        zcr = np.sum(np.diff(np.signbit(window)) != 0) / max(1, len(window) - 1)

        # Peak-to-peak
        ptp = np.ptp(window)

        temporal_feats = [activity, mobility, complexity, baseline, ptp]

        # ----- EMOTION-SPECIFIC RATIOS -----
        # Extract relevant band powers
        theta_i = theta[i, 0] if theta.ndim > 1 else theta[i]
        alpha_low_i = alpha_low[i, 0] if alpha_low.ndim > 1 else alpha_low[i]
        alpha_high_i = alpha_high[i, 0] if alpha_high.ndim > 1 else alpha_high[i]
        beta_i = beta[i, 0] if beta.ndim > 1 else beta[i]

        # Calculate ratios
        alpha_total = alpha_low_i + alpha_high_i

        # Safe division
        theta_beta_ratio = theta_i / beta_i if beta_i > 0 else 0
        alpha_beta_ratio = alpha_total / beta_i if beta_i > 0 else 0
        theta_alpha_ratio = theta_i / alpha_total if alpha_total > 0 else 0

        emotion_feats = [theta_beta_ratio, alpha_beta_ratio, theta_alpha_ratio]

        return (i, spectral_feats, temporal_feats, emotion_feats)

    # Process all windows in parallel
    results = Parallel(n_jobs=num_cores)(
        delayed(process_window)(
            i, start_indices[i], eeg_signal, win_samples,
            sampling_rate, theta, alpha_low, alpha_high, beta, gamma
        ) for i in range(n_windows)
    )

    # Collect results
    for i, spectral_feats, temporal_feats, emotion_feats in results:
        if spectral_feats is not None:  # Skip windows that were too small
            spectral_features[i] = spectral_feats
            temporal_features[i] = temporal_feats
            emotion_ratios[i] = emotion_feats

    # Prepare band power features
    band_powers = np.column_stack([
        theta.flatten() if theta.ndim > 1 else theta,
        alpha_low.flatten() if alpha_low.ndim > 1 else alpha_low,
        alpha_high.flatten() if alpha_high.ndim > 1 else alpha_high,
        beta.flatten() if beta.ndim > 1 else beta,
        gamma.flatten() if gamma.ndim > 1 else gamma
    ])

    # Calculate relative band powers
    total_power = np.sum(band_powers, axis=1, keepdims=True)
    # Avoid division by zero
    total_power[total_power == 0] = 1
    relative_powers = band_powers / total_power

    # Combine all features
    combined_features = np.column_stack([
        band_powers,  # Absolute band powers
        relative_powers,  # Relative band powers
        spectral_features,  # Spectral features
        temporal_features,  # Temporal features
        emotion_ratios  # Emotion-specific ratios
    ])

    # Create feature names for reference
    feature_names = [
        'theta_abs', 'alpha_low_abs', 'alpha_high_abs', 'beta_abs', 'gamma_abs',
        'theta_rel', 'alpha_low_rel', 'alpha_high_rel', 'beta_rel', 'gamma_rel',
        'spectral_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_skewness', 'dominant_freq',
        'activity', 'mobility', 'complexity', 'baseline', 'peak_to_peak',
        'theta_beta_ratio', 'alpha_beta_ratio', 'theta_alpha_ratio'
    ]

    return combined_features, feature_names


def compute_emotion_features_from_eeg(eeg_signals_flat, sampling_rate=1000.0, size=1.0, overlap=0.5, z_normalize=True,
                                      n_jobs=None):
    """
    Compute comprehensive EEG features optimized for emotion detection.

    Parameters
    ----------
    eeg_signals_flat : np.ndarray
        Raw EEG signal.
    sampling_rate : float, optional
        Sampling frequency (Hz).
    size : float, optional
        Window size (seconds).
    overlap : float, optional
        Window overlap (0 to 1).
    z_normalize : bool, optional
        Whether to z-normalize features.
    n_jobs : int, optional

    Returns
    -------
    features : array
        Matrix of extracted features.
    feature_names : list
        Names of the extracted features.
    """
    # Extract combined features
    features, feature_names = extract_combined_eeg_features(
        eeg_signals_flat,
        sampling_rate=sampling_rate,
        size=size,
        overlap=overlap,
        n_jobs=n_jobs
    )

    # Only normalize if needed and if we have valid data
    if z_normalize and features.size > 0:
        # Handle NaNs if present
        features = np.nan_to_num(features)

        # Custom robust z-scoring to avoid precision loss warnings
        # Calculate mean and std for each feature (column)
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)

        # Replace very small standard deviations with a minimum threshold
        # to avoid division by near-zero values
        eps = 1e-8
        stds[stds < eps] = 1.0  # Use 1.0 for features with near-zero variance

        # Apply z-score normalization manually
        features = (features - means) / stds

    return features, feature_names


def safe_stft(signal, fs=1000, nperseg=500, noverlap=250, window='hann', return_db=False):
    """
    A version of STFT that adjusts nperseg/noverlap if the signal is too short.
    Returns f, t, magnitude_spectrogram
    """
    length = len(signal)
    if length <= 1:
        # Return a dummy
        return np.array([]), np.array([]), np.array([[]])

    # Adjust nperseg if needed
    if length < nperseg:
        nperseg = length
    # Adjust overlap
    if noverlap >= nperseg:
        noverlap = max(0, nperseg // 2)

    f, t, Zxx = stft(
        signal,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None
    )
    Sxx = np.abs(Zxx)
    if return_db:
        Sxx = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx


def compute_eeg_bvp_spectrograms(
    eeg_signal,
    bvp_signal,
    fs_eeg=1000,
    fs_bvp=1000,
    window='hann',
    nperseg_eeg=500,
    noverlap_eeg=250,
    nperseg_bvp=500,
    noverlap_bvp=250,
    return_db=True
):
    """
    Compute magnitude (or dB) spectrograms for EEG and BVP signals using STFT.

    Args:
        eeg_signal (1D array): Raw EEG data, shape [T_eeg].
        bvp_signal (1D array): Raw BVP data, shape [T_bvp].
        fs_eeg (int): Sampling rate for EEG in Hz.
        fs_bvp (int): Sampling rate for BVP in Hz.
        window (str): Window function for both STFT calls.
        nperseg_eeg (int): Window size (samples) for EEG STFT.
        noverlap_eeg (int): Overlap (samples) for EEG STFT.
        nperseg_bvp (int): Window size (samples) for BVP STFT.
        noverlap_bvp (int): Overlap (samples) for BVP STFT.
        return_db (bool): If True, return log-scaled (dB) spectrogram.

    Returns:
        f_eeg, t_eeg, Sxx_eeg: freq array, time array, and spectrogram for EEG.
        f_bvp, t_bvp, Sxx_bvp: freq array, time array, and spectrogram for BVP.
    """
    f_eeg, t_eeg, Sxx_eeg = safe_stft(eeg_signal, fs=fs_eeg,
                                      nperseg=nperseg_eeg, noverlap=noverlap_eeg,
                                      window=window, return_db=return_db)
    f_bvp, t_bvp, Sxx_bvp = safe_stft(bvp_signal, fs=fs_bvp,
                                      nperseg=nperseg_bvp, noverlap=noverlap_bvp,
                                      window=window, return_db=return_db)
    return (f_eeg, t_eeg, Sxx_eeg), (f_bvp, t_bvp, Sxx_bvp)
