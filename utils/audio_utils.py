import torch
import torchaudio
import os
import sys
import librosa
import numpy as np
import pandas as pd
from config import SAMPLE_RATE, NUM_SAMPLES, SPECTROGRAM_DIR

def load_audio(file_path):
    """Load an audio file with proper resampling."""
    try:
        signal, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            signal = resampler(signal)
            
        return signal, SAMPLE_RATE
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def pad_or_trim(signal, target_length=NUM_SAMPLES):
    """Pad or trim an audio signal to target length."""
    if signal is None:
        return None
        
    length = signal.shape[1]
    
    # Trim if longer than target length
    if length > target_length:
        signal = signal[:, :target_length]
    
    # Pad if shorter than target length    
    elif length < target_length:
        padding = target_length - length
        signal = torch.nn.functional.pad(signal, (0, padding))
        
    return signal

def detect_noise_type(file_path, threshold=0.7):
    """Simplified noise type detection (placeholder for more sophisticated analysis)."""
    signal, sr = load_audio(file_path)
    if signal is None:
        return "unknown"
        
    # Simple analysis for demonstration - in practice use more sophisticated methods
    signal = signal.numpy().flatten()
    
    # Check RMS energy
    rms = np.sqrt(np.mean(signal**2))
    
    # Check zero-crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(signal).astype(int))))
    zcr = zero_crossings / len(signal)
    
    if rms > threshold and zcr < 0.05:
        return "rain"
    elif zcr > 0.1:
        return "wind"
    elif np.max(signal) > 0.9:
        return "anthropogenic"
    else:
        return "none"
        
def detect_overlapping_calls(file_path):
    """Simplified detection of overlapping bird calls (placeholder)."""
    # In practice, this would use more sophisticated techniques 
    # such as multiple source separation or onset detection
    
    signal, sr = load_audio(file_path)
    if signal is None:
        return "none"
        
    # Simple analysis for demonstration
    signal = signal.numpy().flatten()
    
    # Spectral flatness as a rough proxy for complexity
    spec = np.abs(librosa.stft(signal))
    flatness = librosa.feature.spectral_flatness(S=spec)
    
    if np.mean(flatness) > 0.3:
        return "high"
    elif np.mean(flatness) > 0.15:
        return "medium"
    else:
        return "low"

def generate_spectrogram_path(audio_path):
    """Generate the corresponding spectrogram path for an audio file."""
    filename = os.path.basename(audio_path)
    base_name = os.path.splitext(filename)[0]
    return os.path.join(SPECTROGRAM_DIR, f"{base_name}.pt")