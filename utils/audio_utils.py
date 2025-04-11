import torch
import torchaudio
import os
import sys
import librosa
import numpy as np
import pandas as pd
from config import SAMPLE_RATE, NUM_SAMPLES, SPECTROGRAM_DIR
from typing import Optional, Tuple
import scipy.signal

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

def pad_or_trim(signal: torch.Tensor, target_length: int = NUM_SAMPLES) -> torch.Tensor:
    """
    Simple function to pad or trim an audio signal to a target length.
    
    Args:
        signal (torch.Tensor): Input audio signal
        target_length (int): Desired signal length (defaults to NUM_SAMPLES from config)
        
    Returns:
        torch.Tensor: Padded or trimmed audio signal
    """
    if signal is None:
        return None
        
    # Get the length of the signal
    length = signal.shape[1]
    
    # Pad if too short
    if length < target_length:
        padding = target_length - length
        return torch.nn.functional.pad(signal, (0, padding))
    
    # Trim if too long (take the middle segment)
    elif length > target_length:
        start = (length - target_length) // 2
        return signal[:, start:start + target_length]
    
    # Return as-is if exactly the right length
    else:
        return signal

def generate_spectrogram_path(audio_path):
    """Generate the corresponding spectrogram path for an audio file."""
    filename = os.path.basename(audio_path)
    base_name = os.path.splitext(filename)[0]
    return os.path.join(SPECTROGRAM_DIR, f"{base_name}.pt")