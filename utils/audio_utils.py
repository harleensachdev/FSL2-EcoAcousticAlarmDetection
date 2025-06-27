# audio_utils.py
import torch
import torchaudio
import os
import sys
import librosa
import numpy as np
import pandas as pd
from config import SAMPLE_RATE, NUM_SAMPLES, SPECTROGRAM_DIR
from typing import Optional, Tuple
import re

def parse_filename(file_path):
    """Extract site, date and time from file name"""
    filename = os.path.basename(file_path)
    # Updated pattern to match SMM05537-BG2_20221105_081000 format
    pattern = r'([\w\d]+-[\w\d]+)_(\d{8})_(\d{6})'
    match = re.search(pattern, filename)
    if match:
        site = match.group(1)  # SMM05537-BG2
        date = match.group(2)  # 20221105
        time = match.group(3)  # 081000
        return site, date, time
    else:
        return "unknown", "unknown", "unknown"
    
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
def trim_to_60_seconds(signal, sample_rate = SAMPLE_RATE):
    target_length = 60 * sample_rate

    if signal is None:
        return None
    
    length = signal.shape[1]
    
    # pad if too short
    if length < target_length:
        pad = target_length - target_length
        return torch.nn.functional.pad(signal, (0, pad))
    
    # trim if too long
    elif length > target_length:
        return signal[:, :target_length]
    
    else:
        return signal

def split_into_1sec_segments(signal, sample_rate = SAMPLE_RATE):
    """Split 60 second audio clip into 1 second segment"""
    if signal is None:
        return []
    
    segment_length = sample_rate # 1 second samples
    num_segments = signal.shape[1] // segment_length

    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = signal[:, start:end]
        segments.append(segment)

    return segments
def pad_or_trim(signal: torch.Tensor, target_length: int = NUM_SAMPLES) -> torch.Tensor:
    """
    Pad or trim an audio signal to a target length.
    
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
    
    # Trim if too long (from beginning)
    elif length > target_length:
        return signal[:, :target_length]
    
    # Return as-is if exactly the right length
    else:
        return signal

def generate_spectrogram_path(file_path, segment_idx=None, spectrogram_dir=None):
    """Generate the corresponding spectrogram path for an audio file or segment."""
    from config import SPECTROGRAM_DIR
    # Use provided directory or default from config
    spec_dir = spectrogram_dir or SPECTROGRAM_DIR
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Add segment index if provided
    if segment_idx is not None:
        spectrogram_filename = f"{base_name}_seg{segment_idx:02d}.pt"
    else:
        spectrogram_filename = f"{base_name}.pt"
    spectrogram_path = os.path.join(spec_dir, spectrogram_filename)
    return spectrogram_path