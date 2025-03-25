import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import random
from typing import Dict, List, Optional, Tuple, Union

from config import (
    METADATA_PATH, SAMPLE_RATE, NUM_SAMPLES, 
    N_FFT, HOP_LENGTH, N_MELS, LABEL_MAP, 
    BATCH_SIZE, DEVICE,
    SPECTROGRAM_DIR, N_QUERY, N_SUPPORT, N_WAY
)
from utils.audio_utils import load_audio, pad_or_trim, generate_spectrogram_path

class BirdSoundDataset(Dataset):
    def __init__(
        self, 
        metadata_df: pd.DataFrame, 
        transformation: Optional[torch.nn.Module] = None, 
        load_spectrograms: bool = True, 
        device: str = DEVICE
    ):
        """
        dataset for bird sounds with metadata handling.
        
        Args:
            metadata_df: DataFrame containing metadata
            transformation: Optional spectrogram transformation
            load_spectrograms: Whether to load pre-computed spectrograms
            device: Device to load tensors to
        """
        self.device = device
        self.load_spectrograms = load_spectrograms
        
        # Validate and preprocess metadata
        self._validate_metadata(metadata_df)
        
        # Set up transformation
        self.transformation = self._setup_transformation(transformation)
        
        # Prepare label mapping and indices
        self._prepare_labels()
        
    def _validate_metadata(self, metadata_df: pd.DataFrame):
        """Validate and prepare metadata."""
        # Ensure required columns exist
        required_columns = ['file_path', 'label']
        for col in required_columns:
            if col not in metadata_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Filter out invalid labels
        self.metadata = metadata_df.copy()
        
    def _setup_transformation(self, transformation):
        """Set up spectrogram transformation."""
        if transformation is None:
            return torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            ).to(self.device)
        return transformation.to(self.device)
    
    def _prepare_labels(self):
        """Prepare label mapping and indices."""
        # Determine label mapping
        if not LABEL_MAP:
            unique_labels = sorted(self.metadata['label'].unique())
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_map = LABEL_MAP
        
        # Extract numerical labels
        self.labels = [self.label_map.get(label, -1) for label in self.metadata['label']]
        
        # Mask that marks valid labels ars true (if unknown, label would be -1 and filtered out)
        valid_mask = [label != -1 for label in self.labels]
        self.metadata = self.metadata[valid_mask].reset_index(drop=True)
        self.labels = [label for label, valid in zip(self.labels, valid_mask) if valid]
        
        # Validate labels
        if not self.labels:
            raise ValueError("No valid labels found. Check LABEL_MAP and metadata.")
        
        # Determine classes
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single audio sample and its label.
        
        Args:
            idx: Index of the sample to load
        
        Returns:
            Tensor of audio spectrogram and its corresponding label
        """
        file_path = self.metadata.loc[idx, 'file_path']
        label = self.labels[idx]
        
        if self.load_spectrograms and 'spectrogram_path' in self.metadata.columns:
            spectrogram_path = self.metadata.loc[idx, 'spectrogram_path']
            spectrogram = torch.load(spectrogram_path)
        else:
            waveform = load_audio(file_path, sample_rate=SAMPLE_RATE)
            waveform = pad_or_trim(waveform, num_samples=NUM_SAMPLES)
            spectrogram = self.transformation(waveform)
        
        return spectrogram.to(self.device), torch.tensor(label).to(self.device)

    def get_class_indices(self) -> Dict[int, List[int]]:
        """Get indices of samples for each class."""
        return {
            cls: [i for i, label in enumerate(self.labels) if label == cls]
            for cls in self.classes
        }

def load_metadata(metadata_path: str = METADATA_PATH) -> pd.DataFrame:
    """
    Load and validate metadata from CSV file.
    
    Args:
        metadata_path: Path to metadata CSV file
    
    Returns:
        Validated metadata DataFrame
    """
    try:
        df = pd.read_csv(metadata_path)
        
        if df.empty:
            raise ValueError(f"Metadata file at {metadata_path} is empty")
        
        required_columns = ['file_path', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    except (pd.errors.EmptyDataError, FileNotFoundError) as e:
        raise ValueError(f"Error loading metadata: {e}")

def create_few_shot_datasets(
    metadata_df: pd.DataFrame,
    n_way: int = N_WAY,
    k_shot: int = N_SUPPORT,
    query_size: int = N_QUERY,
    test_size: int = 3,
    seed: int = 42
) -> Tuple[BirdSoundDataset, BirdSoundDataset, BirdSoundDataset, List[int], Dict[str, int]]:
    """
    Create datasets for N-way, K-shot learning with flexible class selection.
    
    Args:
        metadata_df: Input metadata DataFrame
        n_way: Number of classes to select
        k_shot: Number of support samples per class
        query_size: Number of query samples per class
        test_size: Number of test samples per class
        seed: Random seed for reproducibility
    
    Returns:
        support_dataset, query_dataset, test_dataset, valid_classes, selected_label_map
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create master dataset
    full_dataset = BirdSoundDataset(metadata_df)
    class_indices = full_dataset.get_class_indices()
    
    # Select top N most populated classes with enough samples
    class_populations = {cls: len(indices) for cls, indices in class_indices.items()}
    valid_classes = [
        cls for cls, count in class_populations.items() 
        if count >= k_shot + query_size + test_size
    ][:n_way]
    
    if len(valid_classes) < n_way:
        raise ValueError(
            f"Not enough classes with sufficient samples. "
            f"Need {n_way}, have {len(valid_classes)}. "
            f"Class populations: {class_populations}"
        )
    
    # Maps class to numeric labels (reversed to map numeric to class)
    reverse_label_map = {v: k for k, v in full_dataset.label_map.items()}
    selected_label_map = {
        reverse_label_map[cls]: i 
        for i, cls in enumerate(valid_classes)
    }
    
    # Prepare dataset indices
    support_indices, query_indices, test_indices = [], [], []
    
    for cls in valid_classes:
        cls_indices = class_indices[cls].copy()
        random.shuffle(cls_indices)
        
        support_indices.extend(cls_indices[:k_shot])
        query_indices.extend(cls_indices[k_shot:k_shot + query_size])
        test_indices.extend(cls_indices[k_shot + query_size:k_shot + query_size + test_size])
    
    # Create subsets
    support_df = metadata_df.iloc[support_indices].reset_index(drop=True)
    # drop index to remove gaps
    query_df = metadata_df.iloc[query_indices].reset_index(drop=True)
    test_df = metadata_df.iloc[test_indices].reset_index(drop=True)
    
    # Create datasets
    support_dataset = BirdSoundDataset(support_df)
    query_dataset = BirdSoundDataset(query_df)
    test_dataset = BirdSoundDataset(test_df)
    
    return support_dataset, query_dataset, test_dataset, valid_classes, selected_label_map

def create_few_shot_dataloaders(
    support_dataset: BirdSoundDataset, 
    query_dataset: BirdSoundDataset, 
    test_dataset: BirdSoundDataset, 
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for few-shot learning.
    
    Args:
        support_dataset: Support set dataset
        query_dataset: Query set dataset
        test_dataset: Test set dataset
        batch_size: Batch size
    
    Returns:
        support_loader, query_loader, test_loader
    """
    support_loader = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return support_loader, query_loader, test_loader

