import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from typing import List, Optional

from config import (LABEL_MAP
)
class BirdSoundDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Initialize the dataset with a pandas DataFrame
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing spectrogram paths and labels
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data = dataframe
        self.label_map = LABEL_MAP
        
    def __len__(self):
        """
        Return the total number of samples in the dataset
        
        Returns:
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            tuple: (spectrogram tensor, label integer)
        """
        # Ensure idx is an integer (handles both integer and pandas index)
        if not isinstance(idx, int):
            idx = idx.item() if hasattr(idx, 'item') else int(idx)
        
        # Load the spectrogram
        spectrogram_path = self.data.iloc[idx]['spectrogram_path']
        spectrogram = torch.load(spectrogram_path)
        
        # Get the label and convert to integer
        label_str = self.data.iloc[idx]['label']
        label_int = self.label_map[label_str]
        
        
        return spectrogram, label_int

class EpisodicBatchSampler(Sampler):
    """Sampler that yields batches of indices for eposodic training
    ARGS:
      labels(List[int]: List of class labels)
      n_way
      n_support
      n_query
      episdoes"""
    
    def __init__(self,
                 labels: List[int],
                 n_way: int,
                 n_support: int,
                 n_query: int,
                 episodes: int,
                 fixed_classes: Optional[List[int]] = None):
                 
        self.labels = labels
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.episodes = episodes
        self._current_indices = []

        self.unique_labels = list(set(labels))
        
        if len(self.unique_labels) < self.n_way:
            raise ValueError(f"Dataset has only {len(self.unique_labels)} classes, but n_way={self.n_way}")
        
        self.label_indices = {}
        # dictionary of indices for each label
        for label in self.unique_labels:
            self.label_indices[label] = [i for i, l in enumerate(labels) if l == label] 

        required_samples = self.n_support + self.n_query
        for label in self.unique_labels:
            if len(self.label_indices[label]) < required_samples:
                raise ValueError(f"Class {label} has only {len(self.label_indices[label])} samples, "
                                f"but requires {required_samples} (n_support + n_query)")
    
    def __len__(self):
        return self.episodes
    
    def __iter__(self):
        for _ in range(self.episodes):
            episode_classes = self.unique_labels
            # select n way classes (should be 3)

            support_indices = []
            query_indices = []

            for cls in episode_classes:
                # indices for this class
                clsindices = self.label_indices[cls].copy()
                random.shuffle(clsindices)

                # split to support + query
                support_indices.extend(clsindices[:self.n_support])
                query_indices.extend(clsindices[self.n_support: self.n_support + self.n_query])

            # combine support + query indices (support first)
            episode_indices = support_indices + query_indices
            self._current_indices = episode_indices  # Store current indices
            yield episode_indices

class EpisodicDataLoader:
    """Wrapper to create data loader"""

    def __init__(self, 
                 dataset: Dataset, 
                 n_way: int, 
                 n_support: int, 
                 n_query: int, 
                 episodes: int,
                 fixed_classes: Optional[List[int]] = None):
        self.dataset = dataset
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.episodes = episodes 

        self.labels = [label for _, label in self.dataset]
        self.sampler = EpisodicBatchSampler(
            labels = self.labels,
            n_way = n_way,
            n_support = n_support,
            n_query = n_query,
            episodes=episodes,
            fixed_classes = fixed_classes
        )

        self.loader = DataLoader(dataset = self.dataset, batch_sampler = self.sampler, num_workers = 0, pin_memory = True)

    def __iter__(self):
        """Return episodes in format"""
        for batch_indices in self.sampler:
            # Store current indices in the sampler for later use
            self.sampler._current_indices = batch_indices
            
            batch = [self.dataset[i] for i in batch_indices]
            data = [item[0] for item in batch]
            labels = [item[1] for item in batch]

            # convert totensor
            data = torch.stack(data)
            labels = torch.tensor(labels)

            # split to  support + qury on indices
            support_size = self.n_way * self.n_support
            support_data = data[:support_size]
            support_labels = labels[:support_size]
            query_data = data[support_size:]
            query_labels = labels[support_size:]
            
            # Yield support + label dataloaders for this episode
            yield(support_data, support_labels), (query_data, query_labels)

    def __len__(self):
        return len(self.sampler)
