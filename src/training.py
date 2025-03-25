import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PROTO_WEIGHT, RELATION_WEIGHT, N_WAY, N_SUPPORT, N_QUERY

def get_embeddings(model, data_loader, device):
    """Extract embeddings for all samples in the dataset."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            
            # Add channel dimension if needed
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
                
            embeddings = model(inputs, return_embedding=True)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    return torch.cat(all_embeddings), torch.cat(all_labels).to(device)

def create_data_loader(dataset, batch_size):
    """Create a DataLoader from a dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class EpisodeDataset(Dataset):
    """Custom dataset for few-shot learning episodes."""
    def __init__(self, data, labels):
        """
        Args:
            data (torch.Tensor): Input data tensor
            labels (torch.Tensor): Corresponding labels tensor
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_episode_dataset(data, labels, n_way, n_support, n_query):
    """
    Create a dataset for few-shot learning episodes with flexible sampling.
    
    Args:
        data (torch.Tensor): Full dataset
        labels (torch.Tensor): Corresponding labels
        n_way (int): Number of classes
        n_support (int): Number of support samples per class
        n_query (int): Number of query samples per class
    
    Returns:
        Tuple of support and query sets
    """
    # Ensure labels are converted to standard tensor type
    labels = labels.view(-1)
    
    # Find unique classes and their counts
    unique_classes, class_counts = torch.unique(labels, return_counts=True)
    
    print("Debugging create_episode_dataset:")
    print(f"Total classes: {len(unique_classes)}")
    print(f"Class counts: {dict(zip(unique_classes.tolist(), class_counts.tolist()))}")
    
    # Validate inputs
    if len(unique_classes) < n_way:
        raise ValueError(f"Not enough unique classes. Need {n_way}, have {len(unique_classes)}")
    
    support_data = []
    support_labels = []
    query_data = []
    query_labels = []

    for cls in unique_classes[:n_way]:
        # Get indices for current class
        cls_indices = torch.nonzero(labels == cls).squeeze()
        
        # Validate indices
        if cls_indices.numel() == 0:
            raise ValueError(f"No indices found for class {cls}")
        
        # Determine actual support and query sizes based on available samples
        available_samples = len(cls_indices)
        
        # Adjust support and query sizes if not enough samples
        actual_support = min(n_support, available_samples // 2)
        actual_query = min(n_query, available_samples - actual_support)
        
        # If not enough samples, raise a more informative error
        if actual_support + actual_query > available_samples:
            raise ValueError(f"Class {cls} has insufficient samples. "
                             f"Needed: {n_support + n_query}, "
                             f"Available: {available_samples}, "
                             f"Using: {actual_support} support, {actual_query} query")
        
        # Randomly permute indices
        perm = torch.randperm(len(cls_indices))
        cls_indices = cls_indices[perm]
        
        # Split into support and query sets
        support_idx = cls_indices[:actual_support]
        query_idx = cls_indices[actual_support:actual_support+actual_query]
        
        print(f"Class {cls}: support_idx length {len(support_idx)}, query_idx length {len(query_idx)}")
        
        # Add support samples
        for idx in support_idx:
            support_data.append(data[idx])
            support_labels.append(cls)
        
        # Add query samples
        for idx in query_idx:
            query_data.append(data[idx])
            query_labels.append(cls)
    
    # Validate before stacking
    if not query_data:
        raise ValueError("No query data was generated. Check sample selection logic.")
    
    # Convert to tensors
    support_data = torch.stack(support_data)
    support_labels = torch.tensor(support_labels)
    query_data = torch.stack(query_data)
    query_labels = torch.tensor(query_labels)

    return (support_data, support_labels), (query_data, query_labels)

def create_ensemble_loss_fn(n_way=N_WAY, n_support=N_SUPPORT, n_query=N_QUERY, 
                             proto_weight=0.5, relation_weight=0.5):
    """
    Create a combined loss function for the ensemble model
    
    Args:
        n_way (int): Number of classes in the episode
        n_support (int): Number of support samples per class
        n_query (int): Number of query samples per class
        proto_weight (float): Weight for prototypical loss
        relation_weight (float): Weight for relation network loss
    
    Returns:
        A callable loss function for few-shot learning
    """
    proto_criterion = nn.NLLLoss()
    relation_criterion = nn.MSELoss()
    
    def ensemble_loss(model, support_set, query_set):
        """
        Compute the ensemble loss for few-shot learning
        
        Args:
            model: EnsembleModel instance
            support_set: (support_data, support_labels)
            query_set: (query_data, query_labels)
        
        Returns:
            Combined loss value
        """
        device = next(model.parameters()).device
        
        # Unpack support and query sets
        support_data, support_labels = support_set
        query_data, query_labels = query_set
        
        # Ensure correct dimensions
        support_data = support_data.to(device).unsqueeze(1)
        query_data = query_data.to(device).unsqueeze(1)
        support_labels = support_labels.to(device)
        query_labels = query_labels.to(device)
        
        # Get embeddings
        support_embeddings = model.encoder(support_data, return_embedding=True)
        query_embeddings = model.encoder(query_data, return_embedding=True)
        
        # Calculate prototypes
        prototypes = torch.zeros(n_way, support_embeddings.shape[1]).to(device)
        for i in range(n_way):
            mask = support_labels == torch.unique(support_labels)[i]
            prototypes[i] = support_embeddings[mask].mean(dim=0)
        
        # Prototypical loss: eucalidean dist
        dists = torch.cdist(query_embeddings, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)
        
        target_inds = torch.zeros(len(query_labels), dtype=torch.long).to(device)
        for i, label in enumerate(query_labels):
            target_inds[i] = torch.nonzero(torch.unique(support_labels) == label).item()
        
        proto_loss = proto_criterion(log_p_y, target_inds)
        
        # Relation network loss
        relation_scores = []
        relation_targets = []
        
        for i, query_emb in enumerate(query_embeddings):
            query_label = query_labels[i]
            
            for j, prototype in enumerate(prototypes):
                proto_label = torch.unique(support_labels)[j]
                target = 1.0 if query_label == proto_label else 0.0
                
                score = model.relation_net(query_emb.unsqueeze(0), prototype.unsqueeze(0))
                relation_scores.append(score)
                relation_targets.append(target)
        
        relation_scores = torch.cat(relation_scores)
        relation_targets = torch.tensor(relation_targets, dtype=torch.float).to(device).unsqueeze(1)
        relation_loss = relation_criterion(relation_scores, relation_targets)
        
        # Combined loss
        total_loss = proto_weight * proto_loss + relation_weight * relation_loss
        
        return total_loss
    
    return ensemble_loss

def train_few_shot(model, data_or_loader, labels_or_none=None, epochs=10, n_way=N_WAY, n_support=N_SUPPORT, n_query=N_QUERY, **kwargs):
    """
    Train the few-shot ensemble model
    
    Args:
        model: EnsembleModel instance
        data_or_loader: Either full input data tensor or a DataLoader
        labels_or_none: Labels tensor (only if first argument is a tensor)
        epochs: Number of training epochs
        n_way: Number of classes per episode
        n_support: Number of support samples per class
        n_query: Number of query samples per class
        **kwargs: Additional arguments for loss function creation (optimizer)
    
    Returns:
        List of training losses
    """
    # If a DataLoader is passed, extract data and labels
    if isinstance(data_or_loader, DataLoader):
        data = []
        labels = []
        for batch_data, batch_labels in data_or_loader:
            data.append(batch_data)
            labels.append(batch_labels)
        
        data = torch.cat(data)
        labels = torch.cat(labels)
    else:
        # Assume tensor data is passed directly
        data = data_or_loader
        labels = labels_or_none

    # Create ensemble loss function
    loss_fn = create_ensemble_loss_fn(n_way=n_way, n_support=n_support, n_query=n_query, **kwargs)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_losses = []
    
    device = next(model.parameters()).device
    data = data.to(device)
    labels = labels.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Create an episode
        support_set, query_set = create_episode_dataset(
            data, labels, n_way=n_way, n_support=n_support, n_query=n_query
        )
        
        # Compute loss
        loss = loss_fn(model, support_set, query_set)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        train_losses.append(total_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    return train_losses