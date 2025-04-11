import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
from config import N_WAY, N_SUPPORT, N_QUERY, METADATA_PATH, EPISODES, LEARNING_RATE, PROTO_WEIGHT, RELATION_WEIGHT, LABEL_MAP
from src.dataset import EpisodicDataLoader
def evaluate_episodic(model, test_dataset, device, n_way=N_WAY, n_support=N_SUPPORT, n_query=N_QUERY, n_episodes=EPISODES):
    """
    Evaluate model using episodic few-shot learning paradigm
    
    Args:
        model: Model with encoder and relation_net components
        test_dataset: Dataset for testing
        device: Computation device
        n_way: Number of classes per episode
        n_support: Number of support examples per class
        n_query: Number of query examples per class
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Accuracy, detailed results with filenames
    """
    
    # Create episodic dataloader for test set
    episodic_loader = EpisodicDataLoader(
        dataset=test_dataset,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        episodes=n_episodes
    )
    
    model.eval()
    correct = 0
    total = 0
    all_results = []
    
    with torch.no_grad():
        for support_set, query_set in tqdm(episodic_loader, desc="Evaluating"):
            # Unpack support and query sets
            support_data, support_labels = support_set
            query_data, query_labels = query_set
            
            # Move to device
            support_data = support_data.to(device)
            support_labels = support_labels.to(device)
            query_data = query_data.to(device)
            query_labels = query_labels.to(device)
            
            # Get indices from the sampler to access original file paths
            query_indices = []
            episode_indices = list(episodic_loader.sampler._current_indices) if hasattr(episodic_loader.sampler, '_current_indices') else []
            if episode_indices:
                support_size = n_way * n_support
                query_indices = episode_indices[support_size:]
            
            episode_classes = []
            for i in range(n_way):
                class_indices = torch.where(support_labels % n_way == i)[0]
                if class_indices.size(0) > 0:
                    actual_class = support_labels[class_indices[0]].item()
                    episode_classes.append(actual_class)
            
            # Add channel dimension if needed
            if support_data.dim() == 3:
                support_data = support_data.unsqueeze(1)
            if query_data.dim() == 3:
                query_data = query_data.unsqueeze(1)
            
            # Get encodings
            support_embeddings = model.encoder(support_data, return_embedding=True)
            query_embeddings = model.encoder(query_data, return_embedding=True)
            
            # Compute prototypes for each class
            prototypes = []
            for i in range(n_way):
                class_indices = torch.where(support_labels % n_way == i)[0]
                class_prototypes = support_embeddings[class_indices].mean(0)
                prototypes.append(class_prototypes)
            prototypes = torch.stack(prototypes)
            
            # Evaluate each query sample
            for i, query_embedding in enumerate(query_embeddings):
                true_label = query_labels[i] % n_way
                
                # Get file path for this query sample if available
                file_path = None
                if i < len(query_indices) and query_indices[i] < len(test_dataset.data):
                    file_path = test_dataset.data.iloc[query_indices[i]]['file_path']

                # Prototypical prediction
                dists = torch.cdist(query_embedding.unsqueeze(0), prototypes)
                proto_logits = -dists.squeeze(0)
                proto_probs = F.softmax(proto_logits, dim=0)
                proto_pred = torch.argmax(proto_probs).item()
                
                # Relation network prediction
                rel_scores = torch.zeros(n_way, device=device)
                for j in range(n_way):
                    # Create pair of query embedding and prototype
                    relation_pair = torch.cat([
                        query_embedding.unsqueeze(0), 
                        prototypes[j].unsqueeze(0)
                    ], dim=1)
                    rel_scores[j] = model.relation_net(relation_pair)
                
                rel_pred = torch.argmax(rel_scores).item()
                
                combined_probs = PROTO_WEIGHT * proto_probs + RELATION_WEIGHT * F.softmax(rel_scores, dim=0)
                pred_class = torch.argmax(combined_probs).item()
                pred_actual_class = episode_classes[pred_class] if pred_class < len(episode_classes) else -1
                confidence = combined_probs[pred_class].item()
                
                # Track results
                correct += (pred_class == true_label.item())
                total += 1
                
                result = {
                    'file_path': file_path,
                    'true_label': true_label.item(),
                    'prediction': pred_class,
                    'actual_prediction': pred_actual_class,  # Actual class label
                    'proto_pred': proto_pred,
                    'rel_pred': rel_pred,
                    'confidence': confidence,
                    'correct': (pred_class == true_label.item())
                }
                all_results.append(result)
    
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy, all_results


def update_metadata_results(
    results: List[Dict], 
    test_dataset=None,
    metadata_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Updates prediction results in metadata CSV file using string labels.
    
    Args:
        results: List of dictionaries containing evaluation results with file_path
        test_dataset: Optional dataset (not required if results contain file paths)
        metadata_path: Path to the main metadata CSV file
    
    Returns:
        Updated metadata DataFrame
    """
    # Use default path if not provided
    metadata_path = metadata_path or METADATA_PATH
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    # Read metadata file
    metadata_df = pd.read_csv(metadata_path)
    
    # Create reverse mapping from numeric to string labels
    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    
    # Update metadata with results
    updated_count = 0
    
    for result in results:
        # Get the file path directly from the result
        file_path = result.get("file_path")
        if not file_path:
            print("Warning: Result is missing file_path")
            continue
        
        # Get prediction data
        confidence = result.get("confidence", 0.0)
        correct = result.get("correct", False)
        
        # Get the numeric prediction
        prediction_num = result.get("actual_prediction", -1)
        
        # Map numeric prediction to string label using the reverse mapping
        prediction_str = REVERSE_LABEL_MAP.get(prediction_num, "unknown")

        
        # Update metadata DataFrame
        metadata_mask = (metadata_df['file_path'] == file_path)
        if metadata_mask.any():
            metadata_df.loc[metadata_mask, 'prediction_confidence'] = confidence
            metadata_df.loc[metadata_mask, 'prediction_correct'] = correct
            metadata_df.loc[metadata_mask, 'prediction'] = prediction_str  # Store string label
            updated_count += 1
        else:
            print(f"Warning: File {file_path} not found in metadata CSV")
    
    # Save updated CSV
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"Updated prediction results")    
    return metadata_df