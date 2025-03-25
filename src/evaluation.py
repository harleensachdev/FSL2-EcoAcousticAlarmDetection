# evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from config import N_WAY, N_SUPPORT, N_QUERY, METADATA_PATH, AUDIO_DIR
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
def classify_prototypical(embeddings, labels, query_embedding):
    """Classify using prototypical network approach."""
    unique_labels = torch.unique(labels)
    prototypes = []

    # Calculate prototype (mean) for each label
    for label in unique_labels:
        class_embeddings = embeddings[labels == label]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)

    prototypes = torch.stack(prototypes)

    # Make sure query_embedding has shape [1, embedding_size]
    if len(query_embedding.shape) > 2:
        query_embedding = query_embedding.reshape(1, -1)

    # Calculate euclidean distances between query and all prototypes
    dists = torch.cdist(query_embedding, prototypes)

    # Get the class with minimum distance
    min_idx = dists.flatten().argmin().item()
    pred_class = unique_labels[min_idx].item()

    return pred_class

def classify_with_relation(encoder, relation_net, support_embeddings, support_labels, query_embedding, device):
    """Classify using relation network."""
    unique_labels = torch.unique(support_labels)
    prototypes = []

    # Calculate prototype for each class
    for label in unique_labels:
        class_embeddings = support_embeddings[support_labels == label]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append((prototype, label.item()))

    # Compare query to each prototype
    max_score = -1
    predicted_class = None

    for prototype, label in prototypes:
        score = relation_net(
            query_embedding,
            prototype.unsqueeze(0)
        )

        if score.item() > max_score:
            max_score = score.item()
            predicted_class = label

    return predicted_class, max_score

def classify_with_ensemble(encoder, relation_net, support_embeddings, support_labels, query_embedding, 
                          proto_weight, rel_weight, device):
    """Classify using weighted ensemble of prototypical and relation networks."""
    unique_labels = torch.unique(support_labels)
    prototypes = []

    # Calculate prototype for each class
    for label in unique_labels:
        class_embeddings = support_embeddings[support_labels == label]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append((prototype, label.item()))
    
    # Stack prototypes for prototypical network
    proto_stack = torch.stack([p[0] for p in prototypes])
    
    # Prototypical network prediction
    dists = torch.cdist(query_embedding, proto_stack)
    proto_logits = -dists
    proto_probs = F.softmax(proto_logits, dim=1)
    
    # Relation network prediction
    rel_scores = torch.zeros(1, len(prototypes), device=device)
    for i, (prototype, _) in enumerate(prototypes):
        score = relation_net(query_embedding, prototype.unsqueeze(0))
        rel_scores[0, i] = score
    
    # Normalize relation scores
    rel_probs = rel_scores / rel_scores.sum(dim=1, keepdim=True)
    
    # Combine predictions
    ensemble_probs = proto_weight * proto_probs + rel_weight * rel_probs
    
    # Get predicted class
    _, pred_idx = ensemble_probs.max(1)
    pred_class = unique_labels[pred_idx.item()].item()
    confidence = ensemble_probs[0, pred_idx.item()].item()
    
    return pred_class, confidence

def evaluate_model(encoder, relation_net, test_dataloader, device, proto_accuracy=0.5, rel_accuracy=0.5, verbose=True):
    """Evaluate the model on test data using accuracy-based ensemble."""
    encoder.eval()
    relation_net.eval()

    # Get support embeddings and labels
    support_embeddings, support_labels = get_embeddings(encoder, test_dataloader, device)

    correct = 0
    total = 0

    # Calculate weights based on training accuracy
    proto_weight = proto_accuracy / (proto_accuracy + rel_accuracy) if (proto_accuracy + rel_accuracy) > 0 else 0.5
    rel_weight = rel_accuracy / (proto_accuracy + rel_accuracy) if (proto_accuracy + rel_accuracy) > 0 else 0.5

    if verbose:
        print(f"Using weighted ensemble: Prototypical weight = {proto_weight:.2f}, Relation weight = {rel_weight:.2f}")
    
    results = []
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            # Add channel dimension if needed
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            
            labels = labels.to(device)
            batch_size = inputs.size(0)

            # Get embeddings
            query_embeddings = encoder(inputs, return_embedding=True)

            # Classify each query
            for i in range(batch_size):
                query_emb = query_embeddings[i].unsqueeze(0)
                true_label = labels[i].item()

                # Prototypical classification
                proto_pred = classify_prototypical(support_embeddings, support_labels, query_emb)

                # Relation classification
                rel_pred, rel_score = classify_with_relation(
                    encoder, relation_net, support_embeddings, support_labels, query_emb, device
                )

                # Weighted decision
                if proto_weight > rel_weight and proto_weight > 0.6:
                    pred = proto_pred
                    confidence = f"Using prototypical (weight={proto_weight:.2f})"
                elif rel_weight > proto_weight and rel_weight > 0.6:
                    pred = rel_pred
                    confidence = f"Using relation (weight={rel_weight:.2f}, score={rel_score:.2f})"
                else:
                    # Use ensemble for close weights
                    pred, conf = classify_with_ensemble(
                        encoder, relation_net, support_embeddings, support_labels, 
                        query_emb, proto_weight, rel_weight, device
                    )
                    confidence = f"Using ensemble (confidence={conf:.2f})"

                if pred == true_label:
                    correct += 1
                total += 1
                
                result = {
                    "sample_idx": total,
                    "true_label": true_label,
                    "prediction": pred,
                    "correct": pred == true_label,
                    "proto_pred": proto_pred, 
                    "rel_pred": rel_pred,
                    "rel_score": rel_score,
                    "decision": confidence
                }
                results.append(result)
                
                if verbose:
                    print(f"Prediction {total}: {pred} (true: {true_label}) - {confidence}")

    accuracy = correct / total * 100
    if verbose:
        print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy, results
def update_metadata_results(results, test_dataset=None):
    """
    Adds results from evaluation to metadata.csv
    
    Args:
        results: List of dictionaries containing evaluation results
        test_dataset: The test dataset that was used for evaluation
    """
    import os
    import pandas as pd
    from config import METADATA_PATH
    
    # Ensure the directory for metadata exists
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    
    # Load existing metadata
    try:
        metadata_df = pd.read_csv(METADATA_PATH)
        
        # Ensure DataFrame has required columns
        required_columns = ['file_path', 'label', 'spectrogram_path', 'duration', 
                            'noise_type', 'overlapping_calls', 'prediction_confidence', 'prediction_correct']
        for col in required_columns:
            if col not in metadata_df.columns:
                print(f"Adding missing column: {col}")
                metadata_df[col] = None
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        metadata_df = pd.DataFrame(columns=['file_path', 'label', 'spectrogram_path', 'duration', 
                                           'noise_type', 'overlapping_calls', 'prediction_confidence', 'prediction_correct'])
    
    # Get the file paths from the test dataset
    test_files = []
    
    if test_dataset is not None and hasattr(test_dataset, 'metadata'):
        # If we have access to the original metadata in the dataset
        test_files = test_dataset.metadata['file_path'].tolist()
        print(f"Found {len(test_files)} files in test dataset")
    else:
        # Fallback: get all test files from metadata
        test_files = metadata_df[metadata_df['file_path'].str.contains('test/', na=False)]['file_path'].tolist()
        print(f"Using {len(test_files)} test files from metadata (may not match evaluation order)")
    
    # Make sure we have the right number of results
    if len(results) != len(test_files):
        print(f"Warning: Number of results ({len(results)}) doesn't match number of test files ({len(test_files)})")
        print("This may lead to incorrect mapping of results to files")
        
        # If we have more test files than results, truncate the list
        if len(test_files) > len(results):
            test_files = test_files[:len(results)]
        # If we have more results than test files, we'll only use the first len(test_files) results
    
    # Update metadata with results
    updated_count = 0
    for i, result in enumerate(results):
        if i < len(test_files):
            file_path = test_files[i]
            
            # Update prediction confidence and correctness
            confidence = result.get("decision", "none")
            correct = result.get("correct", "none")
            
            # Find the row in the original metadata and update it
            idx = metadata_df[metadata_df['file_path'] == file_path].index
            if not idx.empty:
                metadata_df.loc[idx, 'prediction_confidence'] = confidence
                metadata_df.loc[idx, 'prediction_correct'] = correct
                updated_count += 1
            else:
                print(f"Warning: File {file_path} not found in metadata")
    
    # Save updated metadata
    metadata_df.to_csv(METADATA_PATH, index=False)
    
    print(f"Updated prediction results for {updated_count} test files in metadata.")
    return metadata_df