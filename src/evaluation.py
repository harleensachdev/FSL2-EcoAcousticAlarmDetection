# fsl-2 evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
from config import N_WAY, N_SUPPORT, N_QUERY, METADATA_PATH, EPISODES, LEARNING_RATE, PROTO_WEIGHT, RELATION_WEIGHT, LABEL_MAP, EVALUATEDATAPATH,REQUIRED_CLASSES
from src.dataset import EpisodicDataLoader
from datetime import datetime


def extract_time_from_filename(filename):
    """
    Extract time information from filename.
    Assumes filename contains time in format like 'YYYYMMDD_HHMMSS' or similar patterns.
    
    Args:
        filename: The filename to extract time from
        
    Returns:
        time_key: A string representing the time (e.g., 'YYYY-MM-DD_HH')
    """
    # Remove path and extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Try to extract datetime patterns
    patterns = [
        r'(\d{8})_(\d{6})',  # YYYYMMDD_HHMMSS
        r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD_HH-MM-SS
        r'(\d{4}\d{2}\d{2})_(\d{2}\d{2}\d{2})',  # YYYYMMDD_HHMMSS
    ]
    
    for pattern in patterns:
        match = re.search(pattern, base_name)
        if match:
            date_part = match.group(1)
            time_part = match.group(2)
            
            # Parse date
            if len(date_part) == 8:  # YYYYMMDD
                year = date_part[:4]
                month = date_part[4:6]
                day = date_part[6:8]
                date_str = f"{year}-{month}-{day}"
            else:  # Already formatted
                date_str = date_part
            
            # Parse time (just get hour)
            if len(time_part) >= 2:
                hour = time_part[:2]
                return f"{date_str}_{hour}"
    
    # If no pattern matches, return the filename without extension as fallback
    return base_name


def update_segment_class_counts_with_time_aggregation(experiment_df, results):
    """
    Update segment class counts in experiment DataFrame with time-based aggregation.
    When multiple files have the same time, their counts are averaged.
    
    Args:
        experiment_df: DataFrame with experiment files
        results: List of dictionaries with prediction results
        
    Returns:
        experiment_df: Updated DataFrame with time-aggregated counts
    """
    print(f"Processing {len(results)} results with time-based aggregation...")
    
    # Group results by original file path
    file_predictions = {}
    
    for result in results:
        file_path = result.get('file_path', '')
        if not file_path:
            continue
            
        # Extract the base filename to match with experiment data
        if '_seg' in file_path:
            base_filename = os.path.basename(file_path)
            parts = base_filename.split('_seg')
            if len(parts) > 1:
                base_part = parts[0]
                if '.' in parts[-1]:
                    ext = '.' + parts[-1].split('.')[-1] if '.' in parts[-1] else ''
                    base_filename = base_part + ext
                else:
                    base_filename = base_part
        else:
            base_filename = os.path.basename(file_path)
        
        # Map the prediction to class name using reverse label mapping
        prediction_num = result.get('prediction', -1)
        REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
        prediction = REVERSE_LABEL_MAP.get(prediction_num, "unknown")
        
        if base_filename not in file_predictions:
            file_predictions[base_filename] = {'alarm': 0, 'non_alarm': 0, 'background': 0}
        
        if prediction in file_predictions[base_filename]:
            file_predictions[base_filename][prediction] += 1
        else:
            print(f"Warning: Unknown prediction '{prediction}' for file {base_filename}")
    
    print(f"Grouped results into {len(file_predictions)} unique files")
    
    # Add time extraction and file path information to experiment_df
    experiment_df['time_key'] = experiment_df['file_path'].apply(
        lambda x: extract_time_from_filename(x)
    )
    experiment_df['base_filename'] = experiment_df['file_path'].apply(
        lambda x: os.path.basename(x)
    )
    
    # First, update individual file counts
    for idx, row in experiment_df.iterrows():
        file_path = row['file_path']
        base_filename = os.path.basename(file_path)
        base_filename_no_ext = os.path.splitext(base_filename)[0]
        
        if base_filename in file_predictions:
            counts = file_predictions[base_filename]
            experiment_df.at[idx, 'alarm_count'] = counts['alarm']
            experiment_df.at[idx, 'non_alarm_count'] = counts['non_alarm']
            experiment_df.at[idx, 'background_count'] = counts['background']
        elif base_filename_no_ext in file_predictions:
            counts = file_predictions[base_filename_no_ext]
            experiment_df.at[idx, 'alarm_count'] = counts['alarm']
            experiment_df.at[idx, 'non_alarm_count'] = counts['non_alarm']
            experiment_df.at[idx, 'background_count'] = counts['background']
        else:
            # Try to find a match with different extension
            found_match = False
            for pred_filename in file_predictions.keys():
                if os.path.splitext(pred_filename)[0] == base_filename_no_ext:
                    counts = file_predictions[pred_filename]
                    experiment_df.at[idx, 'alarm_count'] = counts['alarm']
                    experiment_df.at[idx, 'non_alarm_count'] = counts['non_alarm']
                    experiment_df.at[idx, 'background_count'] = counts['background']
                    found_match = True
                    break
            
            if not found_match:
                print(f"Warning: Could not find predictions for file {base_filename}")
                # Set default values
                experiment_df.at[idx, 'alarm_count'] = 0
                experiment_df.at[idx, 'non_alarm_count'] = 0
                experiment_df.at[idx, 'background_count'] = 0
    
    # Now perform time-based aggregation
    print("Performing time-based aggregation...")
    
    # Group by time_key and calculate mean counts
    time_groups = experiment_df.groupby('time_key').agg({
        'alarm_count': 'mean',
        'non_alarm_count': 'mean',
        'background_count': 'mean'
    }).round(1)  # Round to 1 decimal place
    
    # Add aggregated columns
    experiment_df['alarm_count_time_avg'] = experiment_df['time_key'].map(time_groups['alarm_count'])
    experiment_df['non_alarm_count_time_avg'] = experiment_df['time_key'].map(time_groups['non_alarm_count'])
    experiment_df['background_count_time_avg'] = experiment_df['time_key'].map(time_groups['background_count'])
    
    # Count how many files contribute to each time period
    time_file_counts = experiment_df.groupby('time_key').size()
    experiment_df['files_per_time'] = experiment_df['time_key'].map(time_file_counts)
    
    # Print aggregation summary
    print(f"\nTime-based aggregation summary:")
    print(f"Total unique time periods: {len(time_groups)}")
    
    # Show examples of aggregation
    multiple_files_times = time_file_counts[time_file_counts > 1]
    if len(multiple_files_times) > 0:
        print(f"Time periods with multiple files: {len(multiple_files_times)}")
        print("\nExamples of aggregated time periods:")
        for time_key in multiple_files_times.head(3).index:
            files_at_time = experiment_df[experiment_df['time_key'] == time_key]
            print(f"\nTime {time_key} ({len(files_at_time)} files):")
            for _, file_row in files_at_time.iterrows():
                print(f"  {file_row['base_filename']}: "
                      f"[{file_row['alarm_count']}, {file_row['non_alarm_count']}, {file_row['background_count']}]")
            avg_counts = time_groups.loc[time_key]
            print(f"  Average: [{avg_counts['alarm_count']}, {avg_counts['non_alarm_count']}, {avg_counts['background_count']}]")
    else:
        print("No time periods with multiple files found.")
    
    # Return the updated DataFrame
    return experiment_df


def create_time_aggregated_summary(experiment_df):
    """
    Create a summary DataFrame with one row per time period showing aggregated counts.
    
    Args:
        experiment_df: DataFrame with time-aggregated results
        
    Returns:
        summary_df: DataFrame with one row per time period
    """
    # Group by time and get unique values for each time period
    summary_data = []
    
    for time_key in experiment_df['time_key'].unique():
        time_group = experiment_df[experiment_df['time_key'] == time_key]
        
        # Get the first row as representative (since aggregated values are the same for all files in the time group)
        first_row = time_group.iloc[0]
        
        summary_row = {
            'time_key': time_key,
            'num_files': len(time_group),
            'alarm_count_avg': first_row['alarm_count_time_avg'],
            'non_alarm_count_avg': first_row['non_alarm_count_time_avg'],
            'background_count_avg': first_row['background_count_time_avg'],
            'total_segments_avg': (first_row['alarm_count_time_avg'] + 
                                 first_row['non_alarm_count_time_avg'] + 
                                 first_row['background_count_time_avg']),
            'files': ', '.join(time_group['base_filename'].tolist())
        }
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('time_key')
    
    return summary_df


def evaluate_ensemble_classification(model, segment_dataset, support_dataset, device, n_way=3, n_support=5, batch_size=32):
    """
    Evaluate segments using the ensemble model with relation network approach adapted for fsl-2
    """
    model.eval()
    
    print("Preparing support set for ensemble evaluation...")
    
    support_data_by_class = {}
    for i, (spectrogram, label) in enumerate(support_dataset):
        if label not in support_data_by_class:
            support_data_by_class[label] = []
        if len(support_data_by_class[label]) < n_support:
            support_data_by_class[label].append(spectrogram)
    
    if len(support_data_by_class) < n_way:
        raise ValueError(f"Support dataset has only {len(support_data_by_class)} classes, need {n_way}")
    
    class_labels = sorted(support_data_by_class.keys())[:n_way]
    support_images = []
    support_labels = []
    
    class_to_idx = {class_label: idx for idx, class_label in enumerate(class_labels)}
    
    for class_label in class_labels:
        class_spectrograms = support_data_by_class[class_label][:n_support]
        support_images.extend(class_spectrograms)
        support_labels.extend([class_to_idx[class_label]] * len(class_spectrograms))
    
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    
    print(f"Support set classes: {class_labels}")
    print(f"Support labels mapping: {class_to_idx}")
    
    # Compute prototypes for prototypical network part
    if support_images.dim() == 3:
        support_images = support_images.unsqueeze(1)
    
    with torch.no_grad():
        support_embeddings = model.encoder(support_images, return_embedding=True)
        
        # Compute prototypes for each class
        prototypes = []
        for i in range(n_way):
            class_indices = torch.where(support_labels == i)[0]
            if len(class_indices) > 0:
                class_prototypes = support_embeddings[class_indices].mean(0)
                prototypes.append(class_prototypes)
        prototypes = torch.stack(prototypes)
    
    segment_loader = DataLoader(
        segment_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    all_results = []
    segment_idx = 0
    
    with torch.no_grad():
        for batch_spectrograms, _ in tqdm(segment_loader, desc="Evaluating with ensemble"):
            batch_spectrograms = batch_spectrograms.to(device)
            if batch_spectrograms.dim() == 3:
                batch_spectrograms = batch_spectrograms.unsqueeze(1)
            
            # Get embeddings for query samples
            query_embeddings = model.encoder(batch_spectrograms, return_embedding=True)
            
            for i in range(len(batch_spectrograms)):
                current_idx = segment_idx + i
                file_path = segment_dataset.get_file_path(current_idx)
                
                query_embedding = query_embeddings[i]
                
                # Prototypical prediction
                dists = torch.cdist(query_embedding.unsqueeze(0), prototypes)
                proto_logits = -dists.squeeze(0)
                proto_probs = F.softmax(proto_logits, dim=0)
                
                # Relation network prediction
                rel_scores = torch.zeros(n_way, device=device)
                for j in range(n_way):
                    # Create pair of query embedding and prototype
                    relation_pair = torch.cat([
                        query_embedding.unsqueeze(0), 
                        prototypes[j].unsqueeze(0)
                    ], dim=1)
                    rel_scores[j] = model.relation_net(relation_pair)
                
                # Combine predictions
                combined_probs = PROTO_WEIGHT * proto_probs + RELATION_WEIGHT * F.softmax(rel_scores, dim=0)
                predicted_idx = torch.argmax(combined_probs).item()
                confidence = combined_probs[predicted_idx].item()
                
                if predicted_idx < len(class_labels):
                    actual_class_label = class_labels[predicted_idx]
                    reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
                    predicted_label_str = reverse_label_map.get(actual_class_label, 'unknown')
                else:
                    actual_class_label = -1
                    predicted_label_str = 'unknown'
                
                result = {
                    'file_path': file_path,
                    'prediction': predicted_idx,
                    'actual_prediction': actual_class_label,
                    'confidence': confidence,
                    'correct': None
                }
                all_results.append(result)
            
            segment_idx += len(batch_spectrograms)
    
    return all_results


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
    model.eval()
    all_results = []
    
    # Create a DataLoader for batch processing
    data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load metadata for training data to get support samples
    train_metadata = pd.read_csv(METADATA_PATH)
    
    # Create a dictionary to store samples by class
    class_samples = {}
    for cls in REQUIRED_CLASSES:
        cls_metadata = train_metadata[train_metadata['label'] == cls]
        class_samples[cls] = []
        
        # Load the first n_support samples for each class
        for idx, row in cls_metadata.head(n_support).iterrows():
            try:
                spec_path = row['spectrogram_path']
                if os.path.exists(spec_path):
                    spec = torch.load(spec_path)
                    class_samples[cls].append(spec)
            except Exception as e:
                print(f"Error loading support sample {spec_path}: {e}")
    
    # Convert to tensors and move to device
    support_data = []
    support_labels = []
    
    for cls_idx, cls in enumerate(REQUIRED_CLASSES):
        for spec in class_samples[cls]:
            if spec.dim() == 2:  # Add channel dimension if needed
                spec = spec.unsqueeze(0)
            support_data.append(spec)
            support_labels.append(cls_idx)
    
    support_data = torch.stack(support_data).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    
    # Process the support set once to get prototypes
    with torch.no_grad():
        # Get encodings
        support_embeddings = model.encoder(support_data, return_embedding=True)
        
        # Compute prototypes for each class
        prototypes = []
        for i in range(n_way):
            class_indices = torch.where(support_labels == i)[0]
            if len(class_indices) > 0:
                class_prototypes = support_embeddings[class_indices].mean(0)
                prototypes.append(class_prototypes)
        prototypes = torch.stack(prototypes)
    
    # Process all query samples (evaluation segments)
    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(tqdm(data_loader, desc="Evaluating segments")):
            # Move batch to device
            batch_data = batch_data.to(device)
            if batch_data.dim() == 3:
                batch_data = batch_data.unsqueeze(1)  # Add channel dimension
            
            # Get embeddings
            query_embeddings = model.encoder(batch_data, return_embedding=True)
            
            # Process each embedding in the batch
            for i, query_embedding in enumerate(query_embeddings):
                # Get file path for this sample
                sample_idx = batch_idx * data_loader.batch_size + i
                file_path = test_dataset.get_file_path(sample_idx)
                
                if file_path is None:
                    continue
                
                # Prototypical prediction
                dists = torch.cdist(query_embedding.unsqueeze(0), prototypes)
                proto_logits = -dists.squeeze(0)
                proto_probs = F.softmax(proto_logits, dim=0)
                
                # Relation network prediction
                rel_scores = torch.zeros(n_way, device=device)
                for j in range(n_way):
                    # Create pair of query embedding and prototype
                    relation_pair = torch.cat([
                        query_embedding.unsqueeze(0), 
                        prototypes[j].unsqueeze(0)
                    ], dim=1)
                    rel_scores[j] = model.relation_net(relation_pair)
                
                # Combine predictions
                combined_probs = PROTO_WEIGHT * proto_probs + RELATION_WEIGHT * F.softmax(rel_scores, dim=0)
                pred_class = torch.argmax(combined_probs).item()
                confidence = combined_probs[pred_class].item()
                
                # Store result
                result = {
                    'file_path': file_path,
                    'prediction': pred_class,
                    'confidence': confidence,
                }
                all_results.append(result)
    
    return all_results 


def update_segment_class_counts(experiment_df, results):
    """
    Update segment class counts in experiment DataFrame based on evaluation results.
    
    Args:
        experiment_df: DataFrame with experiment files
        results: List of dictionaries with prediction results
    """
    # Group results by original file path
    file_predictions = {}
    for result in results:
        file_path = result.get('file_path', '')
        if not file_path:
            continue
            
        # Extract the base filename to match with experiment data
        segment_filename = os.path.basename(file_path)
        
        # Extract the original file identifier (without _segXX.pt)
        # For example, from "SMM05537-BG2_20221105_081000_seg01.pt" 
        # we want to extract "SMM05537-BG2_20221105_081000"
        match = re.search(r'([\w\d]+-[\w\d]+_\d{8}_\d{6})(?:_seg\d+)?\.pt', segment_filename)
        if match:
            original_id = match.group(1)
        else:
            # If pattern doesn't match, try a simpler approach
            original_id = segment_filename.split('_seg')[0]
        
        # Map the prediction to class name using reverse label mapping
        prediction_num = result.get('prediction', -1)
        REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
        prediction = REVERSE_LABEL_MAP.get(prediction_num, "unknown")
        
        if original_id not in file_predictions:
            file_predictions[original_id] = {'alarm': 0, 'non_alarm': 0, 'background': 0}
        
        # Increment the count for this class
        if prediction in file_predictions[original_id]:
            file_predictions[original_id][prediction] += 1
    
    # Update counts in experiment DataFrame
    updated_count = 0
    for idx, row in experiment_df.iterrows():
        file_path = row['file_path']
        file_basename = os.path.basename(file_path)
        
        # Extract the identifier part without extension
        original_id = os.path.splitext(file_basename)[0]
        
        if original_id in file_predictions:
            counts = file_predictions[original_id]
            experiment_df.at[idx, 'alarm_count'] = counts['alarm']
            experiment_df.at[idx, 'non_alarm_count'] = counts['non_alarm']
            experiment_df.at[idx, 'background_count'] = counts['background']
            updated_count += 1
    
    print(f"Updated class counts for {updated_count} files")
    experiment_df.to_csv(EVALUATEDATAPATH, index=False)
    return experiment_df


def update_metadata_results(
    results: List[Dict], 
    test_dataset=None,
    metadata_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Updates prediction results in metadata CSV file using string labels.
    """
    metadata_path = metadata_path or METADATA_PATH
    
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    try:
        metadata_df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"Warning: Metadata file {metadata_path} not found. Creating new one.")
        metadata_df = pd.DataFrame(columns=['file_path', 'label', 'prediction', 'prediction_confidence', 'prediction_correct'])
    
    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    
    for col in ['prediction', 'prediction_confidence', 'prediction_correct']:
        if col not in metadata_df.columns:
            metadata_df[col] = None
    
    updated_count = 0
    
    for result in results:
        file_path = result.get("file_path")
        if not file_path:
            print("Warning: Result is missing file_path")
            continue
        
        confidence = result.get("confidence", 0.0)
        correct = result.get("correct", False)
        
        prediction_str = result.get("prediction")
        if not prediction_str:
            prediction_num = result.get("actual_prediction", -1)
            prediction_str = REVERSE_LABEL_MAP.get(prediction_num, "unknown")
        
        metadata_mask = (metadata_df['file_path'] == file_path)
        if metadata_mask.any():
            metadata_df.loc[metadata_mask, 'prediction_confidence'] = confidence
            metadata_df.loc[metadata_mask, 'prediction_correct'] = correct
            metadata_df.loc[metadata_mask, 'prediction'] = prediction_str
            updated_count += 1
        else:
            print(f"Warning: File {file_path} not found in metadata CSV")
    
    try:
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Updated prediction results for {updated_count} files")
    except Exception as e:
        print(f"Error saving metadata: {e}")
    
    return metadata_df