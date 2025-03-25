# preprocess.py
import os
import torch
import pandas as pd
import torchaudio
from tqdm import tqdm
import random
import numpy as np

from config import AUDIO_DIR, SPECTROGRAM_DIR, METADATA_PATH, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, LABEL_MAP, EXPERIMENT_PATH, N_WAY, N_SUPPORT, N_QUERY
from utils.audio_utils import load_audio, pad_or_trim, detect_noise_type, detect_overlapping_calls, generate_spectrogram_path

def generate_spectrogram_path(file_path):
    """
    Generates a spectrogram path 
    """
    # Extract subdirectory and filename
    relative_path = os.path.relpath(file_path, AUDIO_DIR)  # e.g., "train/alarm/file1.wav"
    spectrogram_path = os.path.join(SPECTROGRAM_DIR, os.path.splitext(relative_path)[0] + ".pt") # ensures .pt  format for PyTorch
    
    return spectrogram_path
def scan_and_update_metadata():
    """
    Scan audio directory, create spectrograms, and update metadata.
    """
    # Ensure the directory for metadata exists when reset
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    
    # Create metadata file if it doesn't exist or is empty, pandas dataframe
    if not os.path.exists(METADATA_PATH) or os.path.getsize(METADATA_PATH) == 0:
        print("Creating new metadata file...")
        pd.DataFrame(columns=['file_path', 'label', 'spectrogram_path', 'duration', 
                              'noise_type','prediction_confidence', 'prediction_correct']).to_csv(METADATA_PATH, index=False)
        # ensures required column exists
    
    # Load existing metadata
    try:
        metadata_df = pd.read_csv(METADATA_PATH)
        
        # Ensure DataFrame has required columns
        required_columns = ['file_path', 'label', 'spectrogram_path', 'duration', 
                            'noise_type','prediction_confidence', 'prediction_correct']
        for col in required_columns:
            if col not in metadata_df.columns:
                print(f"Adding missing column: {col}")
                metadata_df[col] = None
    #debug1
    except pd.errors.EmptyDataError:
        print("Metadata file is empty. Creating a new DataFrame.")
        metadata_df = pd.DataFrame(columns=['file_path', 'label', 'spectrogram_path', 'duration', 
                                            'noise_type','prediction_confidence', 'prediction_correct'])
    # debug2
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        metadata_df = pd.DataFrame(columns=['file_path', 'label', 'spectrogram_path', 'duration', 
                                            'noise_type','prediction_confidence', 'prediction_correct'])
    
    #  list of all audio files
    audio_files = []
    for root, _, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    # Check which files are not in metadata
    existing_files = set(metadata_df['file_path'].tolist() if 'file_path' in metadata_df.columns else [])
    new_files = [f for f in audio_files if f not in existing_files]
    
    if not new_files:
        print("No new audio files found.")
        return metadata_df
    
    print(f"Found {len(new_files)} new audio files. Processing...")
    
    # Process new files
    new_data = []
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    LABEL_MAPPING = {
        'train/alarm': 'alarm',
        'train/non_alarm': 'non_alarm',
        'train/background': 'background',
        'validation/alarm': 'alarm',
        'validation/non_alarm': 'non_alarm', 
        'validation/background': 'background',
        'test/alarm': 'alarm',
        'test/non_alarm': 'non_alarm',
        'test/background': 'background'
    }
    
    for file_path in tqdm(new_files):
        try:
            # Determine label first based on file path
            label = "unknown"
            for path_pattern, mapped_label in LABEL_MAPPING.items():
                if path_pattern in file_path:
                    label = mapped_label
                    break
            
            # Load audio
            waveform, sr = load_audio(file_path)
            if waveform is None:
                print(f"Skipping {file_path} - could not load audio")
                continue
            
            # Pad or trim, combining cut if necessary and right pad if necessary
            waveform = pad_or_trim(waveform)
            
            # Get duration
            duration = waveform.shape[1] / sr
            
            # Detect noise and overlapping
            noise_type = detect_noise_type(file_path)
            overlapping = detect_overlapping_calls(file_path)
            
            # Create spectrogram
            spec = mel_spectrogram(waveform)
            # Add a small constant and take log
            spec = torch.log(spec + 1e-9)

            # Generate path and ensure directory exists
            spectrogram_path = generate_spectrogram_path(file_path)
            os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)

            # Save spectrogram
            torch.save(spec, spectrogram_path)
            
            # Add to new data
            new_data.append({
                'file_path': file_path,
                'label': label,
                'spectrogram_path': spectrogram_path,
                'duration': duration,
                'noise_type': noise_type,
                'overlapping_calls': overlapping,
                'prediction_confidence' : "none",
                'prediction_correct': "none"
            })
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Add new data to metadata
    new_df = pd.DataFrame(new_data)
    metadata_df = pd.concat([metadata_df, new_df], ignore_index=True)
    
    # Save updated metadata
    metadata_df.to_csv(METADATA_PATH, index=False)
    
    print(f"Added {len(new_data)} new entries to metadata.")
    return metadata_df

def create_all_spectrograms(force_recreate=False):
    """
    Create spectrograms for all audio files in metadata, save in spectrogram directory
    
    Args:
        force_recreate: If True, recreate spectrograms even if they exist
    """
    if not os.path.exists(METADATA_PATH):
        print("Metadata file not found. Run scan_and_update_metadata first.")
        return
    
    metadata_df = pd.read_csv(METADATA_PATH)
    
    # Create mel spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        try:
            file_path = row['file_path']
            spectrogram_path = row['spectrogram_path']
            
            # Skip if spectrogram exists and force_recreate is False
            if os.path.exists(spectrogram_path) and not force_recreate:
                continue
            
            # Load and process audio
            waveform,sr = load_audio(file_path)
            if waveform is None:  # Check if loading failed
                print(f"Skipping {file_path} - could not load audio")
                continue
            # Pad or trim
            waveform = pad_or_trim(waveform)
            
            # Create spectrogram
            spec = mel_spectrogram(waveform)
            # Add a small constant and take log
            spec = torch.log(spec + 1e-9)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)
            
            # Save spectrogram
            torch.save(spec, spectrogram_path)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
def check_class_distribution(metadata_df):
    """
    Check the distribution of classes in the metadata.
    
    Args:
        metadata_df: DataFrame containing metadata
        
    Returns:
        Dictionary with class distribution statistics
    """
    if 'label' not in metadata_df.columns:
        return {"error": "No label column in metadata"}
    
    class_counts = metadata_df['label'].value_counts().to_dict()
    total = len(metadata_df)
    
    distribution = {
        "total_samples": total,
        "class_counts": class_counts,
        "class_percentages": {cls: count/total*100 for cls, count in class_counts.items()}
    }
    
    return distribution

def verify_few_shot_requirements(
    metadata_df, 
    n_way=N_WAY,  
    k_shot=N_SUPPORT, 
    query_size=N_QUERY, 
    test_size=3, 
    minimum_samples_per_class=5
):
    """
    Verify if the dataset meets few-shot requirements with 3 classes.
    """
    # debug the labels
    if 'label' not in metadata_df.columns:
        return {"meets_requirements": False, "error": "No label column in metadata"}
    
    # Get class counts
    class_counts = metadata_df['label'].value_counts().to_dict()
    
    # Check which classes have enough samples
    samples_needed = k_shot + query_size + test_size
    eligible_classes = []
    
    for cls, count in class_counts.items():
        if (count >= samples_needed and 
            count >= minimum_samples_per_class):
            eligible_classes.append(cls)
    
    # Specifically look for alarm, non_alarm, and background
    required_classes = ['alarm', 'non_alarm', 'background']
    has_required_classes = all(cls in eligible_classes for cls in required_classes)
    
    results = {
        "meets_requirements": has_required_classes,
        "eligible_classes": eligible_classes,
        "required_samples_per_class": samples_needed,
        "current_class_counts": class_counts
    }
    
    if not has_required_classes:
        results["suggestion"] = (
            f"Need {samples_needed} samples each for all three classes: "
            f"alarm, non_alarm, background. "
            f"Current class counts: {class_counts}"
        )
    
    return results
def prepare_few_shot_experiment(
    metadata_path=METADATA_PATH, 
    n_way=N_WAY,  # Explicitly set to 3 classes
    k_shot=N_SUPPORT, 
    query_size=N_QUERY, 
    test_size=3, 
    seed=42
):
    """
    Prepare data for a few-shot learning experiment with explicit class selection.
    Returns support set (k-shot examples per class)
    Returns query set (extra examples)
    Returns test set (for evaluation)
    Seed = starting value used for reproducibility
    
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Load metadata
    if not os.path.exists(metadata_path):
        return {"error": "Metadata file not found"}
    
    metadata_df = pd.read_csv(metadata_path)
    
    # Get class counts
    class_counts = metadata_df['label'].value_counts().to_dict()
    
    # Explicitly look for alarm, non_alarm, and background
    required_classes = ['alarm', 'non_alarm', 'background']
    
    # Check if all required classes exist and have enough samples
    eligible_classes = []
    samples_needed = k_shot + query_size + test_size
    
    for cls in required_classes:
        if cls in class_counts and class_counts[cls] >= samples_needed:
            eligible_classes.append(cls)
        else:
            print(f"Warning: {cls} class does not have enough samples")
    
    # Debug classes
    if len(eligible_classes) < n_way:
        raise ValueError(
            f"Not enough classes with sufficient samples. "
            f"Need {n_way}, have {len(eligible_classes)}. "
            f"Class counts: {class_counts}"
        )
    
    # Select classes (preferring to include all required classes if possible)
    selected_classes = required_classes[:n_way]
    
    # Prepare experiment setup
    experiment = {
        "n_way": len(selected_classes),
        "k_shot": k_shot,
        "query_size": query_size,
        "test_size": test_size,
        "selected_classes": selected_classes,
        "support_set": {},
        "query_set": {},
        "test_set": {}
    }
    
    # For each selected class, prepare support, query, and test sets
    for cls in selected_classes:
        # Get all samples for this class
        class_samples = metadata_df[metadata_df['label'] == cls]
        
        # Convert to list of dictionaries for easier handling
        samples = class_samples.to_dict('records')
        
        # Shuffle samples
        random.shuffle(samples)
        
        # Split into support, query, and test sets
        support_samples = samples[:k_shot]
        query_samples = samples[k_shot:k_shot+query_size]
        test_samples = samples[k_shot+query_size:k_shot+query_size+test_size]
        
        # Add to experiment
        experiment["support_set"][cls] = support_samples
        experiment["query_set"][cls] = query_samples
        experiment["test_set"][cls] = test_samples
    
    print(f"Few-shot experiment prepared with {len(selected_classes)} classes, {k_shot} shots.")
    print(f"Using classes: {selected_classes}")
    print(f"Total samples: {len(selected_classes) * (k_shot + query_size + test_size)} "
          f"({k_shot} support + {query_size} query + {test_size} test) × {len(selected_classes)} classes")
    
    return experiment
def save_few_shot_experiment(experiment, output_path):
    """
    Save a few-shot experiment configuration to a CSV file.
    
    Args:
        experiment: Dictionary containing experiment configuration
        output_path: Path to save the experiment configuration CSV
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare experiment data for CSV
    experiment_rows = []
    
    # Add metadata row
    metadata_row = {
        "experiment_type": "few_shot",
        "n_way": experiment["n_way"],
        "k_shot": experiment["k_shot"],
        "query_size": experiment["query_size"],
        "test_size": experiment["test_size"],
        "selected_classes": ",".join(experiment["selected_classes"]),
        "timestamp": pd.Timestamp.now()
    }
    experiment_rows.append(metadata_row)
    
    # Add samples for each set type
    for set_type in ["support_set", "query_set", "test_set"]:
        for cls, samples in experiment[set_type].items():
            for sample in samples:
                sample_row = sample.copy()
                sample_row.update({
                    "set_type": set_type,
                    "class": cls
                })
                experiment_rows.append(sample_row)
    
    # Create DataFrame
    experiment_df = pd.DataFrame(experiment_rows)
    
    # Save to CSV
    experiment_df.to_csv(output_path, index=False)
    
    print(f"Experiment saved to {output_path}")
    return experiment_df
def load_few_shot_experiment(config_path):
    """
    Load a few-shot experiment configuration from a CSV file.
    
    Args:
        config_path: Path to experiment configuration CSV
        
    Returns:
        Dictionary containing the experiment setup
    """
    # Read CSV
    experiment_df = pd.read_csv(config_path)
    
    # Extract metadata (first row)
    metadata = experiment_df.iloc[0]
    
    # Prepare experiment dictionary
    experiment = {
        "n_way": metadata["n_way"],
        "k_shot": metadata["k_shot"],
        "query_size": metadata["query_size"],
        "test_size": metadata["test_size"],
        "selected_classes": metadata["selected_classes"].split(","),
        "support_set": {},
        "query_set": {},
        "test_set": {}
    }
    
    # Separate samples by set type and class
    samples_df = experiment_df.iloc[1:]
    
    for set_type in ["support_set", "query_set", "test_set"]:
        set_samples = samples_df[samples_df["set_type"] == set_type]
        
        for cls in experiment["selected_classes"]:
            class_samples = set_samples[set_samples["class"] == cls]
            
            # Convert to list of dictionaries, dropping set_type and class columns
            experiment[set_type][cls] = class_samples.drop(columns=["set_type", "class"]).to_dict('records')
    
    print(f"Loaded experiment with {experiment['n_way']} classes, {experiment['k_shot']} shots.")
    return experiment


if __name__ == "__main__":
    # Scan for new audio files and update metadata
    metadata_df = scan_and_update_metadata()
    
    # Create spectrograms for all files
    create_all_spectrograms()
    
    # Check class distribution
    dist = check_class_distribution(metadata_df)
    print("Class distribution:")
    for cls, count in dist["class_counts"].items():
        print(f"  {cls}: {count} samples ({dist['class_percentages'][cls]:.2f}%)")
    
    # Verify few-shot requirements
    requirements = verify_few_shot_requirements(metadata_df)
    if requirements["meets_requirements"]:
        print("Dataset meets few-shot requirements.")
        print(f"Eligible classes: {requirements['eligible_classes']}")
    else:
        print("Dataset does not meet few-shot requirements.")
        print(requirements["suggestion"])
    
    # If requirements are met, prepare experiment
    if requirements["meets_requirements"]:

        experiment = prepare_few_shot_experiment()
        experiment_data = save_few_shot_experiment(experiment, EXPERIMENT_PATH)
        # Save experiment configuration
