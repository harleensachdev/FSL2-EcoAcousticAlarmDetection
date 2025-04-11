# # preprocess.py
import os
import torch
import pandas as pd
import torchaudio
from tqdm import tqdm
from config import AUDIO_DIR, SPECTROGRAM_DIR, METADATA_PATH, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_WAY, N_SUPPORT, N_QUERY, TEST_SIZE, REQUIRED_CLASSES
from utils.audio_utils import load_audio, pad_or_trim, generate_spectrogram_path

def generate_spectrogram_path(file_path):
    """Generate a spectrogram path"""
    # Audio path for specific audio file from main AUDIO_DIR, eg., "train/alarm/file1.wav"
    relative_audio_path = os.path.relpath(file_path, AUDIO_DIR)
    # Replace .wav/.mp3 etc. with .pt extension
    spectrogram_filename = os.path.splitext(relative_audio_path)[0] + ".pt"
    spectrogram_path = os.path.join(SPECTROGRAM_DIR, spectrogram_filename)
    return spectrogram_path
def getmetadata():
    """
    1. Scan through audio directory
    2. Check for any spectrograms that have not been created
    3. Update metadata (adding rows)"""
    
    # Ensure metadata directory exists
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

    # Create metadata file if empty/not exist
    if not os.path.exists(METADATA_PATH) or os.path.getsize(METADATA_PATH) == 0:
        print("Creating a new metadata file")
        # Correctly create an empty DataFrame with specified columns
        metadata_df = pd.DataFrame(columns=['file_path', 'label', 'spectrogram_path', 'duration', 'prediction_confidence', 'prediction', 'prediction_correct'])
        # Save the empty DataFrame
        metadata_df.to_csv(METADATA_PATH, index=False)
    else:
        # Load existing metadata file
        metadata_df = pd.read_csv(METADATA_PATH)
    # List out all audio files
    audio_files = []
    # os.walk generates file names in a directory tree (top down/bottom up), output root,dirs, files (dirs not needed)
    for root, _, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    existing_files = set(metadata_df['file_path'].to_list() if 'file_path' in metadata_df.columns else [])
    # error handle blank file paths

    new_files = [f for f in audio_files if f not in existing_files]
    if len(new_files) == 0:
        print("No new audio files")
        return metadata_df
    else:
        print(f"Found {len(new_files)} new audio files to process")

    # Process any new files
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
    

# tqdm for cool progress bar 
    for file_path in tqdm(new_files):
        try:
            "Determine label by file path"
            label = "unknown"
            for part1, part2 in LABEL_MAPPING.items():
                if part1 in file_path:
                    label = part2
                    break
            
            # Load audio
            waveform, sr = load_audio(file_path)
            if waveform is None:
                print(f"Skipping {file_path}, could not load audio")
                continue

            # Pad or trim, cut or right pad
            waveform = pad_or_trim(waveform)

            # Get duration
            duration = waveform.shape[1] / sr

            # Create the spectrogram
            spectrogram = mel_spectrogram(waveform)
            # Add small constant, take log, evenly distribute + linearize data
            spectrogram = torch.log(spectrogram + 1e-9)

            # Generate paths, ensure spectro dir exists
            spectrogram_path = generate_spectrogram_path(file_path)
            os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)

            # Save spectrogram
            torch.save(spectrogram, spectrogram_path)

            # Append to new data

            new_data.append({
                'file_path': file_path,
                'label': label,
                'spectrogram_path': spectrogram_path,
                'duration': duration,
                'prediction_confidence' : "none",
                'prediction_correct': "none"
            })
            
        except Exception as e:
            print(f"Error processing {file_path} : {e}")

    # Add new data to a new dataframe
    new_df = pd.DataFrame(new_data)
    # Concatenate new)Data into existing metadata_df
    metadata_df = pd.concat([metadata_df, new_df], ignore_index=True)

    # Save updated metadata
    metadata_df.to_csv(METADATA_PATH, index=False)
    return metadata_df


def create_all_spectrograms(force_recreate=False):
    """
    Create spectrograms for all audio files in metadata, save in spectrogram directory
    
    Args:
        force_recreate: If True, recreate spectrograms even if they exist
    """
    if not os.path.exists(METADATA_PATH):
        print("Metadata file not found. Run getmetadata first to create the csv")
        return
    
    metadata_df = pd.read_csv(METADATA_PATH)

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
            # Add a small constant and take log, for linearity
            spec = torch.log(spec + 1e-9)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)
            
            # Save spectrogram
            torch.save(spec, spectrogram_path)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

def check_class_distribution(metadata_df):
    """Check the distribution of classes in metadata
    metadata_df: Dataframe containing metadata
    Returns a dictionary with class distribution stats for confirmation"""
    if 'label' not in metadata_df.columns:
        return {"error": "No label column in metadata"}
    class_counts = metadata_df['label'].value_counts().to_dict()
    total = 0
    for cl, count in class_counts.items():
        total += int(count)
    distribution = {
        "total_samples": total,
        "class_counts": class_counts,
        "class_percentages": {cls: count/total*100 for cls, count in class_counts.items()}
    }
    
    return distribution

def verify_few_shot_requirements(metadata_df, n_way=N_WAY, k_shot=N_SUPPORT, query_size=N_QUERY, test_size=TEST_SIZE):
    """
    Verify the latest dataset meets few shot requirements to prevent future errors
    
    Checks:
    1. Total samples per class
    2. Sufficient samples for support set from train directory
    3. Sufficient samples for query set from train directory
    4. Sufficient samples for test set from test directory
    """
    if 'label' not in metadata_df.columns or 'file_path' not in metadata_df.columns:
        return {"error": "Missing label or file_path column in metadata"}
    
    # Total samples needed per class
    total_samples_needed = k_shot + query_size + test_size
    
    # Detailed verification results
    verification_results = {
        "meets_requirements": True,
        "class_details": {}
    }
    
    for cls in REQUIRED_CLASSES:
        # Separate train and test samples
        train_samples = metadata_df[
            (metadata_df['label'] == cls) & 
            (metadata_df['file_path'].str.contains('train/'))
        ]
        test_samples = metadata_df[
            (metadata_df['label'] == cls) & 
            (metadata_df['file_path'].str.contains('test/'))
        ]
        
        # Verify support set samples from train directory
        support_samples = train_samples.head(k_shot)
        if len(support_samples) < k_shot:
            verification_results["meets_requirements"] = False
            verification_results["class_details"][cls] = {
                "train_samples": len(train_samples),
                "support_samples": len(support_samples),
                "support_samples_needed": k_shot,
                "error": f"Insufficient train support samples. Need {k_shot}, have {len(support_samples)}"
            }
            continue
        
        # Verify query set samples from train directory
        query_samples = train_samples.iloc[k_shot:k_shot+query_size]
        if len(query_samples) < query_size:
            verification_results["meets_requirements"] = False
            verification_results["class_details"][cls] = {
                "train_samples": len(train_samples),
                "query_samples": len(query_samples),
                "query_samples_needed": query_size,
                "error": f"Insufficient train query samples. Need {query_size}, have {len(query_samples)}"
            }
            continue
        
        # Verify test set samples from test directory
        test_samples_subset = test_samples.head(test_size)
        if len(test_samples_subset) < test_size:
            verification_results["meets_requirements"] = False
            verification_results["class_details"][cls] = {
                "test_samples": len(test_samples),
                "test_samples_subset": len(test_samples_subset),
                "test_samples_needed": test_size,
                "error": f"Insufficient test samples. Need {test_size}, have {len(test_samples_subset)}"
            }
            continue
        
        # If we've made it this far, this class passes
        verification_results["class_details"][cls] = {
            "train_total_samples": len(train_samples),
            "test_total_samples": len(test_samples),
            "support_samples": len(support_samples),
            "query_samples": len(query_samples),
            "test_samples": len(test_samples_subset),
            "status": "PASS"
        }
    
    # If any class failed, provide a suggestion
    if not verification_results["meets_requirements"]:
        verification_results["suggestion"] = (
            f"Need {k_shot} support samples, {query_size} query samples from train directories, "
            f"and {test_size} test samples from test directories for all three classes: "
            "alarm, non_alarm, background. "
            "Check the class_details for specific requirements."
        )
    
    return verification_results