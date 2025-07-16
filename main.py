# fsl-2 main.py
import os
import torch
import sys
import traceback
import pandas as pd
import torchaudio
from torch.utils.data import DataLoader

# Add the directory containing the preprocessing script to the Python path
preprocessing_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(preprocessing_dir)

from config import (
    AUDIO_DIR,
    SPECTROGRAM_DIR,
    EVALUATEAUDIO_DIR,
    EVALUATEDATAPATH,
    BATCH_SIZE,
    DEVICE,
    N_SUPPORT,
    N_QUERY,
    TEST_SIZE,
    REQUIRED_CLASSES,
    N_WAY,
    EPISODES,
    PROTO_WEIGHT,
    RELATION_WEIGHT,
    LABEL_MAP,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS
)

# Import preprocessing and training functions
from src.preprocess import (
    getmetadata,
    create_all_spectrograms,
    check_class_distribution,
    verify_few_shot_requirements,
    getexperimentdata,
    process_audio_file
)
from src.dataset import BirdSoundDataset, SegmentDataset, EpisodicDataLoader
from src.models import CombinedFreqTemporalCNNEncoder, RelationNetwork, EnsembleModel
from src.training import train_few_shot
from src.evaluation import (
    filter_unprocessed_segments, 
    update_segment_class_counts,
    update_metadata_results,
    evaluate_ensemble_classification,
    update_segment_class_counts_with_time_aggregation,
    create_time_aggregated_summary
)

# Configuration flag for time aggregation
ENABLE_TIME_AGGREGATION = False  # Set to True to enable time aggregation

def preprocess_data():
    """
    Run preprocessing steps to prepare the dataset.
    """
    print("Starting preprocessing...")
    
    # Scan for new audio files and update metadata
    metadata_df = getmetadata()
    
    # Create spectrograms for all training files
    create_all_spectrograms()
    
    # Check class distribution
    dist = check_class_distribution(metadata_df)
    print("Class distribution:")
    for cls, count in dist["class_counts"].items():
        print(f" {cls}: {count} samples ({dist['class_percentages'][cls]:.2f}%)")
    
    return metadata_df

def preprocess_evaluation_data():
    """
    Prepare evaluation data by processing audio files into 1-second segments.
    Creates new entries for new files instead of overwriting existing data.
    """
    print("Preparing evaluation data...")
    
    # Get or create experiment metadata
    experiment_df = getexperimentdata()
    
    # Process any unprocessed files (this will create spectrograms for 1-second segments)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    # For any unprocessed files, process them into segments
    unprocessed_files = experiment_df[experiment_df['processed'] == False]
    print(f"Found {len(unprocessed_files)} unprocessed files")
    
    for idx, row in unprocessed_files.iterrows():
        try:
            file_path = row['file_path']
            print(f"Processing {file_path}...")
            _, segment_paths = process_audio_file(file_path, mel_spectrogram)
            
            if segment_paths:
                # Update paths in DataFrame
                experiment_df.at[idx, 'spectrogram_paths'] = ','.join(segment_paths)
                experiment_df.at[idx, 'processed'] = True
                print(f"Created {len(segment_paths)} segments for {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save updated DataFrame
    experiment_df.to_csv(EVALUATEDATAPATH, index=False)
    print(f"Updated evaluation data saved to {EVALUATEDATAPATH}")
    return experiment_df

def main():
    # Step 1: Create directories if they don't exist
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
    os.makedirs(EVALUATEAUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(EVALUATEDATAPATH), exist_ok=True)
    
    try:
        # Step 2: Run preprocessing for training data
        metadata_df = preprocess_data()
        
        # Step 3: Check if we have enough data for few-shot learning
        requirements = verify_few_shot_requirements(
            metadata_df,
            n_way=N_WAY,
            k_shot=N_SUPPORT,
            query_size=N_QUERY,
            test_size=TEST_SIZE
        )
        
        # Step 4: Prepare few-shot experiment
        if requirements["meets_requirements"]:
            # Filter metadata to include only required classes
            all_metadata = metadata_df[metadata_df['label'].isin(REQUIRED_CLASSES)]
            all_dataset = BirdSoundDataset(all_metadata)
            
            # Create support dataset from training data for prototype creation
            train_metadata = all_metadata[~all_metadata['file_path'].str.contains('test/')]
            support_dataset = BirdSoundDataset(train_metadata)
            
            # Step 5: Initialize models
            encoder = CombinedFreqTemporalCNNEncoder().to(DEVICE)
            relation_net = RelationNetwork().to(DEVICE)
            ensemble_model = EnsembleModel(encoder, relation_net).to(DEVICE)
            
            # Step 6: Train the model
            print("Starting training...")
            train_losses = train_few_shot(
                model=ensemble_model,
                dataset=all_dataset,
                episodes=EPISODES,
                n_way=N_WAY,
                n_support=N_SUPPORT,
                n_query=N_QUERY,
                relation_weight=RELATION_WEIGHT,
                proto_weight=PROTO_WEIGHT
            )
            
                # Step 7: Prepare evaluation data
            print("Preparing evaluation data...")
            experiment_df = preprocess_evaluation_data()
            
            # Create dataset of all 1-second segments for evaluation
            all_segment_paths = []
            for idx, row in experiment_df.iterrows():
                if row['processed'] and row['spectrogram_paths']:
                    segments = row['spectrogram_paths'].split(',')
                    all_segment_paths.extend(segments)
            
            if not all_segment_paths:
                print("No segments found for evaluation!")
                return
            
            # FILTER OUT ALREADY PROCESSED SEGMENTS
            filtered_segment_paths, skipped_count = filter_unprocessed_segments(experiment_df, all_segment_paths)
            
            if not filtered_segment_paths:
                print("All files have already been evaluated! No new segments to process.")
                print(f"Total segments: {len(all_segment_paths)}, Already processed: {skipped_count}")
                return
            
            print(f"Processing {len(filtered_segment_paths)} new segments (skipping {skipped_count} already processed)")
            
            # Create dataset with only unprocessed segments
            segments_df = pd.DataFrame({'file_path': filtered_segment_paths})
            evaluation_dataset = SegmentDataset(segments_df)
            
            # Step 8: Evaluate only the new segments
            print(f"Evaluating model on {len(evaluation_dataset)} NEW segments...")
            results = evaluate_ensemble_classification(
                model=ensemble_model,
                segment_dataset=evaluation_dataset,
                support_dataset=support_dataset,
                device=DEVICE,
                n_way=N_WAY,
                n_support=N_SUPPORT,
                batch_size=BATCH_SIZE
                
    )

            # Step 9: Update experiment DataFrame with segment class counts
            print("Updating experiment data...")
            if ENABLE_TIME_AGGREGATION:
                print("Using time-based aggregation...")
                experiment_df = update_segment_class_counts_with_time_aggregation(experiment_df, results)
                
                # Create time-aggregated summary
                summary_df = create_time_aggregated_summary(experiment_df)
                
                # Show time-aggregated statistics
                print(f"\nTime-aggregated statistics:")
                print(f"Average counts per time period:")
                for class_col in ['alarm_count_time_avg', 'non_alarm_count_time_avg', 'background_count_time_avg']:
                    if class_col in summary_df.columns:
                        avg_count = summary_df[class_col].mean()
                        print(f"  {class_col.replace('_count_time_avg', '')}: {avg_count:.1f}")
                
                # Show examples of time periods with multiple files
                if 'num_files' in summary_df.columns:
                    multiple_files = summary_df[summary_df['num_files'] > 1]
                    if len(multiple_files) > 0:
                        print(f"\nTime periods with multiple files: {len(multiple_files)}")
                        print("Examples:")
                        for _, row in multiple_files.head(3).iterrows():
                            print(f"  {row['time_key']}: {row['num_files']} files, "
                                  f"avg counts [{row.get('alarm_count_time_avg', 0)}, "
                                  f"{row.get('non_alarm_count_time_avg', 0)}, "
                                  f"{row.get('background_count_time_avg', 0)}]")
                
                # Save the time-aggregated results
                experiment_df.to_csv(EVALUATEDATAPATH, index=False)
                print(f"Time-aggregated results saved to {EVALUATEDATAPATH}")
            else:
                print("Using standard file-based counting (time aggregation disabled)...")
                experiment_df = update_segment_class_counts(experiment_df, results)
                
                # Save the updated results
                experiment_df.to_csv(EVALUATEDATAPATH, index=False)
                print(f"File-based results saved to {EVALUATEDATAPATH}")
            
            # Step 10: Update metadata with prediction results (if needed)
            # Only update if results contain files that are in the main metadata
            metadata_results = [r for r in results if not '_seg' in r.get('file_path', '')]
            if metadata_results:
                # This will handle appending new data instead of overwriting
                final_df = update_metadata_results(metadata_results, evaluation_dataset)
                print(f"Final metadata contains {len(final_df)} total entries")
            
            print("Evaluation complete!")
            
        else:
            print("Not enough data for few-shot learning.")
            print(requirements["suggestion"])
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()  # Print the full stack trace for debugging

if __name__ == "__main__":
    import torchaudio
    from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
    main()