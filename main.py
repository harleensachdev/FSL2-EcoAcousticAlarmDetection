import os
import torch
import sys
import traceback
from torch.utils.data import DataLoader

# Add the directory containing the preprocessing script to the Python path
preprocessing_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(preprocessing_dir)

from config import (
    AUDIO_DIR,
    SPECTROGRAM_DIR,
    BATCH_SIZE,
    DEVICE, 
    N_SUPPORT, 
    N_QUERY,
    TEST_SIZE,
    REQUIRED_CLASSES,
    N_WAY,
    EPISODES,
    PROTO_WEIGHT,
    RELATION_WEIGHT
)

# Import preprocessing and training functions
from src.preprocess import (
    getmetadata,
    create_all_spectrograms,
    check_class_distribution,
    verify_few_shot_requirements,
)

from src.dataset import BirdSoundDataset
from src.models import CNNEncoder, RelationNetwork, EnsembleModel
from src.training import train_few_shot
from src.evaluation import evaluate_episodic, update_metadata_results

def preprocess_data():
    """
    Run preprocessing steps to prepare the dataset.
    """
    print("Starting preprocessing...")
    
    # Scan for new audio files and update metadata
    metadata_df = getmetadata()
    
    # Create spectrograms for all files
    create_all_spectrograms()
    
    # Check class distribution
    dist = check_class_distribution(metadata_df)
    print("Class distribution:")
    for cls, count in dist["class_counts"].items():
        print(f"  {cls}: {count} samples ({dist['class_percentages'][cls]:.2f}%)")
    
    return metadata_df

def main():
    # Step 1: Create directories if they don't exist
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
    
    # Step 2: Run preprocessing
    metadata_df = preprocess_data()
    
    requirements = verify_few_shot_requirements(
        metadata_df, 
        n_way=N_WAY, 
        k_shot=N_SUPPORT,  
        query_size=N_QUERY, 
        test_size=TEST_SIZE
    )

    # Step 4: Prepare few-shot experiment
    if requirements["meets_requirements"]:
        all_metadata = metadata_df[metadata_df['label'].isin(REQUIRED_CLASSES)]
        all_dataset = BirdSoundDataset(all_metadata)

        test_metadata = all_metadata[all_metadata['file_path'].str.contains('test/')]
        test_dataset = BirdSoundDataset(test_metadata)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        try:
            # Step 5: Initialize models
            encoder = CNNEncoder().to(DEVICE)
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
            
            # Step 7: Evaluate the model
            print("Evaluating model...")
            accuracy, results = evaluate_episodic(
                model=ensemble_model,
                test_dataset=test_dataset,
                device=DEVICE,
                n_way=N_WAY,
                n_support=N_SUPPORT,
                n_query=N_QUERY,
                n_episodes=EPISODES
            )
            
            print(f"Test Accuracy: {accuracy:.2f}%")
            
            # Update metadata results
            update_metadata_results(results, test_dataset)
        
        except Exception as e:
            print(f"Detailed Error in few-shot setup: {e}")
            traceback.print_exc()  # This prints the full stack trace for debugging
    
    else:
        print("Not enough data for few-shot learning.")
        print(requirements["suggestion"])

if __name__ == "__main__":
    main()