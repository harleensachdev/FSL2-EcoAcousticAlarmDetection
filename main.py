# main.py
import os
import torch
import sys
import traceback


# Add the directory containing the preprocessing script to the Python path
preprocessing_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(preprocessing_dir)

from config import (
    AUDIO_DIR,
    SPECTROGRAM_DIR,
    BATCH_SIZE,
    DEVICE, 
    N_SUPPORT, 
    N_QUERY
)

# Import preprocessing and training functions
from src.preprocess import (
    scan_and_update_metadata,
    create_all_spectrograms,
    check_class_distribution,
    verify_few_shot_requirements,
    prepare_few_shot_experiment
)

from src.dataset import (
    create_few_shot_datasets,
    create_few_shot_dataloaders
)

from src.models import CNNEncoder, RelationNetwork, PrototypicalNet, EnsembleModel
from src.training import train_few_shot
from src.evaluation import evaluate_model, update_metadata_results

def preprocess_data():
    """
    Run preprocessing steps to prepare the dataset.
    """
    print("Starting preprocessing...")
    
    # Scan for new audio files and update metadata
    metadata_df = scan_and_update_metadata()
    
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
        n_way=3, 
        k_shot=N_SUPPORT,  
        query_size=N_QUERY, 
        test_size=3,
        minimum_samples_per_class=5
    )
    
    # Step 4: Prepare few-shot experiment
    if requirements["meets_requirements"] or len(requirements["eligible_classes"]) > 0:
        experiment = prepare_few_shot_experiment(
            n_way=3,  # Match with the requirement
            k_shot=N_SUPPORT, 
            query_size=N_QUERY, 
            test_size=3
        )
        
        # Create few-shot datasets
        try:
            support_dataset, query_dataset, test_dataset, selected_classes, label_map = create_few_shot_datasets(
                metadata_df, 
                n_way=3,  # Match with the requirement
                k_shot=N_SUPPORT,
                query_size=N_QUERY,
                test_size=3
            )
            # Create dataloaders
            support_loader, query_loader, test_loader = create_few_shot_dataloaders(
                support_dataset, 
                query_dataset, 
                test_dataset, 
                batch_size=BATCH_SIZE
            )
            
            # Step 5: Initialize models
            encoder = CNNEncoder().to(DEVICE)
            relation_net = RelationNetwork().to(DEVICE)
            proto_net = PrototypicalNet(encoder).to(DEVICE)
            ensemble_model = EnsembleModel(proto_net, relation_net).to(DEVICE)
            
            # Step 6: Train the model
            print("Starting training...")
            train_losses = train_few_shot(
                ensemble_model, 
                support_loader, 
                query_loader, 
                epochs=10,
                n_way=N_SUPPORT,
                n_support=N_SUPPORT,
                n_query=N_QUERY
            )
            
            # Step 7: Evaluate the model
            print("Evaluating the model...")
            accuracy, results = evaluate_model(
                encoder, 
                relation_net, 
                test_loader, 
                device=DEVICE
            )
            
            # Print results
            print(f"Training Losses: {train_losses}")
            print(f"Test Accuracy: {accuracy:.2f}%")

            update_metadata_results(results, test_dataset)
            
        except Exception as e:
            print(f"Detailed Error in few-shot setup: {e}")
            traceback.print_exc()  # This will print the full stack trace
    else:
        print("Not enough data for few-shot learning.")
        print(requirements["suggestion"])

if __name__ == "__main__":
    main()