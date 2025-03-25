import torch
import os
from datetime import datetime

DATA_DIR = os.path.join(os.getcwd(), "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio_files")
SPECTROGRAM_DIR = os.path.join(DATA_DIR, "spectrograms")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
EXPERIMENT_PATH = os.path.join(DATA_DIR, "experiments.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Audio processing
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050  # 1 second of audio
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# Training parameters
BATCH_SIZE = 16
EPOCHS = 17
EPISODES = 100  # Number of episodes for meta-learning
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Meta-learning parameters
N_WAY = 3  # Number of classes per episode
N_SUPPORT = 3  # Number of support samples per class
N_QUERY = 2  # Number of query samples per class

# Dataset splits
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Label mapping
LABEL_MAP = {
    "alarm": 0,
    "non_alarm": 1,
    "background":2
}

# Model parameters
EMBEDDING_DIM = 128

# Ensemble training weights
PROTO_WEIGHT = 0.6
RELATION_WEIGHT = 0.4

# Run ID for logging
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# Logging
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)