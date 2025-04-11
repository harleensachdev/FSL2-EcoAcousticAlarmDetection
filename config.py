import torch
import os
from datetime import datetime

DATA_DIR = os.path.join(os.getcwd(), "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio_files")
SPECTROGRAM_DIR = os.path.join(DATA_DIR, "spectrograms")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

# Audio processing
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050  # 1 second of audio
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# Training parameters
TEST_SIZE = 30
BATCH_SIZE = 15
EPISODES = 100 
LEARNING_RATE = 0.001
N_WAY = 3  # Number of classes per episode
N_SUPPORT = 5 # Number of support samples per class
N_QUERY = 6 # Number of query samples per class

# Label mapping
LABEL_MAP = {
    "alarm": 0,
    "non_alarm": 1,
    "background":2
}
REQUIRED_CLASSES = ["alarm", "non_alarm", "background"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Model parameters
EMBEDDING_DIM = 128

# Ensemble training weights
PROTO_WEIGHT = 0.6
RELATION_WEIGHT = 0.4

# Run ID for logging
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
