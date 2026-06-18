import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    
    # Data paths
    GOOGLE_SPEECH_COMMANDS = DATA_DIR / "google_speech_commands"
    LIBRIPHRASE = DATA_DIR / "libriphrase_subset"
    VOXCELEB = DATA_DIR / "voxceleb"
    MUSAN = DATA_DIR / "musan"
    ENROLLMENTS_DIR = DATA_DIR / "enrollments"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # Model paths
    KEYWORD_MODEL_PATH = MODELS_DIR / "keyword_model.pt"
    SPEAKER_MODEL_PATH = MODELS_DIR / "speaker_model.pt"
    
    # Audio parameters
    SAMPLE_RATE = 16000
    N_MELS = 40
    N_FFT = 512
    HOP_LENGTH = 160
    WIN_LENGTH = 400
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Keyword detection
    KEYWORD_THRESHOLD = 0.85
    KEYWORD_WINDOW_SIZE = 2.0  # seconds
    
    # Speaker verification
    SPEAKER_THRESHOLD = 0.5
    EMBEDDING_SIZE = 256
    TRIPLET_MARGIN = 0.2
    
    # Authentication
    AUTH_THRESHOLD = 0.7
    AUTH_KEYWORD_WEIGHT = 0.5
    AUTH_SPEAKER_WEIGHT = 0.5
    
    # config.py - Add this line
    NUM_KEYWORD_CLASSES = 10  # Match the number of classes you trained with

    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.ENROLLMENTS_DIR,
            cls.PROCESSED_DIR,
            cls.MODELS_DIR,
            cls.DATA_DIR / "custom_keyword",
            cls.DATA_DIR / "custom_negative",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Initialize directories
Config.create_directories()