import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import soundfile as sf
from pathlib import Path
import random

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.SAMPLE_RATE
        self.n_mels = config.N_MELS
        self.n_fft = config.N_FFT
        self.hop_length = config.HOP_LENGTH
        self.win_length = config.WIN_LENGTH
        
    def extract_mel_spectrogram(self, audio, duration=None):
        """Extract mel spectrogram from audio"""
        if duration is not None:
            target_length = int(duration * self.sample_rate)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        return log_mel
    
    def extract_mfcc(self, audio, n_mfcc=13):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
    
    def extract_voice_embedding(self, audio):
        """Extract voice embedding features"""
        # This is a placeholder for actual speaker embedding extraction
        mel_spec = self.extract_mel_spectrogram(audio)
        # Reshape for neural network input
        mel_spec = np.expand_dims(mel_spec, axis=0)
        return mel_spec

class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, config, augment=False):
        self.audio_paths = audio_paths
        self.labels = labels
        self.config = config
        self.augment = augment
        self.feature_extractor = FeatureExtractor(config)
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
        
        # Extract features
        features = self.feature_extractor.extract_mel_spectrogram(audio)
        features = torch.FloatTensor(features)
        
        # Apply augmentation if enabled
        if self.augment:
            features = self._augment(features)
        
        return features, label
    
    def _augment(self, features):
        """Apply data augmentation"""
        # Time masking
        if random.random() > 0.5:
            time_mask_length = random.randint(1, 20)
            time_start = random.randint(0, features.shape[1] - time_mask_length)
            features[:, time_start:time_start + time_mask_length] = 0
        
        # Frequency masking
        if random.random() > 0.5:
            freq_mask_length = random.randint(1, 5)
            freq_start = random.randint(0, features.shape[0] - freq_mask_length)
            features[freq_start:freq_start + freq_mask_length, :] = 0
        
        return features