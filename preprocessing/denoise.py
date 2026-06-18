import numpy as np
import librosa
from scipy import signal
import noisereduce as nr

class NoiseSuppressor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def suppress_noise(self, audio, noise_sample=None):
        """Apply noise suppression to audio"""
        if noise_sample is not None:
            # Use provided noise sample for reduction
            reduced_audio = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                y_noise=noise_sample,
                prop_decrease=0.8
            )
        else:
            # Use spectral gating
            reduced_audio = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                prop_decrease=0.8
            )
        
        return reduced_audio
    
    def spectral_subtraction(self, audio, noise_floor=0.01):
        """Apply spectral subtraction for noise reduction"""
        # Compute STFT
        n_fft = 2048
        hop_length = 512
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise magnitude
        noise_magnitude = np.mean(magnitude, axis=1, keepdims=True)
        noise_magnitude = np.minimum(noise_magnitude, noise_floor)
        
        # Subtract noise
        clean_magnitude = magnitude - noise_magnitude
        clean_magnitude = np.maximum(clean_magnitude, 0)
        
        # Reconstruct audio
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft, hop_length=hop_length)
        
        return clean_audio