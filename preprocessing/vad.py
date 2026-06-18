import numpy as np
import librosa
from scipy import signal

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, frame_length=0.025, frame_stride=0.010):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_stride = frame_stride
        self.energy_threshold = 0.1
        
    def detect_voice(self, audio):
        """Detect voice activity in audio"""
        # Calculate energy
        frame_length_samples = int(self.frame_length * self.sample_rate)
        frame_stride_samples = int(self.frame_stride * self.sample_rate)
        
        # Create frames
        frames = []
        for i in range(0, len(audio) - frame_length_samples, frame_stride_samples):
            frame = audio[i:i + frame_length_samples]
            frames.append(frame)
        
        # Calculate energy for each frame
        energies = [np.sum(frame**2) for frame in frames]
        
        # Normalize energies
        energies = np.array(energies)
        energies = energies / (np.max(energies) + 1e-8)
        
        # Apply threshold
        voice_frames = energies > self.energy_threshold
        
        return voice_frames, energies
    
    def get_voice_segments(self, audio):
        """Get segments where voice is detected"""
        voice_frames, _ = self.detect_voice(audio)
        
        # Find continuous voice segments
        segments = []
        start = None
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and start is None:
                start = i
            elif not is_voice and start is not None:
                end = i
                segments.append((start, end))
                start = None
        
        if start is not None:
            segments.append((start, len(voice_frames)))
        
        # Convert frame indices to time
        frame_step = int(self.frame_stride * self.sample_rate)
        time_segments = [
            (start * frame_step / self.sample_rate, 
             end * frame_step / self.sample_rate)
            for start, end in segments
        ]
        
        return time_segments