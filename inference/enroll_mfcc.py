"""
MFCC-based enrollment - More reliable speaker enrollment
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from pathlib import Path
import sys
import time
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from preprocessing.vad import VoiceActivityDetector
from preprocessing.feature_extractor import FeatureExtractor

class MFCCEnroller:
    def __init__(self):
        self.config = Config
        self.vad = VoiceActivityDetector(sample_rate=Config.SAMPLE_RATE)
        self.feature_extractor = FeatureExtractor(Config)
        
        # Find working device
        self.audio_device = self._find_working_device()
        
    def _find_working_device(self):
        """Find a working audio input device"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    try:
                        sd.check_input_settings(
                            device=i,
                            samplerate=self.config.SAMPLE_RATE,
                            channels=1
                        )
                        return i
                    except:
                        continue
            return None
        except:
            return None
    
    def extract_mfcc_features(self, audio):
        """Extract MFCC features from audio"""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.config.SAMPLE_RATE,
                n_mfcc=13,
                n_fft=Config.N_FFT,
                hop_length=Config.HOP_LENGTH
            )
            
            # Mean and std for robust features
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features = np.concatenate([mfcc_mean, mfcc_std])
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return None
    
    def enroll_speaker(self, username, num_samples=5):
        """Enroll a speaker using MFCC features"""
        print("\n" + "=" * 60)
        print("👤 MFCC-BASED SPEAKER ENROLLMENT")
        print("=" * 60)
        print(f"User: {username}")
        print(f"Number of samples: {num_samples}")
        print("\n🎤 Speak your wake word clearly for each recording")
        print("📝 Try to speak with the same tone and volume")
        print("=" * 60)
        
        enroll_dir = self.config.ENROLLMENTS_DIR / username
        enroll_dir.mkdir(exist_ok=True)
        
        all_features = []
        
        for i in range(num_samples):
            print(f"\n📢 Recording {i+1}/{num_samples}...")
            input("Press Enter to start recording...")
            
            duration = 3.0
            print("🎤 Recording... Speak now!")
            
            try:
                audio = sd.rec(
                    int(duration * self.config.SAMPLE_RATE),
                    samplerate=self.config.SAMPLE_RATE,
                    channels=1,
                    dtype='float32',
                    device=self.audio_device
                )
                sd.wait()
                audio = audio.flatten()
                print("✅ Recording complete!")
            except Exception as e:
                print(f"❌ Recording failed: {e}")
                continue
            
            # Save audio
            audio_file = enroll_dir / f"sample_{i+1}.wav"
            sf.write(audio_file, audio, self.config.SAMPLE_RATE)
            print(f"✅ Saved: {audio_file}")
            
            # Check for voice
            voice_segments = self.vad.get_voice_segments(audio)
            if not voice_segments:
                print("⚠️  No voice detected. Please speak louder.")
                continue
            
            # Get voice segment
            largest_segment = max(voice_segments, key=lambda x: x[1] - x[0])
            start = int(largest_segment[0] * self.config.SAMPLE_RATE)
            end = int(largest_segment[1] * self.config.SAMPLE_RATE)
            voice_audio = audio[start:end]
            
            # Extract MFCC features
            features = self.extract_mfcc_features(voice_audio)
            
            if features is not None:
                all_features.append(features)
                print(f"✅ Extracted MFCC features (shape: {features.shape})")
            else:
                print("❌ Failed to extract features")
        
        if all_features:
            # Average all features
            avg_features = np.mean(all_features, axis=0)
            
            # Normalize
            avg_features = avg_features / (np.linalg.norm(avg_features) + 1e-8)
            
            # Save MFCC features
            mfcc_file = enroll_dir / "mfcc_features.npy"
            np.save(mfcc_file, avg_features)
            
            # Also save average embedding for compatibility
            embedding_file = enroll_dir / "embedding.npy"
            np.save(embedding_file, avg_features)
            
            # Save info
            info = {
                'username': username,
                'num_samples': len(all_features),
                'feature_type': 'mfcc_mean_std',
                'feature_shape': avg_features.shape,
                'timestamp': time.time()
            }
            with open(enroll_dir / "info.json", 'w') as f:
                json.dump(info, f, indent=2)
            
            print("\n" + "=" * 60)
            print("✅ ENROLLMENT COMPLETE!")
            print("=" * 60)
            print(f"User: {username}")
            print(f"Samples: {len(all_features)}")
            print(f"Features saved to: {mfcc_file}")
            print(f"Feature shape: {avg_features.shape}")
            
            return avg_features
        else:
            print("\n❌ Enrollment failed! No valid samples.")
            return None

if __name__ == "__main__":
    import sys
    username = sys.argv[1] if len(sys.argv) > 1 else "test_user"
    
    enroller = MFCCEnroller()
    enroller.enroll_speaker(username)