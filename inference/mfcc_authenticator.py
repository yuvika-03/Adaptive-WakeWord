"""
MFCC-based Authenticator - More reliable speaker verification
"""

import numpy as np
import sounddevice as sd
import time
import sys
from pathlib import Path
import librosa
import soundfile as sf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.vad import VoiceActivityDetector
from preprocessing.denoise import NoiseSuppressor
from models.keyword_model import KeywordDetector

class MFCCAuthenticator:
    def __init__(self):
        self.config = Config
        self.feature_extractor = FeatureExtractor(Config)
        self.vad = VoiceActivityDetector(sample_rate=Config.SAMPLE_RATE)
        self.noise_suppressor = NoiseSuppressor(sample_rate=Config.SAMPLE_RATE)
        
        # Load keyword model only
        print("📂 Loading keyword model...")
        self.keyword_detector = KeywordDetector(str(Config.KEYWORD_MODEL_PATH), Config)
        print("✅ Keyword model loaded!")
        
        # Load MFCC enrollment
        self.load_enrollment()
        
        # Audio buffer
        self.audio_buffer = []
        self.buffer_size = int(Config.SAMPLE_RATE * 2.0)
        self.processing = False
        
        # Find working device
        self.audio_device = self._find_working_device()
        
        # Thresholds
        self.keyword_threshold = 0.85  # From calibration
        self.speaker_threshold = 0.6   # MFCC similarity threshold
        
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
                        print(f"✅ Using device {i}: {device['name']}")
                        return i
                    except:
                        continue
            return None
        except:
            return None
    
    def load_enrollment(self):
        """Load enrolled user's MFCC features"""
        enroll_dir = self.config.ENROLLMENTS_DIR
        
        if not enroll_dir.exists():
            print("❌ No enrollment found!")
            return False
        
        self.enrolled_features = None
        self.enrolled_user = None
        
        for user_dir in enroll_dir.iterdir():
            if user_dir.is_dir():
                # First try to load MFCC features
                mfcc_file = user_dir / "mfcc_features.npy"
                if mfcc_file.exists():
                    self.enrolled_features = np.load(mfcc_file)
                    self.enrolled_user = user_dir.name
                    print(f"✅ Loaded MFCC enrollment for: {self.enrolled_user}")
                    print(f"   Features shape: {self.enrolled_features.shape}")
                    return True
                
                # If no MFCC, try to create from embedding
                embedding_file = user_dir / "embedding.npy"
                if embedding_file.exists():
                    embedding = np.load(embedding_file)
                    # Use embedding as features (will work with MFCC comparison)
                    self.enrolled_features = embedding
                    self.enrolled_user = user_dir.name
                    print(f"✅ Loaded embedding enrollment for: {self.enrolled_user}")
                    print(f"   Features shape: {self.enrolled_features.shape}")
                    return True
        
        print("❌ No enrollment found! Please run:")
        print("  python enroll_mfcc.py")
        return False
    
    def extract_mfcc_features(self, audio):
        """Extract MFCC features from audio"""
        try:
            # Use librosa to extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.config.SAMPLE_RATE,
                n_mfcc=13,
                n_fft=Config.N_FFT,
                hop_length=Config.HOP_LENGTH
            )
            
            # Take mean across time to get a fixed-size feature vector
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Also add standard deviation for more information
            mfcc_std = np.std(mfcc, axis=1)
            
            # Concatenate mean and std
            features = np.concatenate([mfcc_mean, mfcc_std])
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return None
    
    def compute_similarity(self, features1, features2):
        """Compute cosine similarity between two feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        
        # Ensure same length
        min_len = min(len(features1), len(features2))
        f1 = features1[:min_len]
        f2 = features2[:min_len]
        
        similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
        return float(similarity)
    
    def process_audio(self, audio_chunk):
        """Process audio chunk for authentication"""
        if self.processing or self.enrolled_features is None:
            return
        
        self.processing = True
        
        try:
            # Add to buffer
            self.audio_buffer.extend(audio_chunk)
            if len(self.audio_buffer) > self.buffer_size:
                self.audio_buffer = self.audio_buffer[-self.buffer_size:]
            
            if len(self.audio_buffer) < self.buffer_size:
                self.processing = False
                return
            
            audio = np.array(self.audio_buffer)
            
            # Voice activity detection
            voice_segments = self.vad.get_voice_segments(audio)
            if not voice_segments:
                self.processing = False
                return
            
            # Get largest voice segment
            largest_segment = max(voice_segments, key=lambda x: x[1] - x[0])
            start_sample = int(largest_segment[0] * self.config.SAMPLE_RATE)
            end_sample = int(largest_segment[1] * self.config.SAMPLE_RATE)
            voice_audio = audio[start_sample:end_sample]
            
            if len(voice_audio) < self.config.SAMPLE_RATE * 0.3:
                self.processing = False
                return
            
            # Extract features for keyword detection
            features = self.feature_extractor.extract_mel_spectrogram(voice_audio, duration=2.0)
            
            # Keyword detection
            keyword_class, keyword_conf = self.keyword_detector.detect(features)
            
            # Extract MFCC features for speaker verification
            mfcc_features = self.extract_mfcc_features(voice_audio)
            
            if mfcc_features is not None:
                # Compute similarity with enrollment
                speaker_sim = self.compute_similarity(mfcc_features, self.enrolled_features)
            else:
                speaker_sim = 0.0
            
            # Display results clearly
            print("\n" + "=" * 60)
            print("🔍 AUTHENTICATION RESULTS")
            print("=" * 60)
            print(f"📢 Keyword Confidence: {keyword_conf:.3f}")
            print(f"👤 Speaker Similarity: {speaker_sim:.3f}")
            
            # Decision
            keyword_detected = keyword_conf > self.keyword_threshold
            speaker_verified = speaker_sim > self.speaker_threshold
            
            if keyword_detected and speaker_verified:
                print("\n🔓 ✅ ACCESS GRANTED!")
                print(f"   Welcome {self.enrolled_user}!")
                print(f"   Keyword: {keyword_conf:.3f} (threshold: {self.keyword_threshold:.2f})")
                print(f"   Speaker: {speaker_sim:.3f} (threshold: {self.speaker_threshold:.2f})")
            elif keyword_detected:
                print("\n🔑 ⚠️ Keyword detected but speaker not verified")
                print(f"   Keyword: {keyword_conf:.3f} (threshold: {self.keyword_threshold:.2f})")
                print(f"   Speaker: {speaker_sim:.3f} (threshold: {self.speaker_threshold:.2f})")
                print("   Please speak more clearly or re-enroll")
            elif speaker_verified:
                print("\n👤 ⚠️ Speaker verified but keyword not detected")
                print(f"   Keyword: {keyword_conf:.3f} (threshold: {self.keyword_threshold:.2f})")
                print(f"   Speaker: {speaker_sim:.3f} (threshold: {self.speaker_threshold:.2f})")
                print("   Please say the wake word clearly")
            else:
                print("\n❌ No match found")
                print(f"   Keyword: {keyword_conf:.3f} (threshold: {self.keyword_threshold:.2f})")
                print(f"   Speaker: {speaker_sim:.3f} (threshold: {self.speaker_threshold:.2f})")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"Error: {e}")
        
        self.processing = False
    
    def callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        self.process_audio(indata[:, 0])
    
    def start(self):
        """Start authentication"""
        if self.enrolled_features is None:
            print("\n❌ No enrollment found! Please run:")
            print("  python enroll_mfcc.py")
            return
        
        print("\n" + "=" * 60)
        print("🎤 ADAPTIVE WAKEWORD - MFCC AUTHENTICATION")
        print("=" * 60)
        print(f"👤 Enrolled user: {self.enrolled_user}")
        print(f"🎯 Keyword threshold: {self.keyword_threshold:.2f}")
        print(f"🎯 Speaker threshold: {self.speaker_threshold:.2f}")
        print("\nSpeak your wake word...")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        print()
        
        stream = sd.InputStream(
            samplerate=self.config.SAMPLE_RATE,
            channels=1,
            callback=self.callback,
            blocksize=int(self.config.SAMPLE_RATE * 0.5),
            device=self.audio_device
        )
        
        with stream:
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\n👋 Stopping authentication...")

if __name__ == "__main__":
    authenticator = MFCCAuthenticator()
    authenticator.start()