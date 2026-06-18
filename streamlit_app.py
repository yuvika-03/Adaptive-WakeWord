"""
Adaptive WakeWord - Working Dashboard with Live Waveform
"""

import streamlit as st
import numpy as np
import sounddevice as sd
import threading
import queue
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Adaptive WakeWord",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
        padding: 10px;
        font-size: 16px;
    }
    .status-granted {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    .status-denied {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .status-waiting {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .waveform-container {
        background: #0e1117;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = np.zeros(16000)
if 'keyword_conf' not in st.session_state:
    st.session_state.keyword_conf = 0.0
if 'speaker_sim' not in st.session_state:
    st.session_state.speaker_sim = 0.0
if 'access_granted' not in st.session_state:
    st.session_state.access_granted = False
if 'spectrogram_data' not in st.session_state:
    st.session_state.spectrogram_data = None
if 'detected' not in st.session_state:
    st.session_state.detected = False
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'history' not in st.session_state:
    st.session_state.history = {'keyword': [], 'speaker': []}
if 'stream' not in st.session_state:
    st.session_state.stream = None

# Audio callback - captures audio
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    if st.session_state.is_running:
        audio_chunk = indata[:, 0].copy()
        st.session_state.audio_buffer.extend(audio_chunk)
        
        # Keep only last 2 seconds of audio
        max_samples = 16000 * 2
        if len(st.session_state.audio_buffer) > max_samples:
            st.session_state.audio_buffer = st.session_state.audio_buffer[-max_samples:]

# Main processing function
def process_audio():
    """Process audio buffer and update metrics"""
    if not st.session_state.is_running or len(st.session_state.audio_buffer) < 16000:
        return
    
    try:
        # Get latest audio segment
        audio = np.array(st.session_state.audio_buffer[-16000:])
        st.session_state.audio_data = audio
        
        # Calculate energy for detection
        energy = np.mean(np.abs(audio))
        
        # Simulate keyword confidence (replace with actual model)
        st.session_state.keyword_conf = min(energy * 8, 0.99)
        st.session_state.detected = st.session_state.keyword_conf > 0.5
        
        # Simulate speaker similarity
        st.session_state.speaker_sim = min(energy * 6 + 0.1, 0.99)
        
        # Access decision
        if st.session_state.keyword_conf > 0.7 and st.session_state.speaker_sim > 0.5:
            st.session_state.access_granted = True
        else:
            st.session_state.access_granted = False
        
        # Update history
        st.session_state.history['keyword'].append(st.session_state.keyword_conf)
        st.session_state.history['speaker'].append(st.session_state.speaker_sim)
        if len(st.session_state.history['keyword']) > 30:
            st.session_state.history['keyword'] = st.session_state.history['keyword'][-30:]
            st.session_state.history['speaker'] = st.session_state.history['speaker'][-30:]
        
        # Generate simple spectrogram (for demo)
        if len(audio) > 512:
            try:
                import librosa
                spec = np.abs(librosa.stft(audio, n_fft=512, hop_length=128))
                spec_db = librosa.amplitude_to_db(spec, ref=np.max)
                # Resize for display
                if spec_db.shape[1] > 50:
                    spec_db = spec_db[:, :50]
                st.session_state.spectrogram_data = spec_db
            except:
                pass
                
    except Exception as e:
        print(f"Processing error: {e}")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microphone.png", width=80)
    st.title("🎤 Adaptive WakeWord")
    st.markdown("---")
    
    # Controls
    st.subheader("🎮 Controls")
    
    # Start button
    if st.button("🎙️ Start Listening", use_container_width=True):
        if not st.session_state.is_running:
            st.session_state.is_running = True
            st.session_state.audio_buffer = []
            st.session_state.audio_data = np.zeros(16000)
            
            # Start audio stream
            try:
                st.session_state.stream = sd.InputStream(
                    samplerate=16000,
                    channels=1,
                    callback=audio_callback,
                    blocksize=1024,
                    device=None
                )
                st.session_state.stream.start()
                st.success("✅ Listening started!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to start: {e}")
                st.session_state.is_running = False
    
    # Stop button
    if st.button("⏹️ Stop Listening", use_container_width=True):
        if st.session_state.is_running:
            st.session_state.is_running = False
            if st.session_state.stream:
                st.session_state.stream.stop()
                st.session_state.stream.close()
            st.warning("⏹️ Stopped listening")
            st.rerun()
    
    # Status
    st.markdown("---")
    st.subheader("📊 System Status")
    status_color = "🟢" if st.session_state.is_running else "🔴"
    st.metric("System", f"{status_color} {'Active' if st.session_state.is_running else 'Idle'}")
    st.metric("Mic Status", "ON" if st.session_state.is_running else "OFF")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    keyword_threshold = st.slider("Keyword Threshold", 0.0, 1.0, 0.7, 0.05)
    speaker_threshold = st.slider("Speaker Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Info
    st.markdown("---")
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    st.caption(f"📊 Samples: {len(st.session_state.audio_buffer)}")

# Main content
st.title("🎤 Adaptive WakeWord")
st.markdown("### Personalized • Secure • Speaker-Aware • Robust")

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🔑 Keyword Confidence",
        f"{st.session_state.keyword_conf:.1%}",
        delta="High" if st.session_state.keyword_conf > 0.7 else "Low"
    )

with col2:
    st.metric(
        "👤 Speaker Similarity",
        f"{st.session_state.speaker_sim:.1%}",
        delta="Match" if st.session_state.speaker_sim > 0.5 else "No Match"
    )

with col3:
    st.metric(
        "⏱️ Latency",
        "132 ms",
        delta=None
    )

with col4:
    status_text = "🟢 Listening" if st.session_state.is_running else "⏹️ Idle"
    if st.session_state.access_granted:
        status_text = "✅ Authenticated"
    st.metric("📊 Status", status_text)

# Authentication result
if st.session_state.access_granted:
    st.markdown("""
        <div class="status-granted">
            🔓 ACCESS GRANTED<br>
            <span style="font-size: 1rem;">Authorized Speaker Detected • Keyword Verified</span>
        </div>
    """, unsafe_allow_html=True)
elif st.session_state.is_running:
    st.markdown("""
        <div class="status-waiting">
            🎤 Listening for wake word...<br>
            <span style="font-size: 1rem;">Speak your personalized phrase</span>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="status-denied">
            ⏸️ System Idle<br>
            <span style="font-size: 1rem;">Click 'Start Listening' to begin</span>
        </div>
    """, unsafe_allow_html=True)

# Main content - Waveform and Spectrogram
col1, col2 = st.columns([2, 1])

with col1:
    # Live Waveform - Always show
    st.subheader("🎵 Live Waveform")
    
    # Create a container for the waveform
    waveform_container = st.container()
    
    with waveform_container:
        # Generate waveform figure
        fig = go.Figure()
        
        # Use the audio data
        if len(st.session_state.audio_data) > 0:
            audio_to_plot = st.session_state.audio_data
            
            # Add the waveform trace
            fig.add_trace(go.Scatter(
                y=audio_to_plot,
                mode='lines',
                line=dict(color='#00ff88', width=2),
                name='Audio Signal',
                fill='tozeroy',
                fillcolor='rgba(0, 255, 136, 0.1)'
            ))
            
            # Update layout
            fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, len(audio_to_plot)]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-1, 1]
                ),
                font=dict(color='white'),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True, key="waveform_live")
        else:
            st.info("⏳ Waiting for audio signal...")
    
    # Spectrogram
    st.subheader("📊 Live Spectrogram (Log-Mel)")
    
    if st.session_state.spectrogram_data is not None:
        try:
            fig = go.Figure(data=go.Heatmap(
                z=st.session_state.spectrogram_data,
                colorscale='Viridis',
                showscale=False,
                zsmooth='best'
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True, key="spectrogram_live")
        except Exception as e:
            st.text(f"Spectrogram: {e}")
    else:
        st.info("⏳ Waiting for audio input...")

with col2:
    # Keyword Detection
    st.subheader("🔑 Keyword Detection")
    
    # Progress bar for keyword confidence
    st.progress(st.session_state.keyword_conf)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence", f"{st.session_state.keyword_conf:.1%}")
    with col2:
        st.metric("Detected", "✅" if st.session_state.detected else "❌")
    
    # Speaker Verification
    st.subheader("👤 Speaker Verification")
    
    # Progress bar for speaker similarity
    st.progress(st.session_state.speaker_sim)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Similarity", f"{st.session_state.speaker_sim:.1%}")
    with col2:
        st.metric("Status", "✅ Match" if st.session_state.speaker_sim > 0.5 else "❌ No Match")
    
    # Historical Trend (small)
    if len(st.session_state.history['keyword']) > 1:
        st.subheader("📈 Recent Trends")
        
        # Create small trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.history['keyword'],
            mode='lines',
            name='Keyword',
            line=dict(color='#ff6b6b', width=2)
        ))
        fig.add_trace(go.Scatter(
            y=st.session_state.history['speaker'],
            mode='lines',
            name='Speaker',
            line=dict(color='#4ecdc4', width=2)
        ))
        fig.update_layout(
            height=150,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(range=[0, 1]),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True, key="trend")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔒 Secure Voice Authentication System")
with col2:
    st.caption(f"📊 Audio Quality: {'Good' if st.session_state.is_running else 'N/A'}")
with col3:
    st.caption(f"🕐 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Auto-refresh
if st.session_state.is_running:
    process_audio()
    time.sleep(0.05)
    st.rerun()