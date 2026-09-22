import streamlit as st
import numpy as np
import joblib
import tempfile
import os
import random
from feature_extractor import extract_features

st.set_page_config(page_title="Voice Clone Detector", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_artifacts():
    artifact = joblib.load("voice_clone_model.pkl")
    return artifact["model"], joblib.load("scaler.pkl"), artifact["n_features"]

model, scaler, N_FEATURES = load_artifacts()

st.title("🛡️ AI Voice Cloning Detector")
st.markdown("Detects **real human voice** vs **synthetic / cloned voice**.")

CHALLENGES = [
    "The quick brown fox jumps over the lazy dog.",
    "Verify transfer code seven three nine.",
    "My voice is my password, confirm now.",
    "Purple elephants dance at midnight.",
]
if "challenge" not in st.session_state:
    st.session_state.challenge = random.choice(CHALLENGES)

st.subheader("Step 1 — Liveness Challenge")
st.info(f"🗣️ Read this aloud when recording:\n\n> {st.session_state.challenge}")
if st.button("🔄 New challenge"):
    st.session_state.challenge = random.choice(CHALLENGES)
    st.rerun()

st.subheader("Step 2 — Submit Audio")
tab1, tab2 = st.tabs(["📁 Upload File", "🎙️ Record from Mic"])
audio_bytes = None

with tab1:
    up = st.file_uploader("Upload audio", type=["wav","mp3","flac","ogg","m4a"])
    if up:
        audio_bytes = up.read()
        st.audio(audio_bytes)

with tab2:
    rec = st.audio_input("Record your response")
    if rec:
        audio_bytes = rec.read()
        st.audio(audio_bytes)

threshold = st.slider("Detection threshold (fake prob)", 0.0, 1.0, 0.5, 0.05)

st.subheader("Step 3 — Verdict")
if audio_bytes and st.button("🔍 Analyze", type="primary"):
    with st.spinner("Analyzing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            feats = extract_features(tmp_path).reshape(1, -1)
            if feats.shape[1] != N_FEATURES:
                st.error(f"Feature mismatch: got {feats.shape[1]}, expected {N_FEATURES}")
                st.stop()
            fs = scaler.transform(feats)
            p = model.predict_proba(fs)[0]
            real_conf, fake_conf = p[0]*100, p[1]*100
            if p[1] >= threshold:
                st.error(f"🚨 FAKE / CLONED VOICE — {fake_conf:.2f}% confidence")
            else:
                st.success(f"✅ REAL VOICE — {real_conf:.2f}% confidence")
            c1, c2 = st.columns(2)
            c1.metric("Real", f"{real_conf:.2f}%")
            c2.metric("Fake", f"{fake_conf:.2f}%")
            st.progress(float(p[1]))
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

st.markdown("---")
st.caption("Hackathon prototype · XGBoost + librosa MFCC · 44-dim input")
