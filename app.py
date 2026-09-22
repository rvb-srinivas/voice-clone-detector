import streamlit as st
import numpy as np
import joblib
import tempfile
import os
import random
import io
from datetime import datetime
from feature_extractor import extract_features
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Page config + custom CSS
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Voice Cloning Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .tagline { color: #64748b; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .card {
        background: #f8fafc;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2563eb;
        margin: 1rem 0;
    }
    .verdict-real {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        padding: 1.5rem; border-radius: 12px;
        border-left: 6px solid #16a34a; font-size: 1.3rem; font-weight: 700;
        color: #14532d;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        padding: 1.5rem; border-radius: 12px;
        border-left: 6px solid #dc2626; font-size: 1.3rem; font-weight: 700;
        color: #7f1d1d;
    }
    .challenge-box {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        padding: 1.2rem; border-radius: 12px;
        border: 1px dashed #2563eb; font-size: 1.15rem;
        color: #1e3a8a; font-weight: 600; margin: 0.5rem 0;
    }
    .stButton>button {
        border-radius: 8px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifact = joblib.load("voice_clone_model.pkl")
    return artifact["model"], joblib.load("scaler.pkl"), artifact["n_features"]

model, scaler, N_FEATURES = load_artifacts()

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "challenge" not in st.session_state:
    st.session_state.challenge = None
if "history" not in st.session_state:
    st.session_state.history = []

CHALLENGES = [
    "The quick brown fox jumps over the lazy dog.",
    "Verify transfer code seven three nine.",
    "My voice is my password, confirm now.",
    "Purple elephants dance at midnight.",
    "Confirm the transaction from your registered device.",
    "State the name of your first pet for verification.",
]

# ------------------------------------------------------------------
# Whisper (optional — only if installed)
# ------------------------------------------------------------------
@st.cache_resource
def load_whisper():
    try:
        import whisper
        return whisper.load_model("tiny")
    except Exception:
        return None

whisper_model = load_whisper()

def verify_challenge(audio_path, expected_phrase):
    """Return (matched: bool, transcribed: str) or (None, None) if Whisper unavailable."""
    if whisper_model is None:
        return None, None
    try:
        result = whisper_model.transcribe(audio_path, fp16=False)
        transcribed = result["text"].strip().lower()
        # Loose word-overlap match
        expected_words = set(expected_phrase.lower().split())
        spoken_words = set(transcribed.split())
        overlap = len(expected_words & spoken_words) / max(len(expected_words), 1)
        return overlap >= 0.6, transcribed
    except Exception as e:
        return None, f"STT error: {e}"

# ------------------------------------------------------------------
# Prediction helper
# ------------------------------------------------------------------
def predict(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    feats = extract_features(tmp_path).reshape(1, -1)
    if feats.shape[1] != N_FEATURES:
        os.unlink(tmp_path)
        raise ValueError(f"Feature mismatch: got {feats.shape[1]}, expected {N_FEATURES}")
    scaled = scaler.transform(feats)
    proba = model.predict_proba(scaled)[0]
    return tmp_path, float(proba[0]), float(proba[1])

# ------------------------------------------------------------------
# Feature importance chart
# ------------------------------------------------------------------
@st.cache_data
def get_feature_importance():
    imp = model.feature_importances_
    labels = (
        [f"MFCC {i+1}" for i in range(40)]
        + ["Spectral centroid", "Spectral bandwidth", "Spectral rolloff", "Zero-crossing rate"]
    )
    order = np.argsort(imp)[-10:][::-1]
    return [labels[i] for i in order], [float(imp[i]) for i in order]

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_home, tab_detect, tab_about = st.tabs(["🏠 Home", "🎯 Detect", "ℹ️ About"])

# ============ HOME ============
with tab_home:
    st.markdown('<div class="main-title">🛡️ AI Voice Cloning Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Real-time detection + active prevention of voice-cloning impersonation attacks.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 The Problem")
        st.markdown("Voice cloning fraud caused **$25B+ in losses in 2024**. Traditional detectors only flag fakes — attackers bypass them with new models.")
    with col2:
        st.markdown("### 🛡️ Our Solution")
        st.markdown("**Two layers:** passive XGBoost detection on 44 audio features + active challenge-response liveness verification.")
    with col3:
        st.markdown("### ⚡ Live Demo")
        st.markdown("Click the **🎯 Detect** tab above. Upload audio or use your mic. No install needed.")

    st.markdown("---")
    st.markdown("### 📊 Model at a Glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy (CV)", "82.9%", "+42% over baseline")
    m2.metric("Features", "44", "MFCC + spectral")
    m3.metric("Latency", "<1s", "Per 3-second clip")
    m4.metric("Devices", "Any", "Browser-based")

# ============ DETECT ============
with tab_detect:
    st.markdown("## 🎯 Detect & Prevent")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Step 1: Challenge
        st.markdown("### Step 1 — Liveness Challenge")
        if st.session_state.challenge is None:
            st.session_state.challenge = random.choice(CHALLENGES)
        st.markdown(
            f'<div class="challenge-box">🗣️ Read this aloud when recording:<br><br>"{st.session_state.challenge}"</div>',
            unsafe_allow_html=True
        )
        if st.button("🔄 New challenge"):
            st.session_state.challenge = random.choice(CHALLENGES)
            st.rerun()

        # Step 2: Audio
        st.markdown("### Step 2 — Submit Audio")
        sub1, sub2 = st.tabs(["📁 Upload", "🎙️ Record"])
        audio_bytes = None
        with sub1:
            up = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "ogg", "m4a"], label_visibility="collapsed")
            if up:
                audio_bytes = up.read()
                st.audio(audio_bytes)
        with sub2:
            rec = st.audio_input("Record your response", label_visibility="collapsed")
            if rec:
                audio_bytes = rec.read()
                st.audio(audio_bytes)

        # Step 3: Analyze
        st.markdown("### Step 3 — Analyze")
        threshold = st.slider("Detection threshold (fake probability)", 0.0, 1.0, 0.5, 0.05)

        if audio_bytes and st.button("🔍 Run Detection", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                try:
                    tmp_path, real_p, fake_p = predict(audio_bytes)

                    # Optional challenge verification
                    challenge_match, transcribed = (None, None)
                    if st.session_state.challenge:
                        challenge_match, transcribed = verify_challenge(
                            tmp_path, st.session_state.challenge
                        )
                    os.unlink(tmp_path)

                    is_fake = fake_p >= threshold

                    # Verdict
                    if is_fake:
                        st.markdown(
                            f'<div class="verdict-fake">🚨 FAKE / CLONED VOICE — {fake_p*100:.1f}% confidence</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="verdict-real">✅ REAL VOICE — {real_p*100:.1f}% confidence</div>',
                            unsafe_allow_html=True
                        )

                    # Metrics
                    c1, c2 = st.columns(2)
                    c1.metric("Real", f"{real_p*100:.1f}%")
                    c2.metric("Fake", f"{fake_p*100:.1f}%")
                    st.progress(fake_p)

                    # Challenge verification result
                    if challenge_match is True:
                        st.success(f"✅ Challenge phrase matched (Whisper confidence). Transcribed: \"{transcribed}\"")
                    elif challenge_match is False:
                        st.warning(f"⚠️ Challenge phrase did NOT match. Transcribed: \"{transcribed}\" — possible replay attack.")
                    else:
                        st.info("ℹ️ Whisper STT not installed — challenge verification skipped.")

                    # Save to history
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "real": f"{real_p*100:.1f}%",
                        "fake": f"{fake_p*100:.1f}%",
                        "verdict": "FAKE" if is_fake else "REAL",
                        "challenge": "✓" if challenge_match else ("✗" if challenge_match is False else "—"),
                    })

                    # Download report
                    report = f"""VOICE CLONING DETECTION REPORT
Generated: {datetime.now().isoformat()}
{"="*50}
Verdict:            {"FAKE / CLONED" if is_fake else "REAL"}
Real probability:   {real_p*100:.2f}%
Fake probability:   {fake_p*100:.2f}%
Threshold:          {threshold}
Challenge phrase:   {st.session_state.challenge}
Challenge matched:  {challenge_match}
Transcription:      {transcribed or "N/A"}
Model features:     {N_FEATURES}
{"="*50}
"""
                    st.download_button(
                        "📥 Download report",
                        data=report,
                        file_name=f"voice_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    with col_right:
        st.markdown("### 📈 Top 10 Model Features")
        labels, values = get_feature_importance()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.barh(labels[::-1], values[::-1], color="#2563eb")
        ax.set_xlabel("Importance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)

        st.markdown("### 📜 Session History")
        if st.session_state.history:
            st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)
        else:
            st.caption("No scans yet this session.")

# ============ ABOUT ============
with tab_about:
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
    ### 🎯 The Problem
    Voice cloning fraud caused **$25B+ in losses in 2024**. Attackers use AI tools to impersonate executives, family members, and bank customers. Traditional detectors only flag fakes — attackers bypass them by switching to a new synthesis model.

    ### 🛡️ Two-Layer Defense
    1. **Passive Detection** — XGBoost analyzes 44 audio features (40 MFCCs + spectral centroid, bandwidth, rolloff, zero-crossing rate) extracted via librosa to classify real vs synthesized voice.
    2. **Active Prevention** — A random challenge phrase forces the caller to speak live. Whisper STT verifies the phrase match. A pre-recorded clone can't produce a random phrase on demand.

    ### 🧠 Model Details
    - **Algorithm:** XGBoost Classifier
    - **Features:** 44-dim vector (40 MFCC + 4 spectral)
    - **Training Data:** Kaggle Voice Cloning Dataset
    - **Validation:** 5-fold stratified cross-validation
    - **Latency:** <1 second per 3-second clip on CPU

    ### 🚀 Tech Stack
    `Python` · `librosa` · `XGBoost` · `scikit-learn` · `Streamlit` · `OpenAI Whisper` (optional)

    ### ⚠️ Limitations
    - Trained on 8 source speakers — production would use ASVspoof 2019 with 10,000+ samples
    - 3-second clip scoring — real-time telephony streaming is on the v2 roadmap
    - Whisper challenge verification requires `openai-whisper` to be installed (optional)
    """)

st.markdown("---")
st.caption("🛡️ Hackathon prototype · XGBoost + librosa · Deployed on Streamlit Cloud")
