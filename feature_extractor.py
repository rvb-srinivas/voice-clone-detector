import librosa
import numpy as np

def extract_features(file_path, sr=16000, n_mfcc=40):
    # Load audio
    y, sr = librosa.load(file_path, sr=sr)
    
    # CRITICAL: Trim silence from beginning and end
    y, _ = librosa.effects.trim(y, top_db=25)
    
    # Pad or truncate to exactly 3 seconds
    target_len = int(sr * 3.0)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
        
    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw   = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff   = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr       = np.mean(librosa.feature.zero_crossing_rate(y))
    
    return np.hstack([mfcc_mean, spec_cent, spec_bw, rolloff, zcr])
