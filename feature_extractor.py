import librosa
import numpy as np

def extract_features(file_path, sr=16000, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=sr, duration=3.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw   = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff   = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr       = np.mean(librosa.feature.zero_crossing_rate(y))
    return np.hstack([mfcc_mean, spec_cent, spec_bw, rolloff, zcr])
    
