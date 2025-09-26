import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from sklearn.metrics.pairwise import cosine_similarity
from librosa.sequence import dtw
import soundfile as sf
import os
from scipy.signal import butter, lfilter
import uuid

app = FastAPI(title="Snore Detection API", description="Analyze and compare real vs synthetic snores")

# --- Music detection ---
def is_music(audio, sr):
    """
    Detect if audio is music based on various audio features.
    Returns True if music is detected, False otherwise.
    """
    try:
        # Extract tempo (music typically has consistent tempo)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Extract spectral centroid (music has more varied spectral content)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_centroid_std = np.std(spectral_centroids)
        
        # Extract zero crossing rate (music has more complex patterns)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_std = np.std(zcr)
        
        # Extract chroma features (music has harmonic structure)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_std = np.std(chroma)
        
        # Extract rhythm regularity (music has more regular rhythm)
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        if len(onset_frames) > 1:
            onset_intervals = np.diff(onset_frames)
            rhythm_regularity = 1.0 / (1.0 + np.std(onset_intervals) / np.mean(onset_intervals))
        else:
            rhythm_regularity = 0
        
        # Music detection criteria
        is_tempo_consistent = 60 <= tempo <= 200  # Typical music tempo range
        has_varied_spectrum = spectral_centroid_std > 200  # Music has more spectral variation
        has_complex_patterns = zcr_std > 0.01  # Music has more complex zero-crossing patterns
        has_harmonic_structure = chroma_std > 0.1  # Music has harmonic content
        has_regular_rhythm = rhythm_regularity > 0.3  # Music has more regular rhythm
        
        # Count how many music characteristics are present
        music_indicators = sum([
            is_tempo_consistent,
            has_varied_spectrum,
            has_complex_patterns,
            has_harmonic_structure,
            has_regular_rhythm
        ])
        
        # If 3 or more indicators suggest music, classify as music
        return music_indicators >= 3
        
    except Exception as e:
        # If analysis fails, assume it's not music (safer for snore detection)
        return False

# --- Feature extractor ---
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1).reshape(1, -1)

# --- Generate synthetic snore ---
def generate_snore(duration, sample_rate, pitch_contour, mfcc_template):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    raw_noise = np.random.normal(0, 0.3, len(t))

    # Breathing envelope
    base_env = (np.sin(2 * np.pi * 0.25 * t) + 1) / 2
    jitter = 0.05 * np.sin(2 * np.pi * 0.05 * t)
    breath_env = np.clip(base_env + jitter, 0, 1)

    # Pitch modulation
    pitch_interp = np.interp(t, np.linspace(0, duration, len(pitch_contour)), pitch_contour)
    base_pitch = np.nanmedian(pitch_interp[np.isfinite(pitch_interp)]) if np.any(np.isfinite(pitch_interp)) else 100
    pitch_interp[~np.isfinite(pitch_interp)] = base_pitch
    phase = 2 * np.pi * np.cumsum(pitch_interp) / sample_rate
    pitch_modulation = 1 + 0.1 * np.sin(phase)

    # MFCC influence
    mfcc_energy = np.mean(mfcc_template[:5, :], axis=0)
    mfcc_interp = np.interp(t, np.linspace(0, duration, mfcc_energy.shape[0]), mfcc_energy)

    # Rasping component
    rasp_noise = np.random.normal(0, 0.5, len(t))
    b, a = butter(4, [50/(0.5*sample_rate), 300/(0.5*sample_rate)], btype='band')
    rasp = lfilter(b, a, rasp_noise)
    rasp = np.tanh(rasp * 2) * (breath_env**3) * (1 + 0.5 * mfcc_interp)

    # Nasal component
    b, a = butter(4, [250/(0.5*sample_rate), 650/(0.5*sample_rate)], btype='band')
    nasal = 0.2 * lfilter(b, a, raw_noise) * breath_env * (1 + 0.3 * mfcc_interp)

    snore = 0.6 * raw_noise * breath_env * pitch_modulation + 0.7 * rasp + 0.3 * nasal
    return np.nan_to_num(snore)

@app.post("/analyze_snore")
async def analyze_snore(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load real snore
        y, sr = librosa.load(file_path, sr=22050)
        
        # Amplify the audio to make it louder (increase volume by 3dB)
        amplification_factor = 2.0  # 2x amplification = ~6dB increase
        y = y * amplification_factor
        
        # Normalize to prevent clipping while maintaining the increased volume
        y = librosa.util.normalize(y, norm=np.inf)
        
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Check if the audio is music - if so, don't detect as snore
        if is_music(y, sr):
            return {
                "is_music": True,
                "message": "Music detected - snore analysis skipped",
                "cosine_similarity": None,
                "dtw_distance": None,
                "download_url": None
            }

        # Extract pitch & MFCCs
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5'), sr=sr
        )
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # Generate synthetic snore
        synthetic = generate_snore(duration, sr, f0, mfccs)

        # Save synthetic file
        synth_file = f"/tmp/synthetic_{uuid.uuid4().hex}.wav"
        sf.write(synth_file, synthetic, sr)

        # Feature similarity
        feat_real = extract_features(y, sr)
        feat_synth = extract_features(synthetic, sr)
        cos_sim = float(cosine_similarity(feat_real, feat_synth)[0][0])

        # DTW distance
        mfcc_real = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_synth = librosa.feature.mfcc(y=synthetic, sr=sr, n_mfcc=20)
        D, _ = dtw(mfcc_real.T, mfcc_synth.T, metric='euclidean')
        dtw_distance = float(D[-1, -1])

        return {
            "is_music": False,
            "message": "Snore analysis completed",
            "cosine_similarity": cos_sim,
            "dtw_distance": dtw_distance,
            "download_url": f"/download/{os.path.basename(synth_file)}"
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"/tmp/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    return JSONResponse({"error": "File not found"}, status_code=404)
