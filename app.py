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

# --- Feature extractor ---
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1).reshape(1, -1)

# --- Remove silent portions from audio ---
def remove_silence(audio, sr, silence_threshold=0.01, min_silence_duration=0.5):
    """
    Remove silent portions from the beginning and end of audio.
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        silence_threshold: RMS threshold below which audio is considered silent (default: 0.01)
        min_silence_duration: Minimum duration of silence to be removed in seconds (default: 0.5)
    
    Returns:
        Trimmed audio signal
    """
    # Calculate RMS energy in small windows
    frame_length = int(0.1 * sr)  # 100ms windows
    hop_length = frame_length // 2
    
    # Calculate RMS for each frame
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find frames below silence threshold
    silent_frames = rms < silence_threshold
    
    # Convert frame indices to sample indices
    silent_samples = np.zeros(len(audio), dtype=bool)
    for i, is_silent in enumerate(silent_frames):
        start_sample = i * hop_length
        end_sample = min(start_sample + frame_length, len(audio))
        silent_samples[start_sample:end_sample] = is_silent
    
    # Find continuous silent regions
    silent_regions = []
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silent_samples):
        if is_silent and not in_silence:
            # Start of silence
            silence_start = i
            in_silence = True
        elif not is_silent and in_silence:
            # End of silence
            silence_duration = (i - silence_start) / sr
            if silence_duration >= min_silence_duration:
                silent_regions.append((silence_start, i))
            in_silence = False
    
    # Handle case where audio ends with silence
    if in_silence:
        silence_duration = (len(silent_samples) - silence_start) / sr
        if silence_duration >= min_silence_duration:
            silent_regions.append((silence_start, len(silent_samples)))
    
    # Remove silent regions from beginning and end
    if not silent_regions:
        return audio
    
    # Find the first non-silent region
    first_silent_end = silent_regions[0][1] if silent_regions[0][0] == 0 else 0
    
    # Find the last non-silent region
    last_silent_start = silent_regions[-1][0] if silent_regions[-1][1] == len(audio) else len(audio)
    
    # Extract the non-silent portion
    trimmed_audio = audio[first_silent_end:last_silent_start]
    
    # Ensure we don't return empty audio
    if len(trimmed_audio) == 0:
        return audio
    
    return trimmed_audio

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
        original_duration = librosa.get_duration(y=y, sr=sr)
        
        # Remove silent portions from beginning and end
        y_trimmed = remove_silence(y, sr, silence_threshold=0.01, min_silence_duration=0.5)
        
        # Use trimmed audio for analysis
        y = y_trimmed
        duration = librosa.get_duration(y=y, sr=sr)
        trimmed_duration = original_duration - duration

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
            "cosine_similarity": cos_sim,
            "dtw_distance": dtw_distance,
            "download_url": f"/download/{os.path.basename(synth_file)}",
            "audio_info": {
                "original_duration": round(original_duration, 2),
                "trimmed_duration": round(duration, 2),
                "silence_removed": round(trimmed_duration, 2)
            }
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"/tmp/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    return JSONResponse({"error": "File not found"}, status_code=404)
