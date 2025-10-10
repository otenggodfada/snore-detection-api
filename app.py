import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from sklearn.metrics.pairwise import cosine_similarity
from librosa.sequence import dtw
import soundfile as sf
import os
from scipy.signal import butter, lfilter, find_peaks
import uuid

app = FastAPI(title="Snore Detection API", description="Analyze and compare real vs synthetic snores")

# --- Feature extractor ---
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1).reshape(1, -1)

# --- Advanced Snore Detection ---
def detect_snore_segments(audio, sr, min_duration=0.3, max_duration=3.0):
    """
    Detect individual snore events in audio using advanced signal processing.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        min_duration: Minimum snore duration in seconds
        max_duration: Maximum snore duration in seconds
    
    Returns:
        List of tuples (start_sample, end_sample, confidence_score)
    """
    # Apply bandpass filter for snoring frequencies (30-300 Hz)
    b, a = butter(4, [30/(0.5*sr), 300/(0.5*sr)], btype='band')
    filtered_audio = lfilter(b, a, audio)
    
    # Calculate energy envelope
    frame_length = int(0.05 * sr)  # 50ms windows
    hop_length = frame_length // 2
    
    # RMS energy
    rms = librosa.feature.rms(y=filtered_audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Spectral centroid (snores typically have low centroid)
    spectral_centroid = librosa.feature.spectral_centroid(y=filtered_audio, sr=sr, hop_length=hop_length)[0]
    
    # Zero crossing rate (snores have relatively low ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=filtered_audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Normalize features
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-8)
    centroid_norm = 1 - ((spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-8))
    zcr_norm = 1 - ((zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-8))
    
    # Combined confidence score (weighted combination)
    confidence = 0.5 * rms_norm + 0.3 * centroid_norm + 0.2 * zcr_norm
    
    # Smooth confidence curve
    confidence = np.convolve(confidence, np.ones(5)/5, mode='same')
    
    # Threshold detection
    threshold = np.percentile(confidence, 60)  # Adaptive threshold
    
    # Find regions above threshold
    above_threshold = confidence > threshold
    
    # Find segment boundaries
    segments = []
    in_segment = False
    segment_start = 0
    
    for i in range(len(above_threshold)):
        if above_threshold[i] and not in_segment:
            segment_start = i
            in_segment = True
        elif not above_threshold[i] and in_segment:
            # Convert frame indices to sample indices
            start_sample = segment_start * hop_length
            end_sample = i * hop_length
            duration = (end_sample - start_sample) / sr
            
            # Filter by duration
            if min_duration <= duration <= max_duration:
                # Calculate average confidence for this segment
                segment_confidence = np.mean(confidence[segment_start:i])
                segments.append((start_sample, end_sample, segment_confidence))
            
            in_segment = False
    
    # Handle case where audio ends in a segment
    if in_segment:
        start_sample = segment_start * hop_length
        end_sample = len(audio)
        duration = (end_sample - start_sample) / sr
        if min_duration <= duration <= max_duration:
            segment_confidence = np.mean(confidence[segment_start:])
            segments.append((start_sample, end_sample, segment_confidence))
    
    return segments

def extract_best_snore(audio, sr):
    """
    Extract the best quality snore from audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
    
    Returns:
        Tuple of (best_snore_audio, start_time, end_time, quality_metrics)
    """
    # Detect all snore segments
    segments = detect_snore_segments(audio, sr)
    
    if not segments:
        # No snores detected, return the original audio trimmed
        return audio, 0, len(audio)/sr, {
            "snores_detected": 0,
            "confidence": 0.0,
            "quality_score": 0.0
        }
    
    # Analyze each segment for quality
    best_segment = None
    best_score = -1
    
    for start, end, confidence in segments:
        segment_audio = audio[start:end]
        
        # Quality metrics
        # 1. Energy (louder is better)
        energy = np.sum(segment_audio ** 2) / len(segment_audio)
        
        # 2. Duration (prefer 0.5-2 seconds)
        duration = (end - start) / sr
        duration_score = 1.0 if 0.5 <= duration <= 2.0 else 0.7
        
        # 3. Spectral characteristics (check for snoring frequencies)
        spec = np.abs(librosa.stft(segment_audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Energy in snoring band (50-200 Hz)
        snore_band_mask = (freqs >= 50) & (freqs <= 200)
        snore_band_energy = np.sum(spec[snore_band_mask, :])
        total_energy = np.sum(spec) + 1e-8
        snore_ratio = snore_band_energy / total_energy
        
        # 4. Periodicity (snores have some periodic structure)
        autocorr = librosa.autocorrelate(segment_audio)
        # Find peaks in autocorrelation (excluding zero lag)
        peaks, _ = find_peaks(autocorr[sr//20:sr//2])  # Look for periods between 20-500ms
        periodicity_score = min(len(peaks) / 5.0, 1.0)  # Normalize
        
        # Combined quality score
        quality = (
            0.25 * min(energy * 100, 1.0) +  # Energy
            0.25 * duration_score +           # Duration
            0.25 * min(snore_ratio * 3, 1.0) +  # Spectral content
            0.15 * confidence +               # Detection confidence
            0.10 * periodicity_score          # Periodicity
        )
        
        if quality > best_score:
            best_score = quality
            best_segment = (start, end, confidence, quality)
    
    if best_segment:
        start, end, confidence, quality = best_segment
        best_snore = audio[start:end]
        
        # Add small fade in/out to avoid clicks
        fade_length = min(int(0.01 * sr), len(best_snore) // 10)
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        best_snore[:fade_length] *= fade_in
        best_snore[-fade_length:] *= fade_out
        
        return best_snore, start/sr, end/sr, {
            "snores_detected": len(segments),
            "confidence": float(confidence),
            "quality_score": float(quality),
            "duration": float((end - start) / sr)
        }
    
    return audio, 0, len(audio)/sr, {
        "snores_detected": len(segments),
        "confidence": 0.0,
        "quality_score": 0.0
    }

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
        
        # Extract the best snore segment using advanced detection
        best_snore, start_time, end_time, snore_metrics = extract_best_snore(y_trimmed, sr)
        
        # Save the detected best snore for download
        best_snore_file = f"/tmp/best_snore_{uuid.uuid4().hex}.wav"
        sf.write(best_snore_file, best_snore, sr)
        
        # Use best snore for analysis
        y = best_snore
        duration = librosa.get_duration(y=y, sr=sr)

        # Extract pitch & MFCCs from the best snore
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5'), sr=sr
        )
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # Generate synthetic snore based on best detected snore
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

        # Calculate additional snore characteristics
        # Spectral features
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        
        # Energy features
        rms_energy = float(np.mean(librosa.feature.rms(y=y)))
        
        # Pitch statistics
        valid_f0 = f0[np.isfinite(f0)]
        if len(valid_f0) > 0:
            pitch_mean = float(np.mean(valid_f0))
            pitch_std = float(np.std(valid_f0))
        else:
            pitch_mean = 0.0
            pitch_std = 0.0

        return {
            "detection_results": {
                "snores_detected": snore_metrics["snores_detected"],
                "detection_confidence": round(snore_metrics["confidence"], 3),
                "quality_score": round(snore_metrics["quality_score"], 3),
                "best_snore_timing": {
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "duration": round(snore_metrics.get("duration", duration), 2)
                }
            },
            "snore_characteristics": {
                "spectral_centroid_hz": round(spectral_centroid, 2),
                "spectral_rolloff_hz": round(spectral_rolloff, 2),
                "rms_energy": round(rms_energy, 4),
                "pitch_mean_hz": round(pitch_mean, 2),
                "pitch_std_hz": round(pitch_std, 2)
            },
            "comparison_metrics": {
                "cosine_similarity": round(cos_sim, 3),
                "dtw_distance": round(dtw_distance, 2)
            },
            "downloads": {
                "best_snore_url": f"/download/{os.path.basename(best_snore_file)}",
                "synthetic_snore_url": f"/download/{os.path.basename(synth_file)}"
            },
            "audio_info": {
                "original_duration": round(original_duration, 2),
                "processed_duration": round(duration, 2)
            }
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/detect_all_snores")
async def detect_all_snores(file: UploadFile = File(...)):
    """
    Detect all snore segments in the uploaded audio file.
    Returns detailed information about each detected snore.
    """
    try:
        # Save uploaded file
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load audio
        y, sr = librosa.load(file_path, sr=22050)
        original_duration = librosa.get_duration(y=y, sr=sr)
        
        # Remove silence
        y_trimmed = remove_silence(y, sr, silence_threshold=0.01, min_silence_duration=0.5)
        
        # Detect all snore segments
        segments = detect_snore_segments(y_trimmed, sr)
        
        # Analyze each segment
        snore_details = []
        for idx, (start, end, confidence) in enumerate(segments):
            segment_audio = y_trimmed[start:end]
            duration = (end - start) / sr
            
            # Calculate energy
            energy = float(np.mean(segment_audio ** 2))
            
            # Calculate spectral features
            spec = np.abs(librosa.stft(segment_audio))
            freqs = librosa.fft_frequencies(sr=sr)
            snore_band_mask = (freqs >= 50) & (freqs <= 200)
            snore_band_energy = np.sum(spec[snore_band_mask, :])
            total_energy = np.sum(spec) + 1e-8
            snore_ratio = float(snore_band_energy / total_energy)
            
            # Calculate pitch
            f0, _, _ = librosa.pyin(segment_audio, fmin=50, fmax=300, sr=sr)
            valid_f0 = f0[np.isfinite(f0)]
            pitch_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
            
            snore_details.append({
                "segment_id": idx + 1,
                "start_time": round(start / sr, 2),
                "end_time": round(end / sr, 2),
                "duration": round(duration, 2),
                "confidence": round(float(confidence), 3),
                "energy": round(energy, 6),
                "snore_frequency_ratio": round(snore_ratio, 3),
                "pitch_hz": round(pitch_mean, 2)
            })
        
        return {
            "total_snores_detected": len(segments),
            "audio_duration": round(original_duration, 2),
            "snore_segments": snore_details,
            "summary": {
                "avg_snore_duration": round(np.mean([s["duration"] for s in snore_details]), 2) if snore_details else 0,
                "avg_confidence": round(np.mean([s["confidence"] for s in snore_details]), 3) if snore_details else 0,
                "highest_quality_segment": max(snore_details, key=lambda x: x["confidence"])["segment_id"] if snore_details else None
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

@app.get("/")
async def root():
    return {
        "message": "Snore Detection API - Advanced Snore Analysis",
        "version": "2.0",
        "endpoints": {
            "/analyze_snore": "POST - Analyze audio and extract the best snore with detailed metrics",
            "/detect_all_snores": "POST - Detect and return all snore segments in the audio",
            "/download/{filename}": "GET - Download processed audio files"
        },
        "features": [
            "Advanced snore detection using signal processing",
            "Multi-feature analysis (energy, spectral, pitch, periodicity)",
            "Best snore extraction with quality scoring",
            "Bandpass filtering for snoring frequencies (30-300 Hz)",
            "Adaptive threshold detection",
            "Synthetic snore generation for comparison"
        ]
    }
