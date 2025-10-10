# Snore Detection API 🌙

Advanced snore detection and analysis system using state-of-the-art signal processing techniques.

## 🚀 Features

### Advanced Detection Algorithm
- **Multi-Feature Analysis**: Combines energy, spectral, and temporal features for robust detection
- **Bandpass Filtering**: Focuses on snoring frequencies (30-300 Hz) to eliminate false positives
- **Adaptive Thresholding**: Automatically adjusts to different recording conditions
- **Quality Scoring**: Rates each snore based on energy, duration, spectral content, and periodicity
- **Best Snore Extraction**: Automatically selects the highest quality snore from recordings

### What Makes This Detection Accurate?

1. **Frequency-Focused**: Filters specifically for snoring frequencies (30-300 Hz), eliminating speech, coughs, and other non-snore sounds

2. **Multi-Feature Approach**:
   - **RMS Energy**: Identifies loud snoring events
   - **Spectral Centroid**: Detects low-frequency dominance typical of snores
   - **Zero Crossing Rate**: Filters smooth snoring waveforms from complex sounds
   
3. **Quality Metrics**: Each snore scored by:
   - Energy level (25%)
   - Duration appropriateness (25%)
   - Spectral characteristics (25%)
   - Detection confidence (15%)
   - Periodic breathing patterns (10%)

4. **Smart Filtering**:
   - Minimum duration: 0.3s (eliminates clicks, coughs)
   - Maximum duration: 3.0s (eliminates speech, long breathing)
   - Silence removal from analysis
   - Fade in/out to prevent audio artifacts

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd snore-api

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 🎯 Quick Start

### 1. Start the Server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test with Sample Audio
```bash
# Test the detection
python test_snore_detection.py your_audio_file.wav both
```

### 3. Access Interactive Docs
Open your browser to: `http://localhost:8000/docs`

## 🔧 API Endpoints

### `/analyze_snore` - POST
Analyzes audio and extracts the **best quality snore** with detailed metrics.

**Features**:
- Automatically finds the highest quality snore in the recording
- Returns detailed characteristics (pitch, energy, spectral features)
- Generates synthetic snore for comparison
- Provides downloadable audio of the best detected snore

**Example**:
```bash
curl -X POST "http://localhost:8000/analyze_snore" \
  -F "file=@snoring.wav"
```

### `/detect_all_snores` - POST
Detects **all snore events** in the audio with individual analysis.

**Features**:
- Returns timing and characteristics of every snore
- Ranks snores by quality
- Provides summary statistics
- Useful for sleep study analysis

**Example**:
```bash
curl -X POST "http://localhost:8000/detect_all_snores" \
  -F "file=@night_recording.wav"
```

### `/download/{filename}` - GET
Downloads processed audio files (best snore or synthetic snore).

## 📊 Response Examples

### Analyze Snore Response
```json
{
  "detection_results": {
    "snores_detected": 5,
    "detection_confidence": 0.847,
    "quality_score": 0.923,
    "best_snore_timing": {
      "start_time": 2.34,
      "end_time": 3.89,
      "duration": 1.55
    }
  },
  "snore_characteristics": {
    "spectral_centroid_hz": 145.23,
    "rms_energy": 0.0234,
    "pitch_mean_hz": 98.45
  },
  "downloads": {
    "best_snore_url": "/download/best_snore_abc.wav",
    "synthetic_snore_url": "/download/synthetic_abc.wav"
  }
}
```

## 🎓 Understanding the Results

### Detection Confidence
- **0.8 - 1.0**: High confidence - clear snore detection
- **0.5 - 0.8**: Medium confidence - snore with some noise
- **< 0.5**: Low confidence - noisy environment

### Quality Score
- **0.8 - 1.0**: Excellent - ideal for detailed analysis
- **0.6 - 0.8**: Good - suitable for most purposes
- **0.4 - 0.6**: Fair - some noise interference
- **< 0.4**: Poor - consider re-recording

### Snore Characteristics
- **Spectral Centroid**: 100-200 Hz is typical for snoring
- **Pitch Mean**: 80-150 Hz is common
- **RMS Energy**: Higher values indicate louder snores

## 💡 Tips for Best Results

### Recording Guidelines
1. **Quiet Environment**: Minimize background noise
2. **Proper Distance**: Place microphone 1-2 meters from sleeper
3. **Duration**: Record for 30+ seconds to capture multiple snores
4. **Format**: Use WAV or high-quality audio formats

### Common Issues

**No Snores Detected?**
- Ensure audio actually contains snoring
- Check recording volume
- Try longer recording duration
- Reduce background noise

**Low Confidence Scores?**
- Improve recording quality
- Move microphone closer
- Use quieter environment

## 🧪 Testing

Use the included test script:

```bash
# Test analysis endpoint
python test_snore_detection.py audio.wav analyze

# Test detection endpoint
python test_snore_detection.py audio.wav detect

# Test both endpoints
python test_snore_detection.py audio.wav both
```

## 📚 Documentation

- **Complete Guide**: See `SNORE_DETECTION_GUIDE.md` for detailed documentation
- **API Docs**: Visit `http://localhost:8000/docs` after starting the server
- **Technical Details**: See inline code documentation in `app.py`

## 🛠️ Technology Stack

- **FastAPI**: Modern web framework
- **Librosa**: Audio analysis and feature extraction
- **NumPy/SciPy**: Signal processing and scientific computing
- **Scikit-learn**: Machine learning utilities

## 🔬 Signal Processing Details

### Bandpass Filter
- **Type**: Butterworth 4th order
- **Range**: 30-300 Hz (snoring frequency band)
- **Purpose**: Eliminates speech, environmental noise

### Feature Extraction
- **MFCC**: 20 coefficients for pattern recognition
- **Frame Size**: 50ms windows with 25ms hop
- **Sample Rate**: 22,050 Hz (optimized for snore detection)

### Detection Algorithm
1. Load and preprocess audio
2. Apply bandpass filter for snoring frequencies
3. Extract multi-feature confidence scores
4. Apply adaptive threshold (60th percentile)
5. Identify continuous snore segments
6. Score segments by quality metrics
7. Return best snore or all segments

## 🚀 Deployment

### Docker (Recommended)
```bash
# Build image
docker build -t snore-api .

# Run container
docker run -p 8000:8000 snore-api
```

### Google Cloud Platform
```bash
# Deploy to Cloud Run
gcloud builds submit --config cloudbuild.yaml
```

## 📈 Performance

- **Processing Speed**: 2-5 seconds for 30-second audio
- **Memory Usage**: ~100-200 MB per request
- **Accuracy**: High precision in quiet to moderate noise environments
- **Supported Formats**: WAV, MP3, FLAC, and other librosa-supported formats

## 🤝 Use Cases

1. **Sleep Studies**: Analyze snoring patterns throughout the night
2. **Medical Research**: Extract snore samples for acoustic analysis
3. **Sleep Apps**: Provide users with snore detection and tracking
4. **Audio Processing**: Isolate and study snoring sounds

## 📝 License

See LICENSE file for details.

## 🙋 Support

For detailed usage instructions and troubleshooting, see `SNORE_DETECTION_GUIDE.md`.

---

**Version**: 2.0  
**Detection Algorithm**: Multi-feature signal processing with adaptive thresholding  
**Optimized For**: Accurate snore detection in various recording conditions

