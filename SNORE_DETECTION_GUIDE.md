# Advanced Snore Detection API - User Guide

## Overview
This API now features **advanced snore detection** that can accurately identify and extract the best snore sounds from any audio file. The system uses multiple signal processing techniques to ensure high-quality snore detection.

## Key Features

### 🎯 Advanced Detection Algorithm
The system uses a multi-feature approach:
- **Bandpass Filtering**: Focuses on snoring frequencies (30-300 Hz)
- **Energy Analysis**: Detects sound intensity patterns typical of snoring
- **Spectral Analysis**: Identifies frequency characteristics of snores
- **Zero Crossing Rate**: Filters out non-snore sounds
- **Adaptive Thresholding**: Automatically adjusts to different audio qualities

### 🏆 Quality Scoring
Each detected snore is scored based on:
1. **Energy** (25%): Louder, clearer snores score higher
2. **Duration** (25%): Optimal duration of 0.5-2 seconds
3. **Spectral Content** (25%): Energy in typical snoring bands (50-200 Hz)
4. **Detection Confidence** (15%): Overall signal quality
5. **Periodicity** (10%): Rhythmic breathing patterns

## API Endpoints

### 1. `/analyze_snore` - Extract Best Snore
**Purpose**: Automatically detects and extracts the highest quality snore from audio

**Example Request**:
```bash
curl -X POST "http://localhost:8000/analyze_snore" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_snore_recording.wav"
```

**Response Structure**:
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
    "spectral_rolloff_hz": 287.56,
    "rms_energy": 0.0234,
    "pitch_mean_hz": 98.45,
    "pitch_std_hz": 12.34
  },
  "comparison_metrics": {
    "cosine_similarity": 0.856,
    "dtw_distance": 234.56
  },
  "downloads": {
    "best_snore_url": "/download/best_snore_abc123.wav",
    "synthetic_snore_url": "/download/synthetic_abc124.wav"
  }
}
```

### 2. `/detect_all_snores` - Find All Snore Segments
**Purpose**: Detects and analyzes ALL snore events in the audio

**Example Request**:
```bash
curl -X POST "http://localhost:8000/detect_all_snores" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_snore_recording.wav"
```

**Response Structure**:
```json
{
  "total_snores_detected": 5,
  "audio_duration": 30.5,
  "snore_segments": [
    {
      "segment_id": 1,
      "start_time": 1.23,
      "end_time": 2.45,
      "duration": 1.22,
      "confidence": 0.856,
      "energy": 0.0234,
      "snore_frequency_ratio": 0.678,
      "pitch_hz": 95.67
    },
    ...
  ],
  "summary": {
    "avg_snore_duration": 1.45,
    "avg_confidence": 0.823,
    "highest_quality_segment": 3
  }
}
```

## How It Works

### Detection Process
1. **Audio Loading**: Loads audio at 22,050 Hz sample rate
2. **Silence Removal**: Removes silent portions from beginning/end
3. **Bandpass Filtering**: Applies 30-300 Hz filter to focus on snoring frequencies
4. **Multi-Feature Extraction**:
   - RMS energy (sound intensity)
   - Spectral centroid (frequency distribution)
   - Zero crossing rate (signal complexity)
5. **Confidence Scoring**: Combines features with weighted formula
6. **Segment Detection**: Identifies continuous regions above adaptive threshold
7. **Quality Analysis**: Ranks segments by energy, duration, spectral content, and periodicity
8. **Best Selection**: Returns the highest scoring snore segment

### Why It Detects Snores Accurately

#### Frequency Filtering
Snores typically occur in the 30-300 Hz range. The bandpass filter eliminates:
- High-frequency noise (speech, coughs, rustling)
- Very low-frequency rumble (environmental noise)

#### Multi-Feature Approach
Using multiple features ensures robust detection:
- **Energy**: Snores are louder than background noise
- **Spectral Centroid**: Snores have low-frequency dominance
- **Zero Crossing Rate**: Snores have smoother waveforms than speech

#### Adaptive Thresholding
The system uses percentile-based thresholds (60th percentile) that adapt to:
- Different recording volumes
- Various background noise levels
- Different microphone qualities

#### Duration Filtering
Snores typically last 0.3-3.0 seconds, filtering out:
- Brief coughs or clicks
- Extended breathing or speech

## Tips for Best Results

### Recording Quality
- **Quiet Environment**: Minimize background noise
- **Close Microphone**: Place device 1-2 meters from sleeper
- **Long Recording**: 30+ seconds increases chance of capturing clear snores
- **Good Format**: WAV or high-quality MP3

### Expected Performance
- **High Confidence (>0.8)**: Clear, isolated snore sounds
- **Medium Confidence (0.5-0.8)**: Snores with background noise
- **Low Confidence (<0.5)**: Noisy environment or unclear snoring

### Interpreting Results

#### Quality Score
- **0.8-1.0**: Excellent quality snore, ideal for analysis
- **0.6-0.8**: Good quality, usable for most purposes
- **0.4-0.6**: Fair quality, may have noise interference
- **<0.4**: Poor quality, consider re-recording

#### Snore Characteristics
- **Spectral Centroid**: Lower values (100-200 Hz) = typical snoring
- **Pitch Mean**: 80-150 Hz is common for snoring
- **Snore Frequency Ratio**: >0.5 indicates strong snoring characteristics

## Example Use Cases

### Sleep Study Analysis
Use `/detect_all_snores` to count snoring events throughout the night:
```python
import requests

response = requests.post(
    "http://localhost:8000/detect_all_snores",
    files={"file": open("night_recording.wav", "rb")}
)
data = response.json()
print(f"Total snores detected: {data['total_snores_detected']}")
print(f"Average snore duration: {data['summary']['avg_snore_duration']}s")
```

### Snore Sample Extraction
Use `/analyze_snore` to extract the best snore for further analysis:
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze_snore",
    files={"file": open("snoring_audio.wav", "rb")}
)
data = response.json()

# Download the best snore
best_snore_url = data['downloads']['best_snore_url']
snore_file = requests.get(f"http://localhost:8000{best_snore_url}")
with open("best_snore.wav", "wb") as f:
    f.write(snore_file.content)
```

## Technical Specifications

### Signal Processing
- **Sample Rate**: 22,050 Hz (optimized for snore detection)
- **Filter Type**: Butterworth 4th order bandpass
- **Window Size**: 50ms frames with 25ms hop
- **MFCC Coefficients**: 20 coefficients for feature extraction

### Detection Parameters
- **Min Duration**: 0.3 seconds
- **Max Duration**: 3.0 seconds
- **Confidence Threshold**: Adaptive (60th percentile)
- **Fade In/Out**: 10ms to prevent audio artifacts

### Performance
- **Processing Speed**: ~2-5 seconds for 30-second audio
- **Memory Usage**: ~100-200 MB per request
- **Accuracy**: High precision in quiet to moderate noise environments

## Troubleshooting

### No Snores Detected
- Check if audio actually contains snoring sounds
- Ensure audio is not too quiet (increase recording volume)
- Try longer recording duration
- Reduce background noise

### Low Confidence Scores
- Improve recording quality
- Move microphone closer to source
- Reduce background noise
- Use better quality audio format

### Multiple False Positives
- Ensure audio doesn't contain speech or other periodic sounds
- Check for environmental noise (fans, traffic)
- The system may detect rhythmic breathing as snores

## Running the API

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
# Open browser to: http://localhost:8000/docs
```

## API Information
Visit the root endpoint for API information:
```bash
curl http://localhost:8000/
```

---

**Version**: 2.0  
**Algorithm**: Multi-feature signal processing with adaptive thresholding  
**Accuracy**: Optimized for quiet to moderate noise environments

