#!/usr/bin/env python3
"""
Test script for the Advanced Snore Detection API
This script demonstrates how to use the API endpoints and interpret results.
"""

import requests
import json
import sys
import os

# Configuration
API_BASE_URL = "http://localhost:8000"

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def test_api_status():
    """Test if API is running"""
    print_header("Testing API Status")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("✓ API is running!")
            print(f"Version: {data.get('version', 'N/A')}")
            print("\nAvailable Endpoints:")
            for endpoint, description in data.get('endpoints', {}).items():
                print(f"  • {endpoint}: {description}")
            return True
        else:
            print("✗ API returned unexpected status code:", response.status_code)
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API. Make sure the server is running:")
        print("  uvicorn app:app --reload --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def analyze_snore(audio_file):
    """Test the /analyze_snore endpoint"""
    print_header(f"Analyzing Best Snore from: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f"✗ File not found: {audio_file}")
        return
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (os.path.basename(audio_file), f, 'audio/wav')}
            response = requests.post(f"{API_BASE_URL}/analyze_snore", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Analysis Complete!\n")
            
            # Detection Results
            detection = data.get('detection_results', {})
            print("📊 DETECTION RESULTS")
            print(f"  Snores Detected: {detection.get('snores_detected', 0)}")
            print(f"  Detection Confidence: {detection.get('detection_confidence', 0):.3f}")
            print(f"  Quality Score: {detection.get('quality_score', 0):.3f}")
            
            timing = detection.get('best_snore_timing', {})
            print(f"\n  Best Snore Timing:")
            print(f"    Start: {timing.get('start_time', 0):.2f}s")
            print(f"    End: {timing.get('end_time', 0):.2f}s")
            print(f"    Duration: {timing.get('duration', 0):.2f}s")
            
            # Snore Characteristics
            chars = data.get('snore_characteristics', {})
            print(f"\n🔊 SNORE CHARACTERISTICS")
            print(f"  Spectral Centroid: {chars.get('spectral_centroid_hz', 0):.2f} Hz")
            print(f"  Spectral Rolloff: {chars.get('spectral_rolloff_hz', 0):.2f} Hz")
            print(f"  RMS Energy: {chars.get('rms_energy', 0):.4f}")
            print(f"  Average Pitch: {chars.get('pitch_mean_hz', 0):.2f} Hz")
            print(f"  Pitch Variation: {chars.get('pitch_std_hz', 0):.2f} Hz")
            
            # Quality Assessment
            quality = detection.get('quality_score', 0)
            print(f"\n⭐ QUALITY ASSESSMENT")
            if quality >= 0.8:
                print(f"  Excellent quality snore! (Score: {quality:.3f})")
                print("  This is ideal for detailed analysis.")
            elif quality >= 0.6:
                print(f"  Good quality snore. (Score: {quality:.3f})")
                print("  Suitable for most purposes.")
            elif quality >= 0.4:
                print(f"  Fair quality snore. (Score: {quality:.3f})")
                print("  May have some noise interference.")
            else:
                print(f"  Low quality detection. (Score: {quality:.3f})")
                print("  Consider re-recording in a quieter environment.")
            
            # Downloads
            downloads = data.get('downloads', {})
            print(f"\n💾 DOWNLOAD LINKS")
            print(f"  Best Snore: {API_BASE_URL}{downloads.get('best_snore_url', '')}")
            print(f"  Synthetic Snore: {API_BASE_URL}{downloads.get('synthetic_snore_url', '')}")
            
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"✗ Error during analysis: {e}")

def detect_all_snores(audio_file):
    """Test the /detect_all_snores endpoint"""
    print_header(f"Detecting All Snores in: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f"✗ File not found: {audio_file}")
        return
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (os.path.basename(audio_file), f, 'audio/wav')}
            response = requests.post(f"{API_BASE_URL}/detect_all_snores", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Detection Complete!\n")
            
            print(f"📈 SUMMARY")
            print(f"  Total Snores: {data.get('total_snores_detected', 0)}")
            print(f"  Audio Duration: {data.get('audio_duration', 0):.2f}s")
            
            summary = data.get('summary', {})
            print(f"  Average Snore Duration: {summary.get('avg_snore_duration', 0):.2f}s")
            print(f"  Average Confidence: {summary.get('avg_confidence', 0):.3f}")
            print(f"  Best Segment ID: {summary.get('highest_quality_segment', 'N/A')}")
            
            # Individual segments
            segments = data.get('snore_segments', [])
            if segments:
                print(f"\n🎯 DETECTED SEGMENTS (Top 5)")
                # Sort by confidence and show top 5
                sorted_segments = sorted(segments, key=lambda x: x['confidence'], reverse=True)[:5]
                
                for segment in sorted_segments:
                    print(f"\n  Segment #{segment['segment_id']}")
                    print(f"    Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s ({segment['duration']:.2f}s)")
                    print(f"    Confidence: {segment['confidence']:.3f}")
                    print(f"    Energy: {segment['energy']:.6f}")
                    print(f"    Snore Frequency Ratio: {segment['snore_frequency_ratio']:.3f}")
                    print(f"    Pitch: {segment['pitch_hz']:.2f} Hz")
            else:
                print("\n  No snore segments detected.")
                print("  Tips:")
                print("    - Ensure audio contains actual snoring sounds")
                print("    - Check recording volume is adequate")
                print("    - Try a longer recording")
                
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"✗ Error during detection: {e}")

def download_audio(url, output_file):
    """Download processed audio file"""
    print(f"\nDownloading: {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Saved to: {output_file}")
        else:
            print(f"✗ Download failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    """Main test function"""
    print_header("Advanced Snore Detection API - Test Script")
    
    # Check if API is running
    if not test_api_status():
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <audio_file.wav> [mode]")
        print("\nModes:")
        print("  analyze - Extract and analyze best snore (default)")
        print("  detect  - Detect all snore segments")
        print("  both    - Run both analyses")
        print("\nExample:")
        print(f"  python {sys.argv[0]} snoring.wav both")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "analyze"
    
    # Run appropriate tests
    if mode in ["analyze", "both"]:
        analyze_snore(audio_file)
    
    if mode in ["detect", "both"]:
        detect_all_snores(audio_file)
    
    print_header("Test Complete")
    print("For detailed API documentation, see: SNORE_DETECTION_GUIDE.md")
    print("For interactive API docs, visit: http://localhost:8000/docs")

if __name__ == "__main__":
    main()

