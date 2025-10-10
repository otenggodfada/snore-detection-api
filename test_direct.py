#!/usr/bin/env python3
"""
Direct test of snore detection without requiring server
"""
import sys
import numpy as np
import librosa
from app import extract_best_snore, detect_snore_segments, remove_silence

def test_snore_file(audio_file):
    print("="*60)
    print(f"  Testing: {audio_file}")
    print("="*60)
    print()
    
    try:
        # Load audio
        print("📂 Loading audio file...")
        y, sr = librosa.load(audio_file, sr=22050)
        original_duration = librosa.get_duration(y=y, sr=sr)
        print(f"   Duration: {original_duration:.2f} seconds")
        print(f"   Sample rate: {sr} Hz")
        print()
        
        # Remove silence
        print("🔇 Removing silence...")
        y_trimmed = remove_silence(y, sr, silence_threshold=0.01, min_silence_duration=0.5)
        trimmed_duration = librosa.get_duration(y=y_trimmed, sr=sr)
        print(f"   After trimming: {trimmed_duration:.2f} seconds")
        print()
        
        # Detect all snores
        print("🔍 Detecting snore segments...")
        segments = detect_snore_segments(y_trimmed, sr)
        print(f"   Found {len(segments)} snore segment(s)")
        print()
        
        if len(segments) > 0:
            print("📊 SNORE SEGMENTS DETECTED:")
            for idx, (start, end, confidence) in enumerate(segments, 1):
                duration = (end - start) / sr
                print(f"\n   Segment #{idx}:")
                print(f"      Time: {start/sr:.2f}s - {end/sr:.2f}s")
                print(f"      Duration: {duration:.2f}s")
                print(f"      Confidence: {confidence:.3f}")
            print()
        
        # Extract best snore
        print("⭐ Finding best snore...")
        best_snore, start_time, end_time, metrics = extract_best_snore(y_trimmed, sr)
        print()
        
        # Display results
        print("="*60)
        print("  RESULTS")
        print("="*60)
        print()
        
        print("🎯 DETECTION SUMMARY:")
        print(f"   Snores Detected: {metrics['snores_detected']}")
        print(f"   Detection Confidence: {metrics['confidence']:.3f}")
        print(f"   Quality Score: {metrics['quality_score']:.3f}")
        print()
        
        if metrics['snores_detected'] > 0:
            print("⏱️  BEST SNORE TIMING:")
            print(f"   Start Time: {start_time:.2f}s")
            print(f"   End Time: {end_time:.2f}s")
            print(f"   Duration: {metrics.get('duration', end_time - start_time):.2f}s")
            print()
            
            # Quality assessment
            quality = metrics['quality_score']
            print("⭐ QUALITY ASSESSMENT:")
            if quality >= 0.8:
                print(f"   ✅ EXCELLENT quality! (Score: {quality:.3f})")
                print("   This is ideal for detailed analysis.")
            elif quality >= 0.6:
                print(f"   ✅ GOOD quality. (Score: {quality:.3f})")
                print("   Suitable for most purposes.")
            elif quality >= 0.4:
                print(f"   ⚠️  FAIR quality. (Score: {quality:.3f})")
                print("   May have some noise interference.")
            elif quality > 0:
                print(f"   ❌ LOW quality. (Score: {quality:.3f})")
                print("   Consider re-recording in a quieter environment.")
            else:
                print(f"   ❌ No clear snore detected.")
                print("   Tips:")
                print("      - Ensure audio contains actual snoring")
                print("      - Check recording volume")
                print("      - Try longer recording duration")
            print()
            
            # Additional analysis
            if quality > 0:
                print("🔊 SNORE CHARACTERISTICS:")
                
                # Spectral features
                spec = np.abs(librosa.stft(best_snore))
                freqs = librosa.fft_frequencies(sr=sr)
                snore_band_mask = (freqs >= 50) & (freqs <= 200)
                snore_band_energy = np.sum(spec[snore_band_mask, :])
                total_energy = np.sum(spec) + 1e-8
                snore_ratio = snore_band_energy / total_energy
                
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=best_snore, sr=sr))
                rms_energy = np.mean(librosa.feature.rms(y=best_snore))
                
                print(f"   Spectral Centroid: {spectral_centroid:.2f} Hz")
                print(f"   RMS Energy: {rms_energy:.4f}")
                print(f"   Snore Frequency Ratio: {snore_ratio:.3f}")
                
                # Pitch
                f0, _, _ = librosa.pyin(best_snore, fmin=50, fmax=300, sr=sr)
                valid_f0 = f0[np.isfinite(f0)]
                if len(valid_f0) > 0:
                    pitch_mean = np.mean(valid_f0)
                    pitch_std = np.std(valid_f0)
                    print(f"   Average Pitch: {pitch_mean:.2f} Hz")
                    print(f"   Pitch Variation: {pitch_std:.2f} Hz")
                print()
                
                # Save best snore
                import soundfile as sf
                output_file = "detected_best_snore.wav"
                sf.write(output_file, best_snore, sr)
                print(f"💾 Best snore saved to: {output_file}")
                print()
        
        print("="*60)
        print("  TEST COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_direct.py <audio_file.wav>")
        sys.exit(1)
    
    test_snore_file(sys.argv[1])

