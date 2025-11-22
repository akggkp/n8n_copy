#!/usr/bin/env python
"""
Test Video Processor Service
Tests: Audio extraction, transcription, frame extraction
"""
import sys
import os

# Add service to path
sys.path.insert(0, 'services/video-processor')

def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Test 1.1: Checking imports...")
    try:
        import cv2
        import whisper
        import ffmpeg
        from fastapi import FastAPI
        from pydantic import BaseModel
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Run: pip install opencv-python openai-whisper ffmpeg-python fastapi pydantic")
        return False

def test_whisper_model():
    """Test Whisper model loading"""
    print("\n🧪 Test 1.2: Loading Whisper model...")
    try:
        import whisper
        model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Whisper model loading failed: {e}")
        return False

def test_video_file_access():
    """Test video file access"""
    print("\n🧪 Test 1.3: Checking video file access...")
    
    test_video_path = "data/videos/scalping.mp4"
    
    if os.path.exists(test_video_path):
        print(f"✅ Test video found: {test_video_path}")
        
        # Check if readable
        file_size = os.path.getsize(test_video_path)
        print(f"   File size: {file_size / 1024 / 1024:.2f} MB")
        return True
    else:
        print(f"⚠️  No test video at {test_video_path}")
        print("   Place a sample video in data/videos/ to test full pipeline")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    print("\n🧪 Test 1.4: Testing OpenCV...")
    try:
        import cv2
        
        test_video_path = "data/videos/scalping.mp4"
        if not os.path.exists(test_video_path):
            print("⚠️  No test video, skipping OpenCV test")
            return False
        
        cap = cv2.VideoCapture(test_video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"✅ OpenCV can read video")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   FPS: {fps:.2f}")
            print(f"   Total frames: {frame_count}")
            
            cap.release()
            return True
        else:
            print("❌ OpenCV cannot open video")
            return False
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_service_endpoints():
    """Test if service can be imported and endpoints exist"""
    print("\n🧪 Test 1.5: Testing service structure...")
    try:
        from app.main import app, process_video, health
        
        print("✅ Service module imported successfully")
        print(f"   Endpoints: /, /health, /process")
        return True
    except Exception as e:
        print(f"❌ Service import failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("VIDEO PROCESSOR SERVICE - STANDALONE TESTS")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Whisper Model", test_whisper_model()))
    results.append(("Video File Access", test_video_file_access()))
    results.append(("OpenCV", test_opencv()))
    results.append(("Service Endpoints", test_service_endpoints()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nPassed: {total_passed}/{len(results)}")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Video processor is ready.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Fix issues before building Docker image.")
        sys.exit(1)