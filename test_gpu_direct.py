#!/usr/bin/env python3
"""
Direct GPU Test for Beat Addicts

This script tests GPU detection and audio generation outside of Docker.
"""
import os
import sys
import time
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

def test_gpu_detection():
    """Test if PyTorch can detect any GPUs"""
    print("=== GPU Detection Test ===")
    print(f"PyTorch version: {torch.__version__}")
    
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    
    has_rocm = hasattr(torch._C, "_has_rocm") and torch._C._has_rocm
    print(f"ROCm support: {has_rocm}")
    
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    print(f"XPU available: {has_xpu}")
    
    device = None
    if has_cuda:
        device = torch.device("cuda")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    elif has_xpu:
        device = torch.device("xpu")
        print("Using Intel XPU")
    else:
        device = torch.device("cpu")
        print("No GPU detected, using CPU")
        
    return device

def generate_test_audio(device, output_path="test_gpu_audio.wav"):
    """Generate a test audio file using the specified device"""
    print(f"\n=== Generating test audio using {device.type} ===")
    
    # Parameters
    sample_rate = 44100
    duration = 5.0  # seconds
    
    # Create time array
    t = torch.linspace(0, duration, int(sample_rate * duration), device=device)
    
    # Create audio data - a simple chord with modulation
    freqs = torch.tensor([261.63, 329.63, 392.0], device=device)  # C major chord
    amps = torch.tensor([0.6, 0.5, 0.4], device=device)
    
    start_time = time.time()
    
    # Generate audio
    audio = torch.zeros_like(t)
    for freq, amp in zip(freqs, amps):
        audio += amp * torch.sin(2 * torch.pi * freq * t)
    
    # Add modulation
    mod = 0.2 * torch.sin(2 * torch.pi * 2.0 * t)
    audio = audio * (1.0 + mod)
    
    # Normalize
    audio = audio / torch.max(torch.abs(audio))
    
    # Move back to CPU for saving
    audio_np = audio.cpu().numpy()
    
    elapsed = time.time() - start_time
    print(f"Audio generation completed in {elapsed:.4f} seconds")
    
    # Create output directory
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_path
    
    # Save audio file
    sf.write(output_path, audio_np, sample_rate)
    print(f"Audio saved to {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    print("Beat Addicts Direct GPU Test")
    print("==========================\n")
    
    # Test GPU detection
    device = test_gpu_detection()
    
    # Generate test audio
    audio_path = generate_test_audio(device)
    
    print("\nTest complete!")
    print(f"Audio file generated: {audio_path}")
    print("If you can see GPU information above and the audio file was generated successfully,")
    print("your GPU is working correctly with Beat Addicts.")
