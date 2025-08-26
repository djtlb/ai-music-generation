import os
import torch
import numpy as np
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

def generate_test_audio_with_gpu(project_id, filename_prefix="", duration=5):
    """Generate a test audio file using GPU acceleration if available"""
    try:
        # Create directories
        static_dir = "/tmp/static"
        generated_dir = "/tmp/generated_audio"
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)
        
        # Check for GPU
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
            logger.info("Using AMD GPU for audio generation")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using NVIDIA GPU for audio generation")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for audio generation (no GPU detected)")
        
        # Sample rate and duration
        sample_rate = 44100
        total_samples = int(sample_rate * duration)
        
        # Generate time array
        t = torch.linspace(0, duration, total_samples, device=device)
        
        # Use GPU for synthesis if available - create a more complex sound
        if device.type != "cpu":
            # Create a more complex sound with multiple frequencies using GPU
            freqs = torch.tensor([440.0, 554.37, 659.25], device=device)  # A, C#, E for an A major chord
            amplitudes = torch.tensor([0.5, 0.3, 0.4], device=device)
            
            # Initialize empty audio tensor
            audio_data = torch.zeros(total_samples, device=device)
            
            # Add sine waves with different frequencies
            for freq, amp in zip(freqs, amplitudes):
                audio_data += amp * torch.sin(2 * torch.pi * freq * t)
                
            # Add some modulation
            mod_freq = 2.0  # 2 Hz modulation
            mod = 0.3 * torch.sin(2 * torch.pi * mod_freq * t)
            audio_data = audio_data * (1.0 + mod)
            
            # Normalize
            audio_data = audio_data / torch.max(torch.abs(audio_data))
            
            # Move back to CPU for saving
            audio_data = audio_data.cpu().numpy()
        else:
            # Simple sine wave for CPU
            frequency = 440  # Hz (A note)
            audio_data = 0.5 * np.sin(2 * np.pi * frequency * np.linspace(0, duration, total_samples))
        
        # Create filename
        filename = f"{filename_prefix}{project_id}_final.wav"
        web_path = os.path.join(static_dir, filename)
        
        # Save audio file
        sf.write(web_path, audio_data, sample_rate)
        audio_url = f"/static/{filename}"
        
        logger.info(f"Generated audio file using {device.type}: {web_path}")
        return audio_url
        
    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        return f"/static/{project_id}_final.wav"  # Return a placeholder URL
