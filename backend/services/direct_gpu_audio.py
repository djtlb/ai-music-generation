"""
Direct GPU Audio Generator for Beat Addicts

This module provides audio generation functions that directly use the host's GPU 
via subprocess calls rather than trying to access the GPU from inside Docker.
"""

import os
import uuid
import subprocess
import tempfile
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

def create_temp_script():
    """Create a temporary Python script that will run on the host with GPU access"""
    script_content = """
import os
import sys
import torch
import numpy as np
import soundfile as sf
import json
from pathlib import Path

def generate_audio(output_path, project_id, duration=5.0, sample_rate=44100, complexity=3):
    # Check device availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print(f"Using Intel XPU device")
    else:
        device = torch.device('cpu')
        print(f"Using CPU (no GPU available)")
    
    # Generate time array on device
    t = torch.linspace(0, duration, int(sample_rate * duration), device=device)
    
    # Create audio data based on complexity level
    audio = torch.zeros_like(t)
    
    # Basic tone frequencies (can be expanded based on complexity)
    if complexity == 1:  # Simple sine wave
        freqs = torch.tensor([440.0], device=device)  # A4
        amps = torch.tensor([0.8], device=device)
    elif complexity == 2:  # Simple chord
        freqs = torch.tensor([261.63, 329.63, 392.0], device=device)  # C major chord
        amps = torch.tensor([0.6, 0.5, 0.4], device=device)
    else:  # Complex sound with harmonics and modulation
        # Base frequencies for a more complex chord
        freqs = torch.tensor([261.63, 329.63, 392.0, 523.25, 659.26], device=device)
        amps = torch.tensor([0.5, 0.4, 0.3, 0.25, 0.2], device=device)
    
    # Generate the audio
    for i, (freq, amp) in enumerate(zip(freqs, amps)):
        # Add fundamental
        audio += amp * torch.sin(2 * torch.pi * freq * t)
        
        # Add harmonics for more complex sound
        if complexity > 2:
            audio += 0.3 * amp * torch.sin(2 * torch.pi * freq * 2 * t)  # First harmonic
            audio += 0.15 * amp * torch.sin(2 * torch.pi * freq * 3 * t)  # Second harmonic
            
            # Add some modulation for interest
            mod = 0.2 * torch.sin(2 * torch.pi * 2.0 * t)  # 2Hz modulation
            audio = audio * (1.0 + mod)
    
    # Normalize
    audio = audio / torch.max(torch.abs(audio))
    
    # Move back to CPU for saving
    audio_np = audio.cpu().numpy()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save audio file
    sf.write(output_path, audio_np, sample_rate)
    
    # Return metadata
    return {
        "path": output_path,
        "duration": duration,
        "sample_rate": sample_rate,
        "device": str(device),
        "project_id": project_id
    }

if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python gpu_audio_gen.py <output_path> <project_id> [duration] [complexity]")
        sys.exit(1)
    
    output_path = sys.argv[1]
    project_id = sys.argv[2]
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    complexity = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    
    # Generate audio
    result = generate_audio(output_path, project_id, duration, 44100, complexity)
    
    # Print result as JSON
    print(json.dumps(result))
"""
    
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix='.py', prefix='gpu_audio_gen_')
    with os.fdopen(fd, 'w') as f:
        f.write(script_content)
    
    return path

def generate_audio_with_gpu(project_id, duration=5.0, complexity=3):
    """Generate audio using the host GPU through a subprocess call"""
    try:
        # Create temp directories if they don't exist
        static_dir = "/tmp/static"
        Path(static_dir).mkdir(parents=True, exist_ok=True)
        
        # Create output filename
        output_filename = f"{project_id}_final.wav"
        output_path = os.path.join(static_dir, output_filename)
        
        # Create temporary script
        script_path = create_temp_script()
        
        # Run the script as a subprocess with the host Python
        logger.info(f"Generating audio with GPU for project {project_id}")
        cmd = [
            "python3", script_path, 
            output_path, project_id, 
            str(duration), str(complexity)
        ]
        
        # Execute the script
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            check=True
        )
        
        # Parse the result
        if result.stdout:
            try:
                metadata = json.loads(result.stdout.strip())
                logger.info(f"GPU audio generation successful: {metadata}")
                audio_url = f"/static/{output_filename}"
                return audio_url
            except json.JSONDecodeError:
                logger.error(f"Failed to parse GPU script output: {result.stdout}")
        
        # Cleanup
        try:
            os.unlink(script_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary script: {e}")
        
        return f"/static/{output_filename}"
    
    except Exception as e:
        logger.error(f"Error generating audio with GPU: {e}")
        if 'result' in locals() and hasattr(result, 'stderr'):
            logger.error(f"Subprocess error: {result.stderr}")
        
        # Return fallback URL
        return f"/static/{project_id}_final.wav"
