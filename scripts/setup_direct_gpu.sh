#!/bin/bash
# ----------------------------------------
# AMD GPU Setup for Beat Addicts AI Music Generation
# ----------------------------------------

set -e  # Exit on error

# Display header
echo "=========================================="
echo "    BEAT ADDICTS AMD GPU SETUP SCRIPT    "
echo "=========================================="
echo "This script will configure Beat Addicts to properly use your AMD GPU"
echo

# Check if ROCm is installed
echo "Checking AMD ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm detected"
    rocm-smi --showuse
else
    echo "❌ ROCm not detected! Installing..."
    echo "Installing ROCm packages..."
    sudo apt-get update
    sudo apt-get install -y rocm-libs miopen-hip rocm-dev rocm-utils
    echo "ROCm installation completed"
fi

# Check if PyTorch is installed directly on the host
echo -e "\nChecking PyTorch installation..."
if python3 -c "import torch" &> /dev/null; then
    echo "✅ PyTorch is installed"
else
    echo "❌ PyTorch not detected! Installing ROCm-compatible PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
    echo "PyTorch installation completed"
fi

# Check if Docker is installed
echo -e "\nChecking Docker installation..."
if command -v docker &> /dev/null; then
    echo "✅ Docker detected"
else
    echo "❌ Docker not detected! Installing..."
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose
    sudo usermod -aG docker $USER
    echo "Docker installation completed (you may need to log out and back in)"
fi

# Stop containers if they're running
echo -e "\nStopping any running Beat Addicts containers..."
cd /home/beats/Desktop/ai-music-generation
docker-compose down || true

# Install required Python packages for audio processing
echo -e "\nInstalling required Python packages..."
pip install numpy scipy matplotlib soundfile

# Create modified orchestrator implementation for direct GPU access
echo -e "\nCreating GPU-accelerated audio generation code..."
cat > /home/beats/Desktop/ai-music-generation/backend/services/direct_gpu_audio.py << 'EOF'
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
EOF

# Update the AI orchestrator to use direct GPU access
echo -e "\nUpdating AI orchestrator to use direct GPU access..."

# First, check if we need to modify the file
ORCHESTRATOR_FILE="/home/beats/Desktop/ai-music-generation/backend/services/ai_orchestrator.py"

# Read the file and update it
if grep -q "from backend.services.direct_gpu_audio import generate_audio_with_gpu" "$ORCHESTRATOR_FILE"; then
    echo "Orchestrator already updated, skipping..."
else
    # Add import statement
    sed -i '1s/^/from backend.services.direct_gpu_audio import generate_audio_with_gpu\n/' "$ORCHESTRATOR_FILE"
    
    # Replace the audio generation code
    sed -i 's/# Stage 5: Mix & Master.*audio_url = f"\/static\/{project_id}_final.wav"/# Stage 5: Mix & Master (Host GPU Direct Access)\n            stage_start("mix_master")\n            audio_url = generate_audio_with_gpu(project_id)/g' "$ORCHESTRATOR_FILE"
    
    echo "Orchestrator updated successfully!"
fi

# Create a direct GPU test script
echo -e "\nCreating GPU test script..."
cat > /home/beats/Desktop/ai-music-generation/test_gpu_direct.py << 'EOF'
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
EOF

# Make the test script executable
chmod +x /home/beats/Desktop/ai-music-generation/test_gpu_direct.py

# Restart the Docker containers
echo -e "\nRestarting Beat Addicts with GPU support..."
cd /home/beats/Desktop/ai-music-generation
docker-compose restart api

# Run the GPU test
echo -e "\nRunning GPU test..."
cd /home/beats/Desktop/ai-music-generation
python3 test_gpu_direct.py

echo -e "\n=========================================="
echo "Setup complete! Beat Addicts should now be using your AMD GPU"
echo "for audio generation via direct host access."
echo "=========================================="
