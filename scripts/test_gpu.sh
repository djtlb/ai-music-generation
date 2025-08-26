#!/bin/bash
# Script to test AMD GPU functionality with the current setup

echo "Testing AMD GPU with Beat Addicts..."

# Test GPU detection from host system
echo "Host system GPU detection:"
rocm-smi

# Make sure Docker is running with GPU access
echo -e "\nCurrent container status:"
docker-compose -f docker-compose.rocm.yml ps

# Create a test script for the container
echo -e "\nCreating GPU test script for the container..."
cat > gpu_test.py << 'EOF'
import os
import sys
import subprocess
import platform

# System information
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")

# Check if ROCm is installed
try:
    rocm_version = subprocess.check_output("hipconfig --version", shell=True).decode().strip()
    print(f"ROCm detected: {rocm_version}")
except:
    print("ROCm not detected via hipconfig")

# Check environment variables
print("\nEnvironment variables:")
for var in ["HSA_OVERRIDE_GFX_VERSION", "ROCM_PATH", "HIP_PATH"]:
    print(f"{var}: {os.environ.get(var, 'Not set')}")

# Try to import PyTorch and check GPU
try:
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Check ROCm support
    has_rocm = getattr(torch._C, "_has_rocm", False)
    print(f"PyTorch compiled with ROCm: {has_rocm}")
    
    # Check CUDA support (should be False for ROCm)
    has_cuda = getattr(torch._C, "_has_cuda", False)
    print(f"PyTorch compiled with CUDA: {has_cuda}")
    
    # Check for XPU (Intel GPU) support
    has_xpu = hasattr(torch, "xpu")
    xpu_available = has_xpu and torch.xpu.is_available()
    print(f"PyTorch has XPU module: {has_xpu}")
    print(f"XPU available: {xpu_available}")
    
    # Check available devices
    if has_rocm:
        try:
            device_count = torch.cuda.device_count()  # For ROCm, we use the CUDA API
            print(f"ROCm device count: {device_count}")
            if device_count > 0:
                for i in range(device_count):
                    print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        except Exception as e:
            print(f"Error checking ROCm devices: {e}")
    
    # Try to run a simple tensor operation on GPU
    try:
        if has_rocm and torch.cuda.is_available():
            print("\nRunning tensor operation on ROCm GPU...")
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = x @ y
            print(f"GPU tensor operation successful. Result shape: {z.shape}")
        elif xpu_available:
            print("\nRunning tensor operation on XPU...")
            x = torch.randn(10, 10).xpu()
            y = torch.randn(10, 10).xpu()
            z = x @ y
            print(f"XPU tensor operation successful. Result shape: {z.shape}")
        else:
            print("\nNo GPU available for tensor operations")
    except Exception as e:
        print(f"Error running tensor operation: {e}")
    
except ImportError as e:
    print(f"\nError importing PyTorch: {e}")
except Exception as e:
    print(f"\nUnexpected error: {e}")
EOF

echo "Copying test script to container..."
docker cp gpu_test.py aimusic_api:/tmp/

echo -e "\nRunning GPU test in container:"
docker-compose -f docker-compose.rocm.yml exec api python /tmp/gpu_test.py

echo -e "\nChecking if audio generation can benefit from GPU:"
docker-compose -f docker-compose.rocm.yml exec api bash -c "python -c \"import numpy as np; import time; import os; start=time.time(); audio = np.sin(2*np.pi*440*np.linspace(0, 5, 44100*5)); elapsed=time.time()-start; print(f'Audio generated in {elapsed:.4f} seconds (CPU)');\""

echo -e "\nTest complete!"
