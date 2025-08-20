#!/usr/bin/env python3
"""
Deployment script for AI Music Generation Platform
Optimized for Horizon AI website builder and Hostinger hosting
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "backend/main.py",
        "requirements.txt",
        "index.html",
        "Procfile"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Missing required file: {file}")
            return False
    
    print("✅ All required files present")
    return True

def create_deployment_package():
    """Create deployment-ready package"""
    print("📦 Creating deployment package...")
    
    # Create dist directory
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)
    
    # Copy essential files
    import shutil
    
    files_to_copy = [
        ("backend/", "dist/backend/"),
        ("requirements.txt", "dist/requirements.txt"),
        ("Procfile", "dist/Procfile"),
        ("index.html", "dist/index.html")
    ]
    
    for src, dst in files_to_copy:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
    
    print("✅ Deployment package created in dist/")

def generate_config():
    """Generate configuration for web hosting"""
    config = {
        "name": "ai-music-generation",
        "version": "2.0.0",
        "description": "AI Music Generation Platform",
        "main": "backend/main.py",
        "scripts": {
            "start": "cd backend && python main.py",
            "dev": "cd backend && python main.py"
        },
        "dependencies": {
            "python": "3.11"
        },
        "engines": {
            "python": ">=3.9"
        }
    }
    
    with open("dist/app.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration files generated")

def main():
    """Main deployment function"""
    print("🚀 AI Music Generation - Deployment Prep")
    print("=" * 50)
    
    if not check_requirements():
        sys.exit(1)
    
    create_deployment_package()
    generate_config()
    
    print("\n" + "=" * 50)
    print("✅ Deployment preparation complete!")
    print("\nNext steps for Horizon/Hostinger deployment:")
    print("1. Upload the 'dist/' folder contents to your web hosting")
    print("2. Set environment variables in hosting control panel:")
    print("   - ENV=production")
    print("   - API_KEY=your-secure-api-key")
    print("   - SECRET_KEY=your-secure-secret")
    print("3. Install Python dependencies from requirements.txt")
    print("4. Start the application with: python backend/main.py")
    print("\n🌐 Your AI Music Generation Platform is ready for launch!")

if __name__ == "__main__":
    main()
