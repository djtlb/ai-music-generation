#!/bin/bash

# Beat Addicts Setup Script
# This script installs all required dependencies for both frontend and backend

echo "🎵 Beat Addicts - Setup Script 🎵"
echo "================================="
echo "This script will install all required dependencies for the Beat Addicts application."
echo ""

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install Node.js and npm first."
    exit 1
fi

# Check if python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create required directories
echo "� Creating required directories..."
mkdir -p exports generated_audio static src/lib backend

# Update package.json with newer lucide-react version to avoid conflicts
echo "🔧 Updating package.json dependencies..."
sed -i 's/"lucide-react": "\^0.344.0"/"lucide-react": "\^0.484.0"/g' package.json

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
npm install --save-dev http-proxy portfinder chalk inquirer dotenv boxen ora --legacy-peer-deps
npm install --legacy-peer-deps

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd backend || exit
# Use a virtual environment for Python dependencies to avoid conflicts
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
# Activate virtual environment
source venv/bin/activate
echo "Installing core dependencies first..."
if [ ! -f "core-requirements.txt" ]; then
    echo "Creating core-requirements.txt..."
    cat > core-requirements.txt << EOF
# Core FastAPI Backend Requirements
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0
python-multipart>=0.0.6
websockets>=12.0
EOF
fi
pip install -r core-requirements.txt
echo "Installing full dependencies (this might take a while)..."
pip install -r requirements.txt || {
    echo "⚠️ Some dependencies failed to install. This is okay for now."
    echo "You can try installing them later with: cd backend && source venv/bin/activate && pip install -r requirements.txt"
}
cd ..

# Make connect.js executable
chmod +x connect.js

echo ""
echo "✅ Setup complete!"
echo ""
echo "You can now start the application with:"
echo "  node connect.js"
echo ""
echo "Or, once the connect script has updated package.json:"
echo "  npm run start"
echo ""
