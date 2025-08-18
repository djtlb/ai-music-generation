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

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
npm install --save-dev http-proxy portfinder chalk inquirer dotenv boxen ora
npm install

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd backend || exit
pip install -r requirements.txt
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
