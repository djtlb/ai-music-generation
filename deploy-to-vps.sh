#!/bin/bash
# deploy-to-vps.sh - Deploy all fixes to the VPS

set -e

# Check if REMOTE_HOST is set
if [ -z "$REMOTE_HOST" ]; then
    echo "Error: REMOTE_HOST environment variable not set."
    echo "Usage: REMOTE_HOST=user@your-server-ip ./deploy-to-vps.sh"
    exit 1
fi

echo "Deploying fixes to VPS at $REMOTE_HOST..."

# 1. Copy the fix-backend.sh script to the VPS
echo "Copying fix script to VPS..."
scp fix-backend.sh $REMOTE_HOST:/opt/ai-music-generation/

# 2. Copy docker-compose.override.yml to the VPS
echo "Copying Docker override configuration..."
scp docker-compose.override.yml $REMOTE_HOST:/opt/ai-music-generation/

# 3. Run the fix script on the VPS
echo "Running fix script on VPS..."
ssh $REMOTE_HOST "cd /opt/ai-music-generation && chmod +x fix-backend.sh && sudo ./fix-backend.sh"

# 4. Restart Docker on the VPS
echo "Restarting Docker on VPS..."
ssh $REMOTE_HOST "cd /opt/ai-music-generation && sudo docker-compose down && sudo docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d"

# 5. Verify the backend is running
echo "Verifying backend is running..."
ssh $REMOTE_HOST "curl -v http://localhost:8000/health || echo 'Health check failed!'"

echo "Deployment complete! All fixes have been applied to the VPS."
echo "Your AI music generation system should now be properly configured for creating real audio."
