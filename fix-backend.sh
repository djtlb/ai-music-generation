#!/bin/bash
# fix-backend.sh - Script to fix backend connectivity issues and ensure real audio generation

set -e

echo "Fixing backend connectivity issues and ensuring real audio generation..."

# 1. Fix firewall issues (if you have UFW installed)
if command -v ufw &> /dev/null; then
    echo "Configuring firewall..."
    sudo ufw allow 8000/tcp
    sudo ufw status
fi

# 2. Create audio directories with proper permissions
echo "Creating audio directories with proper permissions..."
mkdir -p generated_audio
mkdir -p uploads
sudo mkdir -p /var/www/html/audio
sudo chmod -R 777 generated_audio uploads
sudo chmod -R 777 /var/www/html/audio

# 3. Fix backend environment settings
echo "Creating custom .env file with proper settings..."
cat > .env.backend << EOL
# AI Music Generation - Backend Settings
ENV=production
API_KEY=ai-music-gen-horizon-2024
LOG_LEVEL=INFO
PORT=8000
WORKERS=4
CORS_ORIGINS=*
ALLOW_ORIGINS=*
ALLOW_HEADERS=*
ENABLE_REAL_AUDIO=1
GENERATE_FULL_LENGTH=1
MAX_GENERATION_TIME=300
AUDIO_OUTPUT_DIR=/app/generated_audio
PUBLIC_AUDIO_DIR=/app/public_audio
EOL

# 4. Fix CORS headers in frontend
echo "Fixing CORS headers in frontend Supabase functions..."
cd supabase/functions/_shared
cat > cors.ts << EOL
// CORS headers for all Supabase Edge Functions
export const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, x-api-key, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
};
EOL
cd ../../..

# 5. Restart Docker containers with fixed configuration
echo "Restarting Docker containers with fixed configuration..."
docker-compose down
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# 6. Test the backend connectivity
echo "Testing backend connectivity..."
sleep 5
curl -v http://localhost:8000/health || echo "Backend health check failed!"

echo "Fix complete! Backend should now be properly configured for real audio generation."
echo "You can test this by using the Supabase function: generate-song-simple"
