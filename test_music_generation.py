#!/usr/bin/env python3
"""
Test script for the AI music generation system.
This script tests the backend directly using the API.
"""

import requests
import json
import time
import os
from datetime import datetime

# Replace with your actual API key and backend URL
API_KEY = "your-backend-api-key"  # Replace with your actual API key
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000") + "/api/v1"

def test_music_generation():
    """Test the music generation endpoint directly."""
    print(f"🎵 Testing music generation at {datetime.now()}")
    
    # Create the request payload
    payload = {
        "prompt": "Create an upbeat pop song with catchy melody and a strong chorus",
        "user_id": "test-user"  # User ID for testing
    }
    
    # Set up headers with API key
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    try:
        # Make the request to start generation
        print("Sending request to generate music...")
        response = requests.post(
            f"{BACKEND_URL}/generate/full-song",
            headers=headers,
            json=payload
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return
            
        # Parse the response
        data = response.json()
        print(f"✅ Request successful! Response: {json.dumps(data, indent=2)}")
        
        # If we have a project ID, check its status
        if "project_id" in data:
            project_id = data["project_id"]
            print(f"📋 Project ID: {project_id}")
            
            # Poll for status a few times
            print("Polling for status...")
            for i in range(3):  # Just poll 3 times for the test
                time.sleep(5)  # Wait 5 seconds between polls
                
                status_response = requests.get(
                    f"{BACKEND_URL}/project/{project_id}/status",
                    headers=headers
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"Status poll {i+1}: {json.dumps(status_data, indent=2)}")
                    
                    # If the status is completed, print the audio URL
                    if status_data.get("status") == "completed":
                        print("🎉 Music generation completed!")
                        audio_url = status_data.get("audio_url")
                        if audio_url:
                            print(f"🎧 Audio URL: {audio_url}")
                        break
                else:
                    print(f"❌ Error checking status: {status_response.status_code}")
                    print(status_response.text)
        
        print("Test completed!")
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_music_generation()
