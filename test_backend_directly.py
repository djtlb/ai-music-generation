#!/usr/bin/env python3
"""
Direct backend test for the AI music generation system.
"""

import requests
import json
import time
import os
from datetime import datetime

# Replace with your actual API key
API_KEY = "YOUR_BACKEND_API_KEY_HERE"  # Replace this with your actual backend API key
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000") + "/api/v1"

def test_backend_directly():
    """Test the backend API directly."""
    print(f"🎵 Testing backend directly at {datetime.now()}")
    
    # First check if the server is up
    try:
        health_response = requests.get("http://168.231.67.14:8000/healthz")
        if health_response.status_code == 200:
            print("✅ Backend server is up and running!")
        else:
            print(f"❌ Backend health check failed: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {str(e)}")
        return
    
    # Create request payload
    payload = {
        "prompt": "Create a short test melody",
        "user_id": "test-user"
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
    test_backend_directly()
