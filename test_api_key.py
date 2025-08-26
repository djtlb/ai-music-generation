import os
import sys
import requests

def test_api_key(api_key):
    """
    Tests if the provided API key is valid by making a request to a protected endpoint.
    """
    url = os.environ.get("BACKEND_URL", "http://localhost:8000") + "/api/v1/system/stats"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print("API key is valid.")
            print("Response from backend:")
            print(response.json())
            return True
        else:
            print(f"API key is invalid or backend is unreachable. Status code: {response.status_code}")
            print("Response text:")
            print(response.text)
            return False
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while connecting to the backend: {e}")
        return False

if __name__ == "__main__":
    # Test default key from .env
    default_key = "your-super-secret-key-12345"
    
    # Test the backend's default key from config.py
    backend_default = "ai-music-gen-2024-secure-key"
    
    # Test the key from app/api/__init__.py
    api_init_key = "ai-music-gen-horizon-2024"
    
    # Test with command line argument if provided
    if len(sys.argv) > 1:
        custom_key = sys.argv[1]
        print("\n=== Testing custom key from command line ===")
        test_api_key(custom_key)
    
    # Test with the three default keys we found
    print("\n=== Testing .env default key ===")
    test_api_key(default_key)
    
    print("\n=== Testing backend/config.py default key ===")
    test_api_key(backend_default)
    
    print("\n=== Testing api/__init__.py default key ===")
    test_api_key(api_init_key)
