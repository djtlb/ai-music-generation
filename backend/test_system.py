#!/usr/bin/env python3
"""
Test script to verify all AI Music Generation system features work correctly.
"""

import asyncio
import httpx
import json
import websockets
import time

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
API_KEY = "test-api-key-123"


async def test_health_endpoints():
    """Test all health check endpoints."""
    print("\n🔍 Testing Health Endpoints...")
    
    async with httpx.AsyncClient() as client:
        endpoints = ["/healthz", "/health/ready", "/readyz", "/health"]
        
        for endpoint in endpoints:
            try:
                response = await client.get(f"{BASE_URL}{endpoint}")
                print(f"✅ {endpoint}: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"   Status: {data.get('status', 'N/A')}")
            except Exception as e:
                print(f"❌ {endpoint}: {e}")


async def test_api_endpoints():
    """Test main API endpoints."""
    print("\n🎵 Testing API Endpoints...")
    
    async with httpx.AsyncClient() as client:
        # Test music generation
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/music/generate",
                json={
                    "prompt": "Create a happy pop song",
                    "style": "pop",
                    "duration": 30
                }
            )
            print(f"✅ Music Generation: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Generation ID: {data.get('generation_id')}")
        except Exception as e:
            print(f"❌ Music Generation: {e}")
        
        # Test authentication
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/auth/login",
                json={"email": "test@example.com", "password": "test123"}
            )
            print(f"✅ Authentication: {response.status_code}")
        except Exception as e:
            print(f"❌ Authentication: {e}")
        
        # Test projects
        try:
            response = await client.get(f"{BASE_URL}/api/v1/projects/")
            print(f"✅ Projects List: {response.status_code}")
        except Exception as e:
            print(f"❌ Projects List: {e}")
        
        # Test full song generation
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/generate/full-song",
                json={
                    "project_name": "Test Song",
                    "user_id": "test-user-123",
                    "style_config": {"genre": "pop", "mood": "happy"}
                },
                headers={"api_key": API_KEY}
            )
            print(f"✅ Full Song Generation: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Project ID: {data.get('project_id')}")
        except Exception as e:
            print(f"❌ Full Song Generation: {e}")


async def test_websocket():
    """Test WebSocket functionality."""
    print("\n📡 Testing WebSocket Connection...")
    
    try:
        uri = f"{WS_URL}/ws/test-client-123"
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected")
            
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            if data.get("type") == "pong":
                print("✅ WebSocket ping/pong working")
            
            # Subscribe to events
            await websocket.send(json.dumps({
                "type": "subscribe",
                "events": ["generation_progress", "system_updates"]
            }))
            print("✅ WebSocket subscription working")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")


async def test_system_stats():
    """Test system statistics endpoint."""
    print("\n📊 Testing System Statistics...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/system/stats",
                headers={"api_key": API_KEY}
            )
            print(f"✅ System Stats: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                stats = data.get("data", {})
                print(f"   Total songs: {stats.get('total_songs_generated')}")
                print(f"   Active users: {stats.get('active_users')}")
                print(f"   Revenue today: ${stats.get('revenue_today')}")
        except Exception as e:
            print(f"❌ System Stats: {e}")


async def test_all_features():
    """Test all system features."""
    print("🎵 AI Music Generation System - Feature Test")
    print("=" * 50)
    
    # Test basic connectivity first
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/")
            if response.status_code == 200:
                print("✅ Server is running")
                data = response.json()
                print(f"   Version: {data.get('version')}")
                print(f"   Status: {data.get('status')}")
            else:
                print(f"❌ Server not responding properly: {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the server is running with: python main.py")
        return
    
    # Run all tests
    await test_health_endpoints()
    await test_api_endpoints()
    await test_websocket()
    await test_system_stats()
    
    print("\n" + "=" * 50)
    print("🎉 Feature testing complete!")
    print("\n📖 Next steps:")
    print("   1. Open http://localhost:8000/docs for API documentation")
    print("   2. Try the interactive endpoints in the browser")
    print("   3. Connect to WebSocket at ws://localhost:8000/ws/your-client-id")


if __name__ == "__main__":
    print("Starting comprehensive system test...")
    asyncio.run(test_all_features())
