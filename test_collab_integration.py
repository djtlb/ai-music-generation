#!/usr/bin/env python
"""
Test script for the Beat Addicts Collab Lab integration.
This script simulates a WebSocket client connecting to the Collab Lab,
sending messages, and receiving responses.
"""

import asyncio
import json
import logging
import websockets
import argparse
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("collab-test")

async def test_collab_session(session_id=None, user_id=None, server_url=None):
    """Test the Collab Lab integration by simulating a WebSocket client."""
    
    # Use defaults if not provided
    session_id = session_id or str(uuid.uuid4())[:8]
    user_id = user_id or f"test-user-{str(uuid.uuid4())[:6]}"
    server_url = server_url or "ws://localhost:8000/api/v1/ws/collab"
    
    # Construct WebSocket URL
    ws_url = f"{server_url}/{session_id}?user_id={user_id}"
    logger.info(f"Connecting to: {ws_url}")
    
    try:
        # Connect to WebSocket
        async with websockets.connect(ws_url) as websocket:
            logger.info(f"Connected to session {session_id} as user {user_id}")
            
            # Receive welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            logger.info(f"Received welcome message: {welcome_data['type']}")
            
            # Update lyrics
            await websocket.send(json.dumps({
                "type": "lyrics_update",
                "content": "This is a test of the collab system\nLet's see if we can generate some music\nWith our amazing Beat Addicts AI",
                "sender_id": user_id
            }))
            logger.info("Sent lyrics update")
            
            # Wait for acknowledgment
            response = await websocket.recv()
            logger.info(f"Received response: {json.loads(response)['type']}")
            
            # Update genre
            await websocket.send(json.dumps({
                "type": "genre_update",
                "content": "rock_punk",
                "sender_id": user_id
            }))
            logger.info("Sent genre update")
            
            # Wait for acknowledgment
            response = await websocket.recv()
            logger.info(f"Received response: {json.loads(response)['type']}")
            
            # Update style tags
            await websocket.send(json.dumps({
                "type": "style_update",
                "content": ["upbeat", "electronic"],
                "sender_id": user_id
            }))
            logger.info("Sent style update")
            
            # Wait for acknowledgment
            response = await websocket.recv()
            logger.info(f"Received response: {json.loads(response)['type']}")
            
            # Generate music
            await websocket.send(json.dumps({
                "type": "generate",
                "content": {
                    "lyrics": "This is a test of the collab system\nLet's see if we can generate some music\nWith our amazing Beat Addicts AI",
                    "genre": "rock_punk",
                    "style_tags": ["upbeat", "electronic"]
                },
                "sender_id": user_id
            }))
            logger.info("Sent generate request")
            
            # Listen for messages for a while
            try:
                for _ in range(10):  # Listen for 10 messages or 30 seconds, whichever comes first
                    response = await asyncio.wait_for(websocket.recv(), timeout=30)
                    response_data = json.loads(response)
                    logger.info(f"Received: {response_data['type']}")
                    
                    if response_data['type'] == 'generation_complete':
                        logger.info("Generation completed successfully!")
                        output_url = response_data.get('content', {}).get('output_url')
                        if output_url:
                            logger.info(f"Output URL: {output_url}")
                        break
                        
                    elif response_data['type'] == 'generation_error':
                        logger.error(f"Generation failed: {response_data.get('content')}")
                        break
            
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for generation to complete")
            
            # Send a chat message
            await websocket.send(json.dumps({
                "type": "chat",
                "content": "Test complete!",
                "sender_id": user_id,
                "sender_name": "Test User"
            }))
            logger.info("Sent chat message")
            
            # Wait for a short while to receive any final messages
            await asyncio.sleep(2)
            
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        return False
        
    logger.info("Test completed successfully")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Collab Lab integration")
    parser.add_argument("--session-id", help="Session ID to join or create")
    parser.add_argument("--user-id", help="User ID to use")
    parser.add_argument("--server", help="WebSocket server URL", default="ws://localhost:8000/api/v1/ws/collab")
    
    args = parser.parse_args()
    
    asyncio.run(test_collab_session(
        session_id=args.session_id,
        user_id=args.user_id,
        server_url=args.server
    ))
