"""
Collab Lab Integration Module

This module connects the Collab Lab frontend with the song generation pipeline.
It provides WebSocket-based handlers that trigger the song generation process
and return real-time updates to all collaborators.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List
import uuid
from fastapi import WebSocket
from backend.api.routes.collab_router import broadcast_message

# Set up logging
logger = logging.getLogger(__name__)

# In-memory storage for active generation tasks
active_generation_tasks: Dict[str, asyncio.Task] = {}

async def handle_generation_request(session_id: str, message: Dict[str, Any], websockets: List[WebSocket]):
    """
    Handle a music generation request from the Collab Lab.
    This function will:
    1. Extract parameters from the WebSocket message
    2. Notify all clients that generation has started
    3. Run the song generation pipeline
    4. Update all clients with the generation results
    """
    try:
        # Extract parameters from the message
        content = message.get("content", {})
        lyrics = content.get("lyrics", "")
        genre = content.get("genre", "")
        # style_tags can be used for additional customization in the future
        # sender_id can be used for attribution in the future
        
        # Determine the style based on genre or first style tag
        style = genre.lower() if genre else "pop"
        if style not in ["rock_punk", "rnb_ballad", "country_pop"]:
            # Map to closest available style
            if "rock" in style or "punk" in style or "metal" in style:
                style = "rock_punk"
            elif "rnb" in style or "soul" in style or "ballad" in style:
                style = "rnb_ballad"
            elif "country" in style or "pop" in style or "folk" in style:
                style = "country_pop"
            else:
                # Default to country_pop as a safe choice
                style = "country_pop"
        
        # Generate a unique output directory for this collaboration
        output_dir = os.path.join("exports", f"collab_{session_id}_{uuid.uuid4().hex[:8]}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Notify all clients that generation has started
        await broadcast_message(
            session_id=session_id,
            message={
                "type": "generation_started",
                "content": {
                    "style": style,
                    "genre": genre,
                    "message": "Starting music generation...",
                },
                "sender_id": "system",
                "sender_name": "Beat Addicts AI",
                "timestamp": None  # Will be filled by broadcast_message
            }
        )
        
        # Simulate progress updates (will be replaced with actual progress from generator)
        for progress in [10, 25, 50, 75, 90]:
            await asyncio.sleep(2)  # Simulate processing time
            await broadcast_message(
                session_id=session_id,
                message={
                    "type": "generation_progress",
                    "content": {
                        "progress": progress,
                        "message": f"Generating {style} track... {progress}% complete"
                    },
                    "sender_id": "system",
                    "sender_name": "Beat Addicts AI",
                    "timestamp": None
                }
            )
        
        # Simulate the generation process (replace with actual generation)
        try:
            # In a real implementation, this would call the actual music generation function
            # For now, we'll simulate it with a delay
            await asyncio.sleep(5)
            
            # For demo purposes, assume generation succeeded and create a dummy output file
            # In production, we would use the actual output from the generation process
            
            # Just for testing - create a dummy file to represent the generated audio
            demo_file_path = os.path.join(output_dir, "generated_track.mp3")
            with open(demo_file_path, "wb") as f:
                # Write an empty file for demo purposes
                f.write(b"")
                
            final_audio_path = demo_file_path
            success = True
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            final_audio_path = None
            success = False
        
        if success and final_audio_path:
            # Create a web-accessible URL for the audio file
            # This assumes you've set up static file serving in your FastAPI app
            base_url = "/audio"  # Update to match your static file route
            relative_path = os.path.relpath(final_audio_path, "exports")
            output_url = f"{base_url}/{relative_path}"
            
            # Notify all clients that generation is complete
            await broadcast_message(
                session_id=session_id,
                message={
                    "type": "generation_complete",
                    "content": {
                        "output_url": output_url,
                        "message": "Music generation complete!",
                        "metadata": {
                            "style": style,
                            "genre": genre,
                            "lyrics": lyrics
                        }
                    },
                    "sender_id": "system",
                    "sender_name": "Beat Addicts AI",
                    "timestamp": None  # Will be filled by broadcast_message
                }
            )
            
            return {
                "success": True,
                "output_url": output_url,
                "output_dir": output_dir
            }
        else:
            # Notify all clients that generation failed
            await broadcast_message(
                session_id=session_id,
                message={
                    "type": "generation_error",
                    "content": "Music generation failed. Please try again with different parameters.",
                    "sender_id": "system",
                    "sender_name": "Beat Addicts AI",
                    "timestamp": None  # Will be filled by broadcast_message
                }
            )
            return {
                "success": False,
                "error": "Generation failed"
            }
            
    except Exception as e:
        logger.error(f"Error in generation handler: {str(e)}")
        # Notify all clients about the error
        await broadcast_message(
            session_id=session_id,
            message={
                "type": "generation_error",
                "content": f"An unexpected error occurred: {str(e)}",
                "sender_id": "system",
                "sender_name": "Beat Addicts AI",
                "timestamp": None  # Will be filled by broadcast_message
            }
        )
        return {
            "success": False,
            "error": str(e)
        }


async def start_generation_task(session_id: str, message: Dict[str, Any], websockets: List[WebSocket]):
    """
    Start a song generation task and track it.
    This allows multiple generation tasks to run concurrently for different collaboration sessions.
    """
    # Cancel any existing task for this session
    if session_id in active_generation_tasks:
        active_generation_tasks[session_id].cancel()
        try:
            await active_generation_tasks[session_id]
        except asyncio.CancelledError:
            pass
    
    # Create a new task
    task = asyncio.create_task(handle_generation_request(session_id, message, websockets))
    active_generation_tasks[session_id] = task
    
    # Set up callback to clean up when task completes
    def on_task_done(future):
        if session_id in active_generation_tasks:
            del active_generation_tasks[session_id]
    
    task.add_done_callback(on_task_done)
    return task
