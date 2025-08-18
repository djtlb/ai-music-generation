"""
Collab Lab - Real-time collaboration router for Beat Addicts
Enables real-time music collaboration sessions via WebSockets
"""

import json
import uuid
import logging
from typing import Dict, List, Set, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Models
class CollabSession(BaseModel):
    id: str
    name: str
    created_at: datetime
    lyrics: str = ""
    genre: str = ""
    style_tags: List[str] = []
    tempo: int = 120
    creator_id: str
    participants: Set[str] = set()
    is_public: bool = True
    
    class Config:
        arbitrary_types_allowed = True

class CollabMessage(BaseModel):
    type: str  # "lyrics_update", "genre_update", "style_update", "chat", "generate", etc.
    content: Any
    sender_id: str
    timestamp: datetime = None

# In-memory storage (replace with Redis in production)
active_sessions: Dict[str, CollabSession] = {}
connected_clients: Dict[str, Set[WebSocket]] = {}

# API Routes
@router.post("/sessions", response_model=CollabSession)
async def create_session(name: str, creator_id: str, is_public: bool = True):
    """Create a new collaboration session"""
    session_id = str(uuid.uuid4())[:8]  # Short, readable ID
    
    session = CollabSession(
        id=session_id,
        name=name,
        created_at=datetime.now(),
        creator_id=creator_id,
        is_public=is_public,
        participants={creator_id}
    )
    
    active_sessions[session_id] = session
    connected_clients[session_id] = set()
    
    logger.info(f"Created collab session: {session_id}")
    return session

@router.get("/sessions", response_model=List[CollabSession])
async def list_sessions(limit: int = 10, creator_id: Optional[str] = None):
    """List active collaboration sessions"""
    sessions = list(active_sessions.values())
    
    # Filter by creator if specified
    if creator_id:
        sessions = [s for s in sessions if s.creator_id == creator_id]
    
    # Only return public sessions unless queried by creator
    if not creator_id:
        sessions = [s for s in sessions if s.is_public]
    
    # Sort by most recent first
    sessions.sort(key=lambda s: s.created_at, reverse=True)
    
    return sessions[:limit]

@router.get("/sessions/{session_id}", response_model=CollabSession)
async def get_session(session_id: str):
    """Get details of a specific collaboration session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return active_sessions[session_id]

# WebSocket Connection
@router.websocket("/ws/collab/{session_id}")
async def collab_websocket(websocket: WebSocket, session_id: str, user_id: str = Query(...)):
    """WebSocket endpoint for real-time collaboration"""
    await websocket.accept()
    
    # Validate session exists
    if session_id not in active_sessions:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    # Add user to session participants
    session = active_sessions[session_id]
    session.participants.add(user_id)
    
    # Add client to connected clients
    if session_id not in connected_clients:
        connected_clients[session_id] = set()
    connected_clients[session_id].add(websocket)
    
    # Send welcome message with current session state
    await websocket.send_json({
        "type": "session_state",
        "content": {
            "lyrics": session.lyrics,
            "genre": session.genre,
            "style_tags": session.style_tags,
            "participants": list(session.participants),
        },
        "timestamp": datetime.now().isoformat(),
    })
    
    # Broadcast user joined message
    await broadcast_message(
        session_id=session_id, 
        message={
            "type": "user_joined",
            "content": f"{user_id} joined the session",
            "sender_id": "system",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
        },
        exclude=websocket
    )
    
    try:
        # Main message loop
        while True:
            # Receive and parse message
            data = await websocket.receive_text()
            message = json.loads(data)
            message["timestamp"] = datetime.now().isoformat()
            
            # Process message based on type
            if message["type"] == "lyrics_update":
                session.lyrics = message["content"]
            elif message["type"] == "genre_update":
                session.genre = message["content"]
            elif message["type"] == "style_update":
                session.style_tags = message["content"]
            elif message["type"] == "generate":
                # Trigger music generation
                from backend.services.collab_integration import start_generation_task
                await start_generation_task(session_id, message, list(connected_clients[session_id]))
            
            # Broadcast message to all clients in session
            await broadcast_message(session_id, message)
            
    except WebSocketDisconnect:
        # Clean up when client disconnects
        if session_id in connected_clients:
            connected_clients[session_id].discard(websocket)
        
        # Remove session if no clients left
        if session_id in connected_clients and len(connected_clients[session_id]) == 0:
            del connected_clients[session_id]
            del active_sessions[session_id]
            logger.info(f"Removed empty session: {session_id}")
        
        # Broadcast user left message
        else:
            await broadcast_message(
                session_id=session_id,
                message={
                    "type": "user_left",
                    "content": f"{user_id} left the session",
                    "sender_id": "system",
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                }
            )
            
            # Remove user from participants
            if session_id in active_sessions:
                active_sessions[session_id].participants.discard(user_id)

async def broadcast_message(session_id: str, message: dict, exclude: Optional[WebSocket] = None):
    """Broadcast a message to all clients in a session"""
    if session_id not in connected_clients:
        return
    
    for client in connected_clients[session_id]:
        if client != exclude:
            await client.send_json(message)
