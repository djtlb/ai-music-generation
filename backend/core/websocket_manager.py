"""
WebSocket connection manager for real-time updates
"""

from fastapi import WebSocket
from typing import Dict, List, Set
import json
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and real-time messaging"""
    
    def __init__(self):
        # Active connections: client_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Subscriptions: client_id -> Set[event_types]
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # Room-based connections for collaboration
        self.rooms: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "welcome",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to AI Music Generation Platform"
        }, client_id)
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        
        # Remove from all rooms
        for room_id, members in self.rooms.items():
            if client_id in members:
                members.remove(client_id)
        
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {str(e)}")
                self.disconnect(client_id)
    
    async def broadcast_message(self, message: dict, exclude_client: str = None):
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id != exclude_client:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to broadcast to {client_id}: {str(e)}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def send_to_room(self, room_id: str, message: dict, exclude_client: str = None):
        """Send message to all clients in a room"""
        if room_id not in self.rooms:
            return
        
        disconnected_clients = []
        
        for client_id in self.rooms[room_id]:
            if client_id != exclude_client and client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to send room message to {client_id}: {str(e)}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def subscribe(self, client_id: str, events: List[str]):
        """Subscribe client to specific events"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].update(events)
            await self.send_personal_message({
                "type": "subscription_confirmed",
                "events": list(self.subscriptions[client_id]),
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
    
    async def unsubscribe(self, client_id: str, events: List[str]):
        """Unsubscribe client from specific events"""
        if client_id in self.subscriptions:
            for event in events:
                self.subscriptions[client_id].discard(event)
            
            await self.send_personal_message({
                "type": "unsubscription_confirmed",
                "events": list(self.subscriptions[client_id]),
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
    
    async def join_room(self, client_id: str, room_id: str):
        """Add client to a room"""
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        
        self.rooms[room_id].add(client_id)
        
        await self.send_personal_message({
            "type": "room_joined",
            "room_id": room_id,
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)
        
        # Notify other room members
        await self.send_to_room(room_id, {
            "type": "user_joined_room",
            "client_id": client_id,
            "room_id": room_id,
            "timestamp": datetime.utcnow().isoformat()
        }, exclude_client=client_id)
    
    async def leave_room(self, client_id: str, room_id: str):
        """Remove client from a room"""
        if room_id in self.rooms and client_id in self.rooms[room_id]:
            self.rooms[room_id].remove(client_id)
            
            await self.send_personal_message({
                "type": "room_left",
                "room_id": room_id,
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
            
            # Notify other room members
            await self.send_to_room(room_id, {
                "type": "user_left_room",
                "client_id": client_id,
                "room_id": room_id,
                "timestamp": datetime.utcnow().isoformat()
            }, exclude_client=client_id)
    
    async def notify_subscribers(self, event_type: str, data: dict):
        """Send notification to all subscribers of an event type"""
        message = {
            "type": "event_notification",
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected_clients = []
        
        for client_id, subscribed_events in self.subscriptions.items():
            if event_type in subscribed_events and client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to notify subscriber {client_id}: {str(e)}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def get_connection_stats(self) -> dict:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "total_rooms": len(self.rooms),
            "room_stats": {room_id: len(members) for room_id, members in self.rooms.items()},
            "subscription_stats": {
                client_id: list(events) for client_id, events in self.subscriptions.items()
            }
        }
