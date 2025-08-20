"""Supabase client integration for AI Music Generation Platform"""

import asyncio
from typing import Dict, Any, Optional, List
import structlog
from supabase import create_client, Client
from app.core.config import settings

logger = structlog.get_logger()

class SupabaseClient:
    """Supabase client wrapper for the AI Music Generation platform"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize Supabase client"""
        try:
            if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
                logger.warning("Supabase credentials not configured")
                return
                
            self.client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_ANON_KEY
            )
            self.initialized = True
            logger.info("✅ Supabase client initialized")
            
            # Test connection
            response = self.client.table('users').select('*').limit(1).execute()
            logger.info("✅ Supabase connection verified")
            
        except Exception as e:
            logger.warning(f"Supabase initialization failed: {e}")
            self.initialized = False
    
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user in Supabase"""
        if not self.initialized or not self.client:
            return {"error": "Supabase not initialized"}
        
        try:
            response = self.client.table('users').insert(user_data).execute()
            return {"success": True, "data": response.data}
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return {"error": str(e)}
    
    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user by ID"""
        if not self.initialized or not self.client:
            return {"error": "Supabase not initialized"}
        
        try:
            response = self.client.table('users').select('*').eq('id', user_id).execute()
            return {"success": True, "data": response.data}
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return {"error": str(e)}
    
    async def save_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save music project to Supabase"""
        if not self.initialized or not self.client:
            return {"error": "Supabase not initialized"}
        
        try:
            response = self.client.table('music_projects').insert(project_data).execute()
            return {"success": True, "data": response.data}
        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            return {"error": str(e)}
    
    async def get_projects(self, user_id: str) -> Dict[str, Any]:
        """Get all projects for a user"""
        if not self.initialized or not self.client:
            return {"error": "Supabase not initialized"}
        
        try:
            response = self.client.table('music_projects').select('*').eq('user_id', user_id).execute()
            return {"success": True, "data": response.data}
        except Exception as e:
            logger.error(f"Failed to get projects: {e}")
            return {"error": str(e)}
    
    async def save_generation_log(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save generation activity log"""
        if not self.initialized or not self.client:
            return {"error": "Supabase not initialized"}
        
        try:
            response = self.client.table('generation_logs').insert(log_data).execute()
            return {"success": True, "data": response.data}
        except Exception as e:
            logger.error(f"Failed to save log: {e}")
            return {"error": str(e)}
    
    async def get_analytics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics data"""
        if not self.initialized or not self.client:
            return {"error": "Supabase not initialized"}
        
        try:
            query = self.client.table('generation_logs').select('*')
            if user_id:
                query = query.eq('user_id', user_id)
            
            response = query.execute()
            return {"success": True, "data": response.data}
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup Supabase resources"""
        if self.client:
            # Supabase client doesn't require explicit cleanup
            logger.info("Supabase client cleaned up")

# Global instance
supabase_client = SupabaseClient()
