"""
AI Music Generation API - Horizon Compatible Single File Application
Consolidated for web deployment without file upload requirements
"""

import asyncio
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import uuid

# Simplified FastAPI setup for Horizon
try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, HTMLResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Fallback to basic HTTP server if FastAPI not available
if not FASTAPI_AVAILABLE:
    import http.server
    import socketserver
    from urllib.parse import urlparse, parse_qs

# Configuration for Horizon deployment
class Config:
    PROJECT_NAME = "AI Music Generation Platform"
    VERSION = "2.0.0"
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://emecscbwfcvbkvxbztfa.supabase.co")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWNzY2J3ZmN2Ymt2eGJ6dGZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyOTc4NTYsImV4cCI6MjA3MDg3Mzg1Nn0.gKtvk5fiqjBToyy9w0_-DF4EmZdBQwUJCpY07XM7QJ0")
    API_KEY = os.getenv("API_KEY", "ai-music-gen-horizon-2024")

config = Config()

# In-memory storage for demo (replace with Supabase in production)
projects_db = {}
generation_logs = []
users_db = {}

# Pydantic models if FastAPI is available
if FASTAPI_AVAILABLE:
    class GenerationRequest(BaseModel):
        prompt: str
        style: Optional[str] = "pop"
        duration: Optional[int] = 30
        user_id: Optional[str] = None

    class ProjectResponse(BaseModel):
        project_id: str
        name: str
        status: str
        created_at: str
        audio_url: Optional[str] = None

# Mock AI Music Generation Service
class AIOrchestrator:
    def __init__(self):
        self.initialized = True
        
    async def initialize(self):
        """Mock initialization"""
        return True
    
    async def create_project(self, name: str, user_id: str, style_config: Dict):
        """Create a new music project"""
        project_id = str(uuid.uuid4())
        project = {
            "id": project_id,
            "name": name,
            "user_id": user_id,
            "style_config": style_config,
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
            "audio_url": None
        }
        projects_db[project_id] = project
        return project_id
    
    async def generate_full_song(self, project_id: str, style_config: Dict, **kwargs):
        """Mock song generation"""
        if project_id in projects_db:
            # Simulate generation process
            await asyncio.sleep(2)  # Simulate processing time
            
            projects_db[project_id]["status"] = "completed"
            projects_db[project_id]["audio_url"] = f"/api/audio/{project_id}.mp3"
            
            # Log generation
            log_entry = {
                "id": str(uuid.uuid4()),
                "project_id": project_id,
                "user_id": projects_db[project_id]["user_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "style": style_config.get("style", "unknown"),
                "status": "completed"
            }
            generation_logs.append(log_entry)
            
        return {"status": "completed", "project_id": project_id}
    
    async def get_project_status(self, project_id: str):
        """Get project generation status"""
        if project_id in projects_db:
            return projects_db[project_id]
        return {"error": "Project not found"}
    
    async def get_total_songs(self):
        """Get total generated songs count"""
        return len([p for p in projects_db.values() if p["status"] == "completed"])
    
    async def health_check(self):
        """Health check"""
        return "healthy"

# Global AI orchestrator instance
ai_orchestrator = AIOrchestrator()

# HTML Template for Horizon
HORIZON_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Music Generation Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .header { text-align: center; margin-bottom: 3rem; }
        .header h1 { font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        
        .music-generator { 
            background: rgba(255,255,255,0.1); 
            padding: 2rem; 
            border-radius: 15px; 
            backdrop-filter: blur(10px);
            margin-bottom: 2rem;
        }
        
        .form-group { margin-bottom: 1.5rem; }
        .form-group label { display: block; margin-bottom: 0.5rem; font-weight: bold; }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 1rem;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.2);
            color: white;
            font-size: 1rem;
        }
        .form-group input::placeholder, .form-group textarea::placeholder {
            color: rgba(255,255,255,0.7);
        }
        
        .btn { 
            background: #ff6b6b; 
            color: white; 
            padding: 1rem 2rem; 
            border: none;
            border-radius: 50px; 
            font-weight: bold; 
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
            font-size: 1rem;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        
        .projects { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-top: 2rem; }
        .project-card { 
            background: rgba(255,255,255,0.1); 
            padding: 1.5rem; 
            border-radius: 10px; 
            backdrop-filter: blur(10px);
        }
        .project-card h3 { margin-bottom: 1rem; color: #4ecdc4; }
        .status { 
            padding: 0.5rem 1rem; 
            border-radius: 20px; 
            font-size: 0.8rem; 
            font-weight: bold;
            display: inline-block;
            margin-bottom: 1rem;
        }
        .status.completed { background: #4ecdc4; color: white; }
        .status.processing { background: #feca57; color: white; }
        .status.created { background: #54a0ff; color: white; }
        
        .api-info { 
            background: rgba(0,0,0,0.3); 
            padding: 1.5rem; 
            border-radius: 10px; 
            margin-top: 2rem;
            font-family: 'Courier New', monospace;
        }
        .api-endpoint { 
            background: #2c3e50; 
            padding: 0.5rem 1rem; 
            border-radius: 5px; 
            margin: 0.5rem 0;
            font-size: 0.9rem;
        }
        
        #loading { display: none; text-align: center; margin: 1rem 0; }
        .spinner { 
            border: 4px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 4px solid #ff6b6b;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 AI Music Generation</h1>
            <p>Create professional music with artificial intelligence • <span id="status">🔄 Loading...</span></p>
        </div>
        
        <div class="music-generator">
            <h2>Generate Your Music</h2>
            <form id="musicForm">
                <div class="form-group">
                    <label for="prompt">Music Prompt:</label>
                    <textarea id="prompt" name="prompt" placeholder="Describe the music you want to create (e.g., 'Upbeat pop song with electronic elements')" rows="3" required></textarea>
                </div>
                
                <div class="form-group">
                    <label for="style">Music Style:</label>
                    <select id="style" name="style">
                        <option value="pop">Pop</option>
                        <option value="rock">Rock</option>
                        <option value="jazz">Jazz</option>
                        <option value="classical">Classical</option>
                        <option value="electronic">Electronic</option>
                        <option value="hip-hop">Hip Hop</option>
                        <option value="folk">Folk</option>
                        <option value="ambient">Ambient</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="duration">Duration (seconds):</label>
                    <input type="number" id="duration" name="duration" value="30" min="15" max="120" required>
                </div>
                
                <button type="submit" class="btn" id="generateBtn">🎼 Generate Music</button>
            </form>
            
            <div id="loading">
                <div class="spinner"></div>
                <p>Generating your music... This may take a few moments.</p>
            </div>
        </div>
        
        <div class="projects">
            <div id="projectsList">
                <!-- Projects will be loaded here -->
            </div>
        </div>
        
        <div class="api-info">
            <h3>API Endpoints</h3>
            <div class="api-endpoint">GET /api/health - System health check</div>
            <div class="api-endpoint">POST /api/generate - Generate music</div>
            <div class="api-endpoint">GET /api/projects - List all projects</div>
            <div class="api-endpoint">GET /api/stats - System statistics</div>
        </div>
    </div>
    
    <script>
        // API Base URL
        const API_BASE = '';
        
        // DOM elements
        const musicForm = document.getElementById('musicForm');
        const generateBtn = document.getElementById('generateBtn');
        const loading = document.getElementById('loading');
        const projectsList = document.getElementById('projectsList');
        const statusElement = document.getElementById('status');
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            checkSystemStatus();
            loadProjects();
        });
        
        // Check system status
        async function checkSystemStatus() {
            try {
                const response = await fetch(`${API_BASE}/api/health`);
                const data = await response.json();
                
                if (data.status === 'healthy') {
                    statusElement.innerHTML = '✅ System Online';
                    statusElement.style.color = '#4ecdc4';
                } else {
                    statusElement.innerHTML = '⚠️ System Issues';
                    statusElement.style.color = '#feca57';
                }
            } catch (error) {
                statusElement.innerHTML = '❌ System Offline';
                statusElement.style.color = '#ff6b6b';
            }
        }
        
        // Handle form submission
        musicForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(musicForm);
            const data = {
                prompt: formData.get('prompt'),
                style: formData.get('style'),
                duration: parseInt(formData.get('duration')),
                user_id: 'horizon_user_' + Math.random().toString(36).substr(2, 9)
            };
            
            generateBtn.disabled = true;
            loading.style.display = 'block';
            
            try {
                const response = await fetch(`${API_BASE}/api/generate`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.project_id) {
                    setTimeout(() => {
                        loadProjects();
                    }, 3000); // Reload projects after 3 seconds
                }
                
            } catch (error) {
                alert('Generation failed: ' + error.message);
            } finally {
                generateBtn.disabled = false;
                loading.style.display = 'none';
            }
        });
        
        // Load projects
        async function loadProjects() {
            try {
                const response = await fetch(`${API_BASE}/api/projects`);
                const data = await response.json();
                
                if (data.projects) {
                    displayProjects(data.projects);
                }
            } catch (error) {
                console.log('Failed to load projects:', error);
            }
        }
        
        // Display projects
        function displayProjects(projects) {
            if (projects.length === 0) {
                projectsList.innerHTML = '<div class="project-card"><h3>No projects yet</h3><p>Generate your first music track above!</p></div>';
                return;
            }
            
            const projectsHtml = projects.map(project => `
                <div class="project-card">
                    <h3>${project.name || 'Untitled Project'}</h3>
                    <span class="status ${project.status}">${project.status}</span>
                    <p><strong>Style:</strong> ${project.style_config?.style || 'Unknown'}</p>
                    <p><strong>Created:</strong> ${new Date(project.created_at).toLocaleString()}</p>
                    ${project.audio_url ? `<p><a href="${project.audio_url}" style="color: #4ecdc4;">🎧 Listen</a></p>` : ''}
                </div>
            `).join('');
            
            projectsList.innerHTML = projectsHtml;
        }
        
        // Auto-refresh projects every 10 seconds
        setInterval(loadProjects, 10000);
    </script>
</body>
</html>
"""

# FastAPI app if available, otherwise fallback
if FASTAPI_AVAILABLE:
    # FastAPI Application
    app = FastAPI(
        title=config.PROJECT_NAME,
        version=config.VERSION,
        description="AI Music Generation Platform - Horizon Compatible"
    )
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the main HTML interface"""
        return HTMLResponse(content=HORIZON_HTML_TEMPLATE, status_code=200)
    
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": config.VERSION,
            "service": "AI Music Generation"
        }
    
    @app.post("/api/generate")
    async def generate_music(request: GenerationRequest):
        """Generate music from prompt"""
        try:
            # Create project
            project_id = await ai_orchestrator.create_project(
                name=f"Generated: {request.prompt[:30]}...",
                user_id=request.user_id or "anonymous",
                style_config={"style": request.style, "duration": request.duration}
            )
            
            # Start generation (async)
            asyncio.create_task(ai_orchestrator.generate_full_song(
                project_id=project_id,
                style_config={"style": request.style, "duration": request.duration}
            ))
            
            return {
                "success": True,
                "project_id": project_id,
                "message": "Music generation started",
                "estimated_completion": datetime.utcnow() + timedelta(seconds=30)
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/projects")
    async def get_projects():
        """Get all projects"""
        projects = list(projects_db.values())
        return {"projects": projects, "total": len(projects)}
    
    @app.get("/api/project/{project_id}/status")
    async def get_project_status(project_id: str):
        """Get project status"""
        status = await ai_orchestrator.get_project_status(project_id)
        return status
    
    @app.get("/api/stats")
    async def get_stats():
        """Get system statistics"""
        total_songs = await ai_orchestrator.get_total_songs()
        return {
            "total_songs_generated": total_songs,
            "active_projects": len(projects_db),
            "total_logs": len(generation_logs),
            "system_uptime": "Online",
            "version": config.VERSION
        }
    
    # Export the app
    application = app

else:
    # Fallback HTTP server for basic hosting
    class HorizonHTTPHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/' or self.path == '/index.html':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(HORIZON_HTML_TEMPLATE.encode())
            elif self.path == '/api/health':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = json.dumps({"status": "healthy", "version": config.VERSION})
                self.wfile.write(response.encode())
            else:
                super().do_GET()
        
        def do_POST(self):
            if self.path == '/api/generate':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                try:
                    data = json.loads(post_data.decode())
                    project_id = str(uuid.uuid4())
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    response = json.dumps({
                        "success": True,
                        "project_id": project_id,
                        "message": "Music generation started"
                    })
                    self.wfile.write(response.encode())
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = json.dumps({"error": str(e)})
                    self.wfile.write(response.encode())
    
    def create_server(port=8000):
        """Create HTTP server"""
        with socketserver.TCPServer(("", port), HorizonHTTPHandler) as httpd:
            print(f"Serving at port {port}")
            httpd.serve_forever()
    
    application = create_server

# Main entry point
def main():
    """Main function for different deployment scenarios"""
    if FASTAPI_AVAILABLE:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
    else:
        application(int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    main()
