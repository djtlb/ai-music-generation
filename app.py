"""Simple AI Music Generation for Horizon"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime

app = FastAPI(title="AI Music Generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Simple in-memory storage
projects = {}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>AI Music Generator</title></head>
    <body style="font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px;">
        <h1>🎵 AI Music Generator</h1>
        <form onsubmit="generateMusic(event)">
            <div style="margin: 20px 0;">
                <label>Music Prompt:</label><br>
                <input type="text" id="prompt" placeholder="Describe your music..." style="width: 100%; padding: 10px;">
            </div>
            <button type="submit" style="background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 5px;">Generate</button>
        </form>
        <div id="result" style="margin-top: 30px;"></div>
        
        <script>
        async function generateMusic(e) {
            e.preventDefault();
            const prompt = document.getElementById('prompt').value;
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt})
            });
            const data = await response.json();
            document.getElementById('result').innerHTML = 
                `<div style="background: #f0f8ff; padding: 15px; border-radius: 5px;">
                    <h3>✅ Generated!</h3>
                    <p><strong>Project ID:</strong> ${data.project_id}</p>
                    <p><strong>Status:</strong> ${data.status}</p>
                    <p><strong>Prompt:</strong> "${prompt}"</p>
                </div>`;
        }
        </script>
    </body>
    </html>
    """

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    project_id = str(uuid.uuid4())[:8]
    
    project = {
        "id": project_id,
        "prompt": data.get("prompt", ""),
        "status": "completed",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    projects[project_id] = project
    
    return JSONResponse({
        "project_id": project_id,
        "status": "completed",
        "message": "Music generated successfully!"
    })

@app.get("/health")
async def health():
    return {"status": "healthy", "projects": len(projects)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
