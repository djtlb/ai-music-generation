
"""Simple AI Music Generation for Horizon"""
import os
import uuid
import wave
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="AI Music Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://beataddicts.pro", "http://localhost:3000", "http://localhost:8000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory storage for demo; in production, use a database
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

 

GENERATED_AUDIO_DIR = "generated_audio"
os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    project_id = str(uuid.uuid4())[:8]
    # Simulate audio generation: create a 1-second silent WAV file
    audio_path = os.path.join(GENERATED_AUDIO_DIR, f"{project_id}.wav")
    framerate = 44100
    duration = 1
    silence_frames = (b"\x00\x00") * (framerate * duration)
    with wave.open(audio_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(silence_frames)
    project = {
        "id": project_id,
        "prompt": prompt,
        "status": "completed",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "audio_url": f"/audio/{project_id}"
    }
    projects[project_id] = project
    return JSONResponse({
        "project_id": project_id,
        "status": "completed",
        "audio_url": f"/audio/{project_id}",
        "message": "Music generated successfully!"
    })

@app.get("/audio/{project_id}")
async def get_audio(project_id: str):
    audio_path = os.path.join(GENERATED_AUDIO_DIR, f"{project_id}.wav")
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/wav", filename=f"{project_id}.wav")
    return JSONResponse({"error": "Audio not found"}, status_code=404)


@app.get("/health")
async def health():
    return {"status": "healthy", "projects": len(projects)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
