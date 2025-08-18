from fastapi import FastAPI

app = FastAPI(title="AI Music Dev Backend", version="0.1.0", docs_url="/docs")

@app.get("/")
async def root():
    return {"message": "Dev backend running", "health": "/health"}

@app.get("/health")
async def health():
    return {"status": "ok"}
