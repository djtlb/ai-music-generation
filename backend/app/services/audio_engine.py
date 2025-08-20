
class AudioEngine:
    async def initialize(self): pass
    async def cleanup(self): pass
    async def health_check(self): return "healthy"
    async def get_queue_size(self): return 0

audio_engine = AudioEngine()
