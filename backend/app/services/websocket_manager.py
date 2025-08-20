
class WebSocketManager:
    async def connect(self, websocket, client_id):
        await websocket.accept()
    def disconnect(self, client_id): pass
    async def send_personal_message(self, message, client_id): pass
    async def subscribe(self, client_id, events): pass
    async def unsubscribe(self, client_id, events): pass

websocket_manager = WebSocketManager()
