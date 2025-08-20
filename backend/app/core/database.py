
async def init_db(): pass
async def close_db(): pass
async def get_db_session():
    class MockSession:
        async def execute(self, query): pass
    yield MockSession()
