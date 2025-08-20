
class BlockchainService:
    async def initialize(self): pass
    async def cleanup(self): pass
    async def health_check(self): return "healthy"
    async def get_transaction_count(self): return 0

blockchain_service = BlockchainService()
