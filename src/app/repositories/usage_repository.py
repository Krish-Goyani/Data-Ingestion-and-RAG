from motor.motor_asyncio import AsyncIOMotorClient
from typing import Dict
from src.app.config.database import mongodb_database
from fastapi import Depends

class CostStorageRepo:
    def __init__(self, collection = Depends(mongodb_database.get_usage_collection)):
        
        self.collection = collection

    async def store_cost_details(self, cost_details: Dict) -> str:
        result = await self.collection.insert_one(cost_details)
        return str(result.inserted_id)
