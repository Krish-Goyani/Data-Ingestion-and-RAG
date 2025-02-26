from typing import List, Dict
from fastapi import Depends
from src.app.config.database import mongodb_database

class ImageMappingRepo:
    def __init__(self, collection = Depends(mongodb_database.get_images_collection)):
        self.collection = collection

    async def store_image_mappings(self, record_ids: List[str], base64_list: List[str]) -> List[str]:
        """
        Stores a list of record IDs with corresponding base64 images into the MongoDB collection.
        
        Args:
            record_ids (List[str]): List of unique record IDs.
            base64_list (List[str]): List of base64 encoded images.
        
        Returns:
            List[str]: A list of inserted document IDs.
        """
        if len(record_ids) != len(base64_list):
            raise ValueError("The number of record IDs must match the number of base64 images.")
        
        documents = [
            {"record_id": rid, "base64_image": b64}
            for rid, b64 in zip(record_ids, base64_list)
        ]
        
        result = await self.collection.insert_many(documents)
        return [str(id_) for id_ in result.inserted_ids]

    async def fetch_base64_images(self, record_ids: List[str]) -> List[str]:
        """
        Fetches base64_image field values from documents whose record_id is in the provided list.
        
        Args:
            record_ids (List[str]): List of record IDs to query.
        
        Returns:
            List[str]: A list of base64 image strings from the matching documents.
        """
        cursor = self.collection.find({"record_id": {"$in": record_ids}})
        documents = await cursor.to_list(length=None)
        return [doc["base64_image"] for doc in documents if "base64_image" in doc]
