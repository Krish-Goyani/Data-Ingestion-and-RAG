import os
import httpx
import unstructured_client
from unstructured_client.models import shared
from src.app.config.settings import settings
from fastapi import UploadFile

class UnstructuredAPIService:
    def __init__(self):
        # Initialize the unstructured client with the API key and server URL.
        self.client = unstructured_client.UnstructuredClient(
            api_key_auth=settings.UNSTRUCTURED_API_KEY,
            server_url=settings.UNSTRUCTURED_API_URL,
            client=httpx.Client(verify=False)
        )

    async def process_file(self, upload_file: UploadFile):
        """
        Process the UploadFile by sending its content to the unstructured API.
        
        Args:
            upload_file (UploadFile): The uploaded file from FastAPI.
        
        Returns:
            List[str]: A list of base64 encoded images (and table data) extracted from the file.
        """
        # Ensure the file pointer is at the beginning.
        upload_file.file.seek(0)
        
        # Read file content asynchronously.
        file_content = await upload_file.read()
        if not file_content:
            raise Exception(f"Uploaded file {upload_file.filename} is empty.")
        
        # Build the request payload using the raw bytes.
        req = {
            "partition_parameters": {
                "files": {
                    "content": file_content,  # raw bytes are passed directly
                    "file_name": upload_file.filename,
                },
                "strategy": shared.Strategy.HI_RES,
                "languages": ['eng'],
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 15,
                "extract_image_block_types": ["Image", "Table"],
                "pdf_infer_table_structure": True,
            }
        }
        
        try:
            # Call the unstructured API asynchronously.
            response = await self.client.general.partition_async(request=req)
            # Convert response elements to a list.
            elements = list(response.elements)
            # Filter and return the base64 images (or table images) from the metadata.
            return [x["metadata"]["image_base64"] for x in elements if x["type"] in ["Image", "Table"]]
        except Exception as e:
            raise Exception(f"Error processing file {upload_file.filename}: {e}")
