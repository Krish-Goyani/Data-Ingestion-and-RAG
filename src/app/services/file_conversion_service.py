import os
import tempfile

import pymupdf4llm
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from starlette import status


class FileConversionService:
    def __init__(self, output_format="markdown") -> None:
        """self.config = {
            "output_format": output_format,
            "languages": "en",
            "disable_image_extraction": True,
            "workers": 6,
            "force_ocr" : False,

        }

        self.config_parser = ConfigParser(self.config)

        self.converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config=self.config_parser.generate_config_dict(),
        )"""
        pass

    """async def convert_to_markdown(self, file_bytes: UploadFile):
        try:
            # ✅ Use in-memory file instead of saving to disk
            with tempfile.NamedTemporaryFile(
                delete=False, mode="wb", suffix=".pdf"
            ) as temp_pdf:
                # Write the bytes to the temporary file
                temp_pdf.write(await file_bytes.read())
                temp_pdf_path = temp_pdf.name  # Get the path of the temporary file

            # Now that we have a file path, pass it to the converter
            rendered = self.converter(temp_pdf_path)

            text, metadata, images = text_from_rendered(rendered)

            # Optionally, delete the temp file after conversion
            os.remove(temp_pdf_path)

            return text
        
        except Exception as e:
            raise JSONResponse(
                content={"detail": f"Error during PDF conversion: {str(e)}"},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )"""

    async def convert_to_makedown(self, file_bytes: UploadFile):
        try:
            # ✅ Use in-memory file instead of saving to disk
            with tempfile.NamedTemporaryFile(
                delete=False, mode="wb", suffix=".pdf"
            ) as temp_pdf:
                # Write the bytes to the temporary file
                temp_pdf.write(await file_bytes.read())
                temp_pdf_path = temp_pdf.name  # Get the path of the temporary file

            md_text = pymupdf4llm.to_markdown(
                temp_pdf,
                write_images=False,
            )
            os.remove(temp_pdf_path)
            return md_text

        except Exception as e:
            print(e)

    async def convert_to_text(self, file_bytes: UploadFile):
        try:
            # Read file content
            content = await file_bytes.read()
            text = content.decode("utf-8")
            return text
        except Exception as e:
            print(e)
