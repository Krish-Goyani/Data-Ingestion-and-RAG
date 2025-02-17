import streamlit as st
import requests

# FastAPI endpoint URL
FASTAPI_URL = "http://127.0.0.1:8000/upload/"

st.title("Upload a File and Send to FastAPI")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "json", "xlsx", "pdf"])  # Allowed file types

if uploaded_file is not None:
    st.write(f"Filename: {uploaded_file.name}")
    
    # Read file as bytes
    file_bytes = uploaded_file.getvalue()
    
    # Send file to FastAPI without saving
    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
    with st.spinner("Uploading file..."):
        try:
            response = requests.post(FASTAPI_URL, files=files)
            response_data = response.json()  # Parse response as JSON

            if response.status_code == 200:
                st.success("File successfully uploaded and processed!")
                st.json(response_data)  # Display API response

            else:
                # Extract custom error message if API returns a structured JSON error
                error_message = response_data.get("message", "An unknown error occurred.")
                st.error(f"Error: {error_message}")

        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
