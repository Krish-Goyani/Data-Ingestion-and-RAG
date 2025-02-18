import streamlit as st
import requests

# Set the page layout
st.set_page_config(page_title="Text Chunk Viewer", layout="wide")

# Title of the app
st.title("📄 Text Chunk Viewer")

# File uploader
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

# User inputs for chunk size and overlap size
chunk_size = st.number_input("Chunk Size (Characters)", min_value=100, max_value=5000, value=1000, step=100)
chunk_overlap = st.number_input("Chunk Overlap (Characters)", min_value=0, max_value=1000, value=200, step=50)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    # Send the file along with chunk size and overlap to FastAPI backend
    files = {"file": uploaded_file.getvalue()}
    params = {"chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap)}
    response = requests.post("http://127.0.0.1:8000/upload/", files=files, params=params)

    if response.status_code == 200:
        result = response.json()
        chunks = result["chunks"]

        st.subheader("📌 Extracted Chunks")

        # Display chunks directly without using expanders
        for i, chunk in enumerate(chunks):
            # Display chunk text with some spacing
            st.markdown(f"**Chunk {i+1}:**")
            st.text_area(f"Chunk {i+1}", value=chunk, height=200, max_chars=chunk_size, disabled=True)
            st.markdown("---")  # Separator between chunks

    else:
        st.error("Failed to process the file.")
