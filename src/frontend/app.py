import streamlit as st
import requests

# Set the page layout
st.set_page_config(page_title="Text Chunk Viewer", layout="wide")

# Title of the app
st.title("📄 Text Chunk Viewer")

# Create a form for file upload and processing
with st.form(key='upload_form'):
    # File uploader within the form
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    # User inputs for chunk size and overlap size within the form
    chunk_size = st.number_input("Chunk Size (Characters)", min_value=100, max_value=5000, value=700, step=100)
    chunk_overlap = st.number_input("Chunk Overlap (Characters)", min_value=0, max_value=1000, value=20, step=50)

    # User query input
    user_query = st.text_input("Enter your query to retrieve similar chunks", "")

    # Submit button for the form
    submit_button = st.form_submit_button(label='Process File')

# Check if the form is submitted and a file is uploaded
if submit_button:
    if uploaded_file is not None and user_query:
        st.success(f"Uploaded: {uploaded_file.name}")

        # Send the file along with chunk size, overlap, and user query to FastAPI backend
        files = {"file": uploaded_file.getvalue()}
        params = {
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "query": user_query
        }
        response = requests.post("http://127.0.0.1:8000/upload/", files=files, params=params)

        if response.status_code == 200:
            result = response.json()

            # Extract chunks from the response
            chunks = result.get("chunks", [])
            pinecone_chunks = result.get("pinecone_chunks", [])
            qdrant_chunks = result.get("qdrant_chunks", [])
            all_chunks = result.get("all_chunks", [])
            st.subheader("📌 All Chunks")
            for i, chunk in enumerate(all_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.text_area(f"Chunk {i+1}", value=chunk, height=200, max_chars=chunk_size, disabled=True)
                st.markdown("---")
            # Compare Pinecone and Qdrant chunks side by side
            st.subheader("🔍 Pinecone vs. Qdrant Chunk Comparison")
            if pinecone_chunks and qdrant_chunks:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🟢 Pinecone Retrieved Chunks")
                    for i, chunk in enumerate(pinecone_chunks):
                        st.text_area(f"Pinecone Chunk {i+1}", value=chunk, height=150, disabled=True)
                        st.markdown("---")

                with col2:
                    st.markdown("### 🔵 Qdrant Retrieved Chunks")
                    for i, chunk in enumerate(qdrant_chunks):
                        st.text_area(f"Qdrant Chunk {i+1}", value=chunk, height=150, disabled=True)
                        st.markdown("---")
            else:
                st.warning("No similar chunks found in Pinecone or Qdrant.")

            # Display extracted chunks
            st.subheader("📌 Extracted Chunks")
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.text_area(f"Chunk {i+1}", value=chunk, height=200, max_chars=chunk_size, disabled=True)
                st.markdown("---")
            st.subheader("📌 All Chunks")
            for i, chunk in enumerate(all_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.text_area(f"Chunk {i+1}", value=chunk, height=200, max_chars=chunk_size, disabled=True)
                st.markdown("---")

        else:
            st.error("Failed to process the file.")
    else:
        st.warning("Please upload a file and enter a query before submitting.")
