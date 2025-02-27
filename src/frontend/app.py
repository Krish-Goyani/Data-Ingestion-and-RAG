import streamlit as st
import requests
import base64

# Set page config and title.
st.set_page_config(page_title="RAG Application", layout="wide")
st.title("RAG Application Frontend")

# Backend endpoint URLs
FILE_PROCESS_ENDPOINT = "http://127.0.0.1:8000/upload/"
QUERY_RESPONSE_ENDPOINT = "http://127.0.0.1:8000/query/"

# ------------------------------
# Step 1: File Upload & Processing
# ------------------------------
if "file_processed" not in st.session_state:
    st.subheader("Step 1: Upload File and Process")
    with st.form(key="file_upload_form"):
        uploaded_file = st.file_uploader("Upload a file (.txt or .pdf)", type=["txt", "pdf"])
        chunk_size = st.number_input("Chunk Size (Characters)", min_value=100, max_value=5000, value=700, step=100)
        chunk_overlap = st.number_input("Chunk Overlap (Characters)", min_value=0, max_value=1000, value=20, step=50)
        submit_file = st.form_submit_button("Process File")
    
    if submit_file:
        if uploaded_file is None:
            st.error("Please upload a file")
        else:
            st.info("Processing file, please wait...")
            # Build form data and file payload.
            data = {
                "chunk_size": str(chunk_size),
                "chunk_overlap": str(chunk_overlap)
            }
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            try:
                response = requests.post(url=FILE_PROCESS_ENDPOINT, data=data, files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.file_processed = True
                    st.session_state.process_result = result
                    st.success("File processed successfully!")
                else:
                    st.error("File processing failed with status code " + str(response.status_code))
            except Exception as e:
                st.error(f"Error processing file: {e}")

# ------------------------------
# Step 2: Chat Interface for Query Response
# ------------------------------
#st.session_state.file_processed = True
if "file_processed" in st.session_state:
    st.subheader("Step 2: Ask Your Query")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display previous chat messages.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input field.
    prompt = st.chat_input("Enter your query")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Send the query as form data.
        data = {"query": prompt}
        try:
            response = requests.post(url=QUERY_RESPONSE_ENDPOINT, data=data)
            if response.status_code == 200:
                data_json = response.json()
                answer = data_json.get("response", "No answer provided.")
                final_chunks = data_json.get("final_chunks", [])
                retrieved_images = data_json.get("retrieved_images", [])
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                
                if final_chunks:
                    st.subheader("Final Re-Ranked Chunks:")
                    for i, chunk in enumerate(final_chunks):
                        st.markdown(f"**Chunk {i+1}:** {chunk}")
                
                if retrieved_images:
                    st.subheader("Retrieved Images:")
                    for i, img_b64 in enumerate(retrieved_images):
                        try:
                            img_data = base64.b64decode(img_b64)
                            st.image(img_data, caption=f"Image {i+1}", use_column_width=True)
                        except Exception as e:
                            st.error(f"Error decoding image {i+1}: {e}")
            else:
                st.error("Query response failed with status code " + str(response.status_code))
        except Exception as e:
            st.error(f"Error generating query response: {e}")
