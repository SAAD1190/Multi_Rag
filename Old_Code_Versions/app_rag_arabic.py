# Streamlit App with FAISS
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Initialize FAISS index
dimension = embedding_model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(dimension)

# Store document texts
document_texts = []

# Streamlit App
st.title("Arabic Chatbot with FAISS")

# Upload and Process PDF Documents
st.sidebar.header("Upload Arabic Documents (PDF Only)")
uploaded_files = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.sidebar.write("Processing uploaded documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for uploaded_file in uploaded_files:
        try:
            # Extract text from PDF
            pdf_reader = PdfReader(uploaded_file)
            file_content = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    file_content += text

            # Split text into chunks
            texts = text_splitter.split_text(file_content)

            # Add to FAISS index
            embeddings = embedding_model.encode(texts)
            faiss_index.add(np.array(embeddings))
            document_texts.extend(texts)
        except Exception as e:
            st.sidebar.error(f"Failed to process {uploaded_file.name}: {e}")
            continue

    st.sidebar.success("Arabic documents added and indexed successfully!")

# Main Chatbot Functionality
st.subheader("Ask a Question in Arabic")
query = st.text_input("Enter your question in Arabic:")

if st.button("Get Answer"):
    if query.strip():
        try:
            # Preprocess query and search FAISS
            query_embedding = embedding_model.encode([query])
            distances, indices = faiss_index.search(np.array(query_embedding), 5)

            # Display results
            if len(indices[0]) > 0:
                st.write("**Relevant Answers:**")
                for idx, score in zip(indices[0], distances[0]):
                    st.write(f"- {document_texts[idx]} (Score: {score:.2f})")
            else:
                st.write("No relevant answers found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid question.")
