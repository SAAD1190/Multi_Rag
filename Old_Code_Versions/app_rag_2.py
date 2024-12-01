# Dependencies
import os
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
import faiss

# Initialize FAISS (for SentenceTransformer - Arabic)
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
faiss_dimension = embedding_model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(faiss_dimension)
faiss_document_texts = []  # To store FAISS document texts

# Initialize Chroma (for English with Ollama)
persist_directory = "./RAG/chroma_db"
base_url = "http://127.0.0.1:11434"
vector_store = Chroma(persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model="llama3.1", base_url=base_url))
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combined_docs_chain = create_stuff_documents_chain(Ollama(model="llama3.1", base_url=base_url), retrieval_qa_chat_prompt)

# Streamlit App Setup
st.title("Multilingual Chatbot | منصة متعددة اللغات")

# Sidebar: Language Selection
st.sidebar.header("Settings")
language = st.sidebar.selectbox("Choose Language", ["Arabic", "English"])

# Sidebar: Model Selection for English
selected_model = None
if language == "English":
    selected_model = st.sidebar.selectbox("Choose Model", ["Llama 3.1", "Llama 3.2:1b"])

# Sidebar: Upload Documents
st.sidebar.header("Upload Documents | تحميل المستندات (PDF Only)")
uploaded_files = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.sidebar.write("Processing uploaded documents... | ... جاري معالجة المستندات ")
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

            # Add to FAISS and Chroma
            embeddings = embedding_model.encode(texts)
            faiss_index.add(np.array(embeddings))
            faiss_document_texts.extend(texts)

            vector_store.add_texts(texts)
        except Exception as e:
            st.sidebar.error(f"Failed to process {uploaded_file.name}: {e}")
            continue

    vector_store.persist()
    st.sidebar.success("Documents added and indexed successfully!")

# Main Chatbot Functionality
st.subheader(f"Ask a Question in {language}")
query = st.text_input("Enter your question | أدخل سؤالك")

if st.button("Get Answer | احصل على الإجابة"):
    if query.strip():
        try:
            if language == "Arabic":
                # Arabic: Use FAISS with SentenceTransformer
                query_embedding = embedding_model.encode([query])
                distances, indices = faiss_index.search(np.array(query_embedding), 5)
                if len(indices[0]) > 0:
                    st.write("**Relevant Answers (FAISS):**")
                    for idx in indices[0]:
                        st.write(f"- {faiss_document_texts[idx]}")
                else:
                    st.write("No relevant answers found.")
            elif language == "English":
                # English: Use Ollama with Chroma
                model_name = "llama3.1" if selected_model == "Llama 3.1" else "llama3.2:1b"
                llm = Ollama(model=model_name, base_url=base_url)
                retrieval_chain = create_retrieval_chain(
                    retriever, create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
                )
                with st.spinner("Searching for the answer..."):
                    response = retrieval_chain.invoke({"input": query})
                    st.write("**Answer (Ollama):**")
                    st.write(response.get("answer", "No answer found."))
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid question.")
