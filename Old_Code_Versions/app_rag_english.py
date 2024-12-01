# Dependencies
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# Load environment variables
load_dotenv()

# Vector Store Directory
persist_directory = "./RAG/chroma_db"

# Streamlit App Setup
st.title("GI-IADS Chatbot")

# User Input: Choose LLM
st.sidebar.header("LLM Configuration")
available_llms = {
    "llama 3.1": "llama3.1",
    "llama3.2:1b": "llama3.2:1b",
}
selected_llm = st.sidebar.selectbox("Choose LLM", list(available_llms.keys()))
llm_model_name = available_llms[selected_llm]

# Set up LLM and Embedding Models
base_url = "http://127.0.0.1:11434"
llm = Ollama(model=llm_model_name, base_url=base_url)
embed_model = OllamaEmbeddings(model=llm_model_name, base_url=base_url)

# Vector Store
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

# Retrieval Chain Setup
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combined_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, combined_docs_chain)

# Document Upload Section (PDFs Only)
st.sidebar.header("Add Documents (PDF Only)")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Files", accept_multiple_files=True, type=["pdf"]
)

if uploaded_files:
    st.sidebar.write("Processing uploaded documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for uploaded_file in uploaded_files:
        try:
            # Initialize a PDF reader
            pdf_reader = PdfReader(uploaded_file)

            # Extract text from each page of the PDF
            file_content = ""
            for page in pdf_reader.pages:
                file_content += page.extract_text()

            # Split text into chunks and add to the vector store
            texts = text_splitter.split_text(file_content)
            vector_store.add_texts(texts)
        except Exception as e:
            st.sidebar.error(f"Failed to process {uploaded_file.name}: {str(e)}")
            continue

    # Persist the updated vector store
    vector_store.persist()
    st.sidebar.success("PDF files added and indexed successfully!")

# Main Chatbot Functionality
st.subheader("Ask a Question")
question = st.text_input("Please ask a question")
if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Searching for the answer..."):
            response = retrieval_chain.invoke({"input": question})
            st.write("**Answer:**")
            st.write(response.get("answer", "No answer found."))
    else:
        st.error("Please enter a valid question.")