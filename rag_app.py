# Dependencies
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

# FAISS (for Arabic with SentenceTransformer)
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
faiss_dimension = embedding_model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(faiss_dimension)
faiss_document_texts = []  # Store FAISS document texts

# Chroma (for English and French with Ollama)
persist_directory = "./RAG/chroma_db"
base_url = "http://127.0.0.1:11434"
vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=OllamaEmbeddings(model="llama3.1", base_url=base_url),
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Streamlit
language = st.sidebar.selectbox("Choose Language | اختر اللغة | Choisissez la langue", ["English", "العربية", "Français"])

# Language-specific Titles and Labels
if language == "English":
    st.title("Multilingual Chatbot")
    sidebar_title = "Settings"
    upload_header = "Upload Documents (PDF Only)"
    processing_message = "Processing uploaded documents..."
    success_message = "Documents added and indexed successfully!"
    query_label = "Enter your question:"
    button_label = "Get Answer"
    no_results_message = "No relevant answers found."
    answer_label = "**Answer:**"
    spinner_message = "Searching for the answer..."
    model_options = ["Llama 3.1", "Llama 3.2:1b"]
    selected_model = st.sidebar.selectbox("Choose Model", model_options)
elif language == "العربية":
    st.title("منصة تفاعلية متعددة اللغات")
    sidebar_title = "الإعدادات"
    upload_header = "(فقط PDF) تحميل المستندات"
    processing_message = "...جاري معالجة المستندات"
    success_message = "تمت إضافة المستندات وفهرستها بنجاح!"
    query_label = ": أدخل سؤالك"
    button_label = "احصل على الإجابة"
    no_results_message = "لم يتم العثور على إجابات ذات صلة."
    answer_label = "**: الإجابة**"
    spinner_message = "...جاري البحث عن الإجابة"
    selected_model = None  # No model selection for Arabic
elif language == "Français":
    st.title("Chatbot Multilingue")
    sidebar_title = "Paramètres"
    upload_header = "Télécharger des documents (PDF uniquement)"
    processing_message = "Traitement des documents téléchargés..."
    success_message = "Documents ajoutés et indexés avec succès!"
    query_label = "Entrez votre question :"
    button_label = "Obtenez une réponse"
    no_results_message = "Aucune réponse pertinente trouvée."
    answer_label = "**Réponse :**"
    spinner_message = "Recherche de la réponse..."
    model_options = ["Llama 3.1", "Llama 3.2:1b"]
    selected_model = st.sidebar.selectbox("Choisissez un modèle", model_options)

# Sidebar: Settings and Upload
st.sidebar.header(sidebar_title)
st.sidebar.header(upload_header)
uploaded_files = st.sidebar.file_uploader(upload_header, accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.sidebar.write(processing_message)
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

            # Add documents dynamically based on language
            if language == "العربية":
                embeddings = embedding_model.encode(texts)
                faiss_index.add(np.array(embeddings))
                faiss_document_texts.extend(texts)
            elif language in ["English", "Français"]:
                vector_store.add_texts(texts)
        except Exception as e:
            st.sidebar.error(f"Failed to process {uploaded_file.name}: {e}")
            continue

    vector_store.persist()
    st.sidebar.success(success_message)

# Main Chatbot Functionality
query = st.text_input(query_label)

if st.button(button_label):
    if query.strip():
        try:
            if language == "العربية":
                # Arabic: Use FAISS with SentenceTransformer
                query_embedding = embedding_model.encode([query])
                distances, indices = faiss_index.search(np.array(query_embedding), 5)
                if len(indices[0]) > 0:
                    st.write(answer_label)
                    for idx in indices[0]:
                        st.write(f"- {faiss_document_texts[idx]}")
                else:
                    st.write(no_results_message)
            elif language in ["English", "Français"]:
                # English and French: Use selected Llama model with Chroma
                model_name = "llama3.1" if selected_model == "Llama 3.1" else "llama3.2:1b"
                llm = Ollama(model=model_name, base_url=base_url)
                retrieval_chain = create_retrieval_chain(
                    retriever, create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
                )
                with st.spinner(spinner_message):
                    response = retrieval_chain.invoke({"input": query})
                    st.write(answer_label)
                    st.write(response.get("answer", no_results_message))
        except Exception as e:
            st.error(f"An error occurred: {e}" if language in ["English", "Français"] else f"حدث خطأ: {e}")
    else:
        st.error("Please enter a valid question." if language in ["English", "Français"] else "يرجى إدخال سؤال صحيح.")
