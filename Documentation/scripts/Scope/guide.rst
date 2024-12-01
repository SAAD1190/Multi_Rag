Step-by-Step Guide to Using the Chatbot Interface
--------------------------------------------------
.. figure:: /Documentation/images/guide.png
   :width: 700
   :align: center
   :alt: Guide

--------------------------------------------------

Prerequisites
-------------

- **Install Ollama Server**: Ensure that the Ollama server is installed and running on your local machine. This is essential for embedding generation and model execution.
  - Visit the official Ollama documentation for installation instructions.

- **Download Required Models**: Use the `ollama` command-line tool to download the necessary models, such as `mistral`. Example:
   ```
   ollama download mistral
   ```
This ensures that the required models are available locally for processing English and French documents.

- **Python Environment**: Ensure all required Python packages are installed:

   ```
   pip install streamlit PyPDF2 langchain faiss-cpu sentence-transformers
   ```

- **Start Ollama Server**: Run the Ollama server before launching the chatbot:


These steps must be completed before using the chatbot to ensure smooth operation and model availability.

Streamlit Chatbot Interface
-------------

Follow these steps to effectively interact with the multilingual chatbot and retrieve answers from uploaded documents:

1. **Launch the Application**:
   - Open the application by running the Streamlit script in your Python environment (streamlit run rag_app.py).
   - The interface will load in your default web browser.

2. **Select Your Preferred Language**:
   - Navigate to the sidebar on the left side of the interface.
   - Use the dropdown menu labeled "Choose Language | اختر اللغة | Choisissez la langue" to select one of the available languages: English, العربية, or Français.
   - The interface will automatically update to display all prompts, messages, and labels in the selected language.

3. **Upload Documents**:
   - In the sidebar, find the section labeled "Upload Documents (PDF Only)" or its equivalent in the chosen language.
   - Click the "Browse Files" button and select one or more PDF files from your device.
   - Uploaded files will be processed, with their content indexed for retrieval.
   - A success message will confirm that the documents have been added and indexed successfully.

4. **Ask a Question**:
   - In the main interface, locate the text input field labeled "Enter your question:" (or its equivalent in the selected language).
   - Type your question in natural language, ensuring it relates to the content of the uploaded documents.

5. **Get Your Answer**:
   - Click the button labeled "Get Answer" (or its language-specific equivalent).
   - The chatbot will process your query and search for relevant answers in the indexed documents.
   - If relevant information is found, it will display the answer under the label "**Answer:**".
   - If no relevant information is found, you will see a message indicating that no results are available.

6. **Switch Between Models (English and French Only)**:
   - For English or French queries, use the "Choose Model" dropdown in the sidebar to select either "Llama 3.1" or "Llama 3.2:1b".
   - The selected model will be used to generate responses for your queries.

7. **Review Results**:
   - The application will display the retrieved answer or document excerpts directly in the main interface.
   - For Arabic queries, the relevant document chunks retrieved by FAISS will be listed.
   - For English and French queries, a natural language response will be generated using the Llama model.

8. **Handle Errors (if any)**:
   - If an error occurs during file upload or processing, the interface will display a detailed error message.
   - Ensure that the uploaded files are in PDF format and that your question is valid and relevant.