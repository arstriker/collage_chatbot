# AI Chatbot for College Customer Support

## Overview
This project is an AI-powered chatbot designed to provide customer support for a college or university. It leverages a Retrieval Augmented Generation (RAG) approach, allowing it to answer questions based on information from uploaded PDF documents. The chatbot aims to assist students, prospective students, and staff by providing quick and relevant answers to their queries.

## Features
- **AI-Powered Q&A:** Ask questions in natural language and receive answers from the chatbot.
- **PDF Document Knowledge Base:** Upload PDF documents (e.g., course catalogs, FAQs, policy documents) to create a knowledge base for the chatbot.
- **Local LLM Processing:** Utilizes Ollama to run language models locally, ensuring data privacy and control. Currently configured for `llama3.2:3b`.
- **Efficient Information Retrieval:** Employs ChromaDB for vector storage and similarity search to quickly find relevant information from the documents.
- **Relevance Re-ranking:** Uses a CrossEncoder model to re-rank retrieved document chunks, improving the quality of context provided to the LLM.
- **User-Friendly Web Interface:** Built with Streamlit for easy interaction, including document uploading and chatting.
- **Helpful Quick Links:** The chatbot interface includes a sidebar with configurable links to important college resources.
- **Modular Design:** Separate applications for PDF uploading/processing and the chatbot interface.

## How it Works
The chatbot uses a Retrieval Augmented Generation (RAG) pipeline:
1.  **PDF Upload and Processing (`pdfupload.py`):**
    *   Users upload PDF documents through the `pdfupload.py` Streamlit interface.
    *   These documents are loaded, parsed, and split into smaller, manageable chunks using Langchain and PyMuPDF.
2.  **Embedding and Vector Storage:**
    *   Each text chunk is converted into a numerical vector (embedding) using the `nomic-embed-text` model via Ollama.
    *   These embeddings, along with their corresponding text and metadata, are stored in a ChromaDB persistent vector store (in the `./demo-rag-chroma` directory).
3.  **User Query and Retrieval (`chatbot.py`):**
    *   A user asks a question through the `chatbot.py` Streamlit interface.
    *   The user's question is also converted into an embedding using the same `nomic-embed-text` model.
    *   ChromaDB is queried to find the most similar document chunks (based on vector similarity) to the user's question.
4.  **Re-ranking:**
    *   The retrieved document chunks are then re-ranked using a `cross-encoder/ms-marco-MiniLM-L-6-v2` model to further refine the relevance of the context. The top N most relevant chunks are selected.
5.  **LLM Response Generation:**
    *   The selected, re-ranked document chunks (the context) and the original user question are passed to a local Large Language Model (`llama3.2:3b` via Ollama).
    *   The LLM generates a comprehensive answer based *only* on the provided context and the user's question.
    *   The answer is streamed back to the user in the chatbot interface.

## Technologies Used
-   **Python 3.8+**
-   **Streamlit:** For creating the web-based user interface.
-   **Langchain (`langchain-community`, `langchain-core`):** For document loading (PyMuPDFLoader) and text splitting (RecursiveCharacterTextSplitter).
-   **ChromaDB (`chromadb`):** For creating and managing the vector store.
-   **Ollama (`ollama`):** For running local Large Language Models (LLMs) and embedding models.
    -   LLM: `llama3.2:3b` (or user-configured)
    -   Embedding Model: `nomic-embed-text:latest`
-   **Sentence Transformers (`sentence-transformers` via `CrossEncoder`):** For re-ranking retrieved documents.
-   **PyMuPDF (`pymupdf`):** For parsing and extracting text from PDF files.

## Prerequisites
-   Python 3.8 or higher.
-   [Ollama](https://ollama.com/) installed and running.
-   Access to a terminal or command prompt.

## Setup and Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ensure Ollama is running:**
    Start your Ollama application. By default, it should be accessible at `http://localhost:11434`. The application code uses this default URL.
5.  **Pull necessary Ollama models:**
    Open your terminal and run the following commands to download the required models:
    ```bash
    ollama pull nomic-embed-text:latest
    ollama pull llama3.2:3b
    ```
    *Note: If you wish to use a different LLM, you can pull it via `ollama pull <model-name>` and update the `model` parameter in `chatbot.py` and `pdfupload.py` within the `call_llm` function.*

## Running the Application

There are two main parts to this application: uploading PDFs to the knowledge base and interacting with the chatbot.

### 1. Uploading PDF Documents
This step populates the vector store with information the chatbot can use.
```bash
streamlit run pdfupload.py
```
-   Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.
-   Use the sidebar to upload PDF files.
-   Click the "Process" button to embed the document(s) and add them to the ChromaDB vector store. You should see a success message.

### 2. Interacting with the Chatbot
Once you have processed some PDF documents, you can start asking questions.
```bash
streamlit run chatbot.py
```
-   Open the URL provided by Streamlit (usually `http://localhost:8501`, though if `pdfupload.py` is still running, it might be a different port like `8502`) in your web browser.
-   The main interface will display a "Vidya Chatbot" header and a text area to ask questions.
-   Type your question related to the content of the uploaded PDFs and click "🔥 Ask".
-   The chatbot will stream its response.
-   You can also use the sidebar links for quick navigation to college-related websites.

## Project Structure
```
.
├── chatbot.py            # Main Streamlit application for the chatbot UI
├── pdfupload.py          # Streamlit application for uploading and processing PDFs
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── demo-rag-chroma/      # Directory where ChromaDB stores its data (created automatically)
└── d500x300.jpg          # Logo image (ensure path is correct or remove if not used)
```
*Note: The `d500x300.jpg` path in `chatbot.py` and `pdfupload.py` is currently an absolute path (`D:\Arohan resources\chatbot\d500x300.jpg`). You might need to adjust this to a relative path (e.g., `./d500x300.jpg`) or remove the image loading if the image is not present in your project directory.*

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or find bugs.

## License
This project is open-source. Please refer to the `LICENSE` file if one is included, otherwise, assume it is provided as-is without a specific license. (Consider adding a `LICENSE` file, e.g., MIT or Apache 2.0).
