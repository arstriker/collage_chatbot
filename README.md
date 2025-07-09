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
-   **Python:** Version 3.8+
-   **Streamlit:** For building the interactive web user interface.
-   **Langchain (`langchain-community`, `langchain-core`):** Used for document loading (`PyMuPDFLoader`) and text splitting (`RecursiveCharacterTextSplitter`).
-   **ChromaDB (`chromadb`):** Serves as the vector store for creating and managing document embeddings.
-   **Ollama (`ollama`):** Enables running local Large Language Models (LLMs) and embedding models.
    -   **LLM:** Configured with `llama3.2:3b` (can be changed by the user).
    -   **Embedding Model:** Uses `nomic-embed-text:latest`.
-   **Sentence Transformers (`sentence-transformers`):** The `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is used for re-ranking retrieved document chunks to improve context relevance.
-   **PyMuPDF (`pymupdf`):** For parsing text and metadata from PDF files.

## Prerequisites
-   Python 3.8 or a newer version.
-   [Ollama](https://ollama.com/) installed and actively running on your system.
-   A terminal or command prompt for executing commands.

## Setup and Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    *On Windows:*
    ```bash
    venv\Scripts\activate
    ```
    *On macOS/Linux:*
    ```bash
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    Navigate to the project directory in your terminal (if you aren't already there) and run:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ensure Ollama is Running:**
    Start your Ollama desktop application or ensure the Ollama service is running. The Python application connects to Ollama using its default API endpoint (`http://localhost:11434`).
5.  **Download Ollama Models:**
    If you haven't already, pull the necessary models using the following commands in your terminal:
    ```bash
    ollama pull nomic-embed-text:latest
    ```
    ```bash
    ollama pull llama3.2:3b
    ```
    *Note: If you wish to use a different LLM, you can pull it via `ollama pull <model-name>` and update the `model_name` parameter in `chatbot.py` and `pdfupload.py` (specifically in the `call_llm` function and `OllamaEmbeddingFunction` initialization if the embedding model changes).*

## Running the Application

The application consists of two main Streamlit interfaces: one for uploading and processing PDF documents, and another for interacting with the chatbot.

### 1. Uploading PDF Documents (`pdfupload.py`)
This script allows you to add documents to the chatbot's knowledge base.
-   **Navigate to the project directory** in your terminal.
-   **Run the PDF upload application:**
    ```bash
    streamlit run pdfupload.py
    ```
-   Streamlit will typically open the application in your default web browser (e.g., at `http://localhost:8501`).
-   In the web interface, use the sidebar to **upload your PDF file(s)**.
-   Click the **"⚡️ Process"** button. This will:
    -   Load the PDF content.
    -   Split it into manageable chunks.
    -   Generate embeddings for each chunk.
    -   Store these embeddings in the ChromaDB vector store (`./demo-rag-chroma` directory).
-   A success message will appear once processing is complete.

### 2. Interacting with the Chatbot (`chatbot.py`)
After populating the knowledge base, you can start asking questions.
-   **Navigate to the project directory** in your terminal.
-   **Run the chatbot application:**
    ```bash
    streamlit run chatbot.py
    ```
-   This will open a new Streamlit interface in your browser (if `pdfupload.py` is still running, this might be on a different port, e.g., `http://localhost:8502`).
-   The interface features:
    -   A **logo** and the title "Vidya Chatbot".
    -   A **text area** labeled "**Ask a question related to College:**".
    -   An "**🔥 Ask**" button.
    -   A **sidebar** with useful links (e.g., Vidya Website, document guidelines).
-   Type your question into the text area and click "🔥 Ask".
-   The chatbot will retrieve relevant information from the processed PDFs, generate an answer using the local LLM, and stream the response back to you.
-   You can expand sections to see the raw retrieved documents and the most relevant document IDs.

## Project Structure
```plaintext
.
├── chatbot.py            # Main Streamlit application for the chatbot UI
├── pdfupload.py          # Streamlit application for uploading and processing PDFs
├── requirements.txt      # Python dependencies for the project
├── README.md             # This documentation file
├── demo-rag-chroma/      # Directory where ChromaDB stores its vector data (auto-generated)
└── d500x300.jpg          # Logo image displayed in the UI
```
**Important Note on Image Path:**
The scripts `chatbot.py` and `pdfupload.py` currently use an absolute path for the logo image: `D:\Arohan resources\chatbot\d500x300.jpg`.
For the application to work correctly on other systems or if you move the project, you should:
1.  Place the `d500x300.jpg` file in the root of the project directory.
2.  Change the `logo_path` variable in both `chatbot.py` and `pdfupload.py` to a relative path:
    ```python
    logo_path = "d500x300.jpg"
    ```
    Or, if you prefer to place it in an `assets` subfolder:
    ```python
    logo_path = "assets/d500x300.jpg"
    ```
    (And ensure the image is in `assets/d500x300.jpg`).

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.
