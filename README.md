# Bhagavad Gita Chatbot

A Streamlit-based chatbot that reads a Bhagavad Gita PDF, splits it into chunks, embeds the text using Hugging Face sentence-transformers, stores the vectors in Chroma, and answers questions with an open-source LLM running via Ollama.

## Features

- Upload or use a local PDF file
- Text chunking with LangChain RecursiveCharacterTextSplitter
- Hugging Face embeddings for semantic search
- Chroma vector database for retrieval
- Open-source local LLM via Ollama
- Simple Streamlit UI

## Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face Embeddings
- Chroma
- Ollama
- PDFPlumber

## Project Structure

```text
Chatbot/
├── bhagavadgita-chatbot.py
├── requirements.txt
├── README.md
├── .gitignore
├── bgita.pdf
└── chroma_db/
```

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Install Ollama

Download and install Ollama from: https://ollama.com

Then pull a model:

```bash
ollama pull llama3.2
```

Start the Ollama service:

```bash
ollama serve
```

### 2. Run the app

```bash
streamlit run bhagavadgita-chatbot.py
```

## Notes

- The app expects the PDF file named `bgita.pdf` in the same folder as the script.
- The Chroma database will be created in a local `chroma_db` directory.
- If the PDF is missing or contains no extractable text, the app will show an error message.
- This project uses local open-source tooling and does not depend on Google API keys.

## License

This project is for educational and personal use.
