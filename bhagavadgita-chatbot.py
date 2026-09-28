import os

import pdfplumber
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.header("Bhagavad Gita Chatbot")

script_dir = os.path.dirname(__file__)
pdf_path = os.path.join(script_dir, "bgita.pdf")


@st.cache_resource
def load_vectorstore():
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found: {pdf_path}")
        return None

    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    if not text.strip():
        st.error("The PDF does not contain extractable text.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="bhagavad_gita",
        persist_directory=os.path.join(script_dir, "chroma_db"),
    )


vectorstore = load_vectorstore()
user_question = st.text_input("Type your question here")

if vectorstore is None:
    st.stop()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4},
)

llm = OllamaLLM(
    model="llama3.2",
    temperature=0.3,
    num_predict=1000,
    base_url="http://localhost:11434",
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions about a PDF document.\n\n"
            "Guidelines:\n"
            "1. Provide complete, well-explained answers using the context below.\n"
            "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
            "3. If the context mentions related information, include it to give fuller picture.\n"
            "4. Only use information from the provided context - do not use outside knowledge.\n"
            "5. Summarize long information, ideally in bullets where needed\n"
            "6. If the information is not in the context, say so politely.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if user_question:
    response = chain.invoke(user_question)
    st.write(response)
