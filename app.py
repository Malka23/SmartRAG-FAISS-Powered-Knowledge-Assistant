# smartrag_app.py

import streamlit as st
import fitz  # PyMuPDF
import faiss
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  # small & fast

model = load_model()

# PDF Text Extractor
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split text into chunks
def split_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Embed & build FAISS index
def build_faiss_index(text_chunks):
    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Retrieve relevant chunks
def retrieve_chunks(query, index, text_chunks, embeddings, k=5):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k)
    return [text_chunks[i] for i in I[0]]

# Streamlit UI
def main():
    st.title("🧠 SmartRAG: FAISS-Powered Knowledge Assistant")
    uploaded_file = st.file_uploader("📄 Upload a PDF document", type=["pdf"])

    if uploaded_file:
        with st.spinner("Reading and indexing document..."):
            full_text = extract_text_from_pdf(uploaded_file)
            text_chunks = split_text(full_text)
            index, embeddings = build_faiss_index(text_chunks)
            st.success("Document processed and indexed!")

            query = st.text_input("🔎 Ask a question based on the document:")
            if query:
                relevant_chunks = retrieve_chunks(query, index, text_chunks, embeddings)
                context = "\n".join(relevant_chunks)
                st.subheader("📚 Retrieved Context:")
                st.write(context)

                # You can plug in OpenAI/LLM here to generate final answers
                st.subheader("🧠 Suggested Answer:")
                st.write("Use the above context with a language model for generating answers.")

if __name__ == "__main__":
    main()
