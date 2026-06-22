# smartrag_app.py

import streamlit as st
import fitz  # PyMuPDF
import faiss
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load Flan-T5 for answer generation
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

model = load_model()
generator = load_generator()

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
def retrieve_chunks(query, index, text_chunks, k=5):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k)
    return [text_chunks[i] for i in I[0]]

# Generate answer using Flan-T5
def generate_answer(context, query):
    prompt = f"Answer the question based on the context below.\n\nContext: {context[:1500]}\n\nQuestion: {query}\n\nAnswer:"
    result = generator(prompt, max_length=256, do_sample=False)
    return result[0]["generated_text"]

# Streamlit UI
def main():
    st.set_page_config(page_title="SmartRAG", page_icon="🧠", layout="centered")
    st.title("🧠 SmartRAG: FAISS-Powered Knowledge Assistant")
    st.caption("Upload any PDF — ask questions — get answers.")

    uploaded_file = st.file_uploader("📄 Upload a PDF document", type=["pdf"])

    if uploaded_file:
        with st.spinner("Reading and indexing document..."):
            full_text = extract_text_from_pdf(uploaded_file)
            text_chunks = split_text(full_text)
            index, embeddings = build_faiss_index(text_chunks)
        st.success(f"Document indexed — {len(text_chunks)} chunks ready.")

        query = st.text_input("🔎 Ask a question based on the document:")
        if query:
            with st.spinner("Retrieving context and generating answer..."):
                relevant_chunks = retrieve_chunks(query, index, text_chunks)
                context = "\n".join(relevant_chunks)
                answer = generate_answer(context, query)

            st.subheader("🧠 Answer")
            st.write(answer)

            with st.expander("📚 View Retrieved Context"):
                st.write(context)

if __name__ == "__main__":
    main()
