# 🧠 SmartRAG — FAISS-Powered Knowledge Assistant

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0078D4?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-MiniLM-FFD21F?style=flat-square&logo=huggingface)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF_Parsing-blue?style=flat-square)

> Upload any PDF — ask questions — get answers. A **RAG pipeline built from scratch** using FAISS vector search and sentence embeddings, with a plug-in ready slot for any LLM.

---

## 🧠 About the Project

SmartRAG is a **Retrieval-Augmented Generation (RAG)** system that turns any PDF into a queryable knowledge base. It extracts text, chunks it, embeds it using a transformer model, indexes it with FAISS, and retrieves the most semantically relevant content for any user query.

The architecture is **LLM-agnostic** — the retrieved context can be piped into OpenAI GPT, HuggingFace models, or any other LLM for final answer generation.

---

## ✨ Features

- 📄 **PDF Upload** — Upload any PDF and instantly make it searchable
- ✂️ **Smart Chunking** — Text split into 500-word chunks for optimal retrieval
- 🔢 **Dense Embeddings** — `all-MiniLM-L6-v2` encodes chunks semantically
- ⚡ **FAISS Similarity Search** — Top-5 most relevant chunks retrieved instantly
- 🔌 **LLM-Agnostic** — Plug in OpenAI, Flan-T5, or any model for answer generation
- 🖥️ **Clean Streamlit UI** — Upload, query, and read results in one page

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.8+ |
| PDF Parsing | PyMuPDF (fitz) |
| Embeddings | SentenceTransformer (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (IndexFlatL2) |
| Frontend | Streamlit |
| Numerics | NumPy |

---

## 📁 Project Structure

```
SmartRAG_Faiss/
│
├── app.py               # Full RAG pipeline + Streamlit UI
└── requirements.txt     # Project dependencies
```

---

## ⚙️ How It Works — RAG Pipeline

```
User Uploads PDF
      ↓
PyMuPDF extracts full text
      ↓
Text split into 500-word chunks
      ↓
SentenceTransformer embeds all chunks
      ↓
FAISS IndexFlatL2 stores embeddings
      ↓
User types a query
      ↓
Query embedded → FAISS Top-5 search
      ↓
Retrieved chunks shown as context
      ↓
[Plug in LLM here for final answer]
```

1. **PDF Parsing** — PyMuPDF reads every page and extracts raw text
2. **Chunking** — Text is split into 500-word non-overlapping chunks
3. **Embedding** — Each chunk is encoded into a 384-dim vector via `all-MiniLM-L6-v2`
4. **Indexing** — FAISS stores all vectors for fast L2 similarity search
5. **Retrieval** — On a query, the top-5 nearest chunks are returned as context
6. **Generation** — Context is ready to pass into any LLM for answer generation

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/SmartRAG_Faiss.git
cd SmartRAG_Faiss

# 2. Install dependencies
pip install streamlit pymupdf faiss-cpu sentence-transformers numpy

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🖥️ Usage

1. Launch the app with `streamlit run app.py`
2. Click **"Browse files"** and upload any PDF (research paper, report, book chapter)
3. Wait for the document to be indexed — you'll see a success message
4. Type a natural language question in the text input
5. SmartRAG retrieves the most relevant chunks from your document

---

## 🔌 Adding an LLM for Answer Generation

The retrieval pipeline is complete. To add answer generation, replace the placeholder in `app.py` with any of these:

**Option A — OpenAI GPT**
```python
import openai
openai.api_key = "your-key"
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": f"Context: {context}\nQuestion: {query}"}]
)
st.write(response.choices[0].message.content)
```

**Option B — HuggingFace Flan-T5**
```python
from transformers import pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-base")
answer = generator(f"Context: {context}\nQuestion: {query}", max_length=200)
st.write(answer[0]["generated_text"])
```

---

## 💡 Example Use Cases

| PDF Type | Sample Query |
|----------|-------------|
| Research Paper | *"What methodology was used in this study?"* |
| Legal Document | *"What are the termination clauses?"* |
| Textbook Chapter | *"Explain the key concepts of supervised learning"* |
| Financial Report | *"What was the revenue in Q3?"* |

---

## 📦 Dependencies

```
streamlit
pymupdf
faiss-cpu
sentence-transformers
numpy
```

---

## 🙋‍♂️ Author

**Your Name**
- GitHub: [@Malka23](https://github.com/Malka23)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/malka-naaz-870338145)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project helpful, please give it a star!**
