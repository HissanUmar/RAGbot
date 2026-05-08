"""Free deployment entry point for the retrieval-only RAG app.

This Streamlit app deploys the local embedding model + FAISS retrieval path.
The Hugging Face LLM is optional and not required for deployment.
"""

from __future__ import annotations

from pathlib import Path
import os

import streamlit as st

from rag_pipeline import RAGPipeline


st.set_page_config(page_title="RAG Retriever", page_icon="🔎", layout="wide")

st.title("RAG Retriever")
st.caption("Local embeddings + FAISS, deployable for free. LLM is optional.")


def ensure_demo_documents(document_dir: Path) -> int:
    """Create starter documents when the app is deployed with an empty folder."""
    document_dir.mkdir(parents=True, exist_ok=True)
    existing_documents = list(document_dir.glob("*.txt"))
    if existing_documents:
        return len(existing_documents)

    demo_documents = {
        "python.txt": """Python is a high-level, interpreted programming language known for simplicity and readability.
Python is commonly used for web development, data analysis, automation, and artificial intelligence.
""",
        "machine_learning.txt": """Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Common techniques include supervised learning, unsupervised learning, and reinforcement learning.
""",
        "rag.txt": """Retrieval-Augmented Generation combines retrieval with language generation.
It improves answers by grounding them in relevant source documents.
""",
    }

    for filename, content in demo_documents.items():
        (document_dir / filename).write_text(content, encoding="utf-8")

    return len(demo_documents)


def save_uploaded_documents(uploaded_files, document_dir: Path) -> int:
    """Persist uploaded text files to the document directory."""
    saved_count = 0
    document_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".txt"):
            continue

        target_path = document_dir / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_count += 1

    return saved_count


@st.cache_resource(show_spinner=True)
def load_pipeline(enable_remote_llm: bool) -> RAGPipeline:
    pipeline = RAGPipeline(enable_remote_llm=enable_remote_llm)
    pipeline.setup_pipeline(force_rebuild=False)
    return pipeline


with st.sidebar:
    st.header("Deployment Mode")
    st.write("This app runs the embedding model locally and serves FAISS retrieval.")
    st.info("No Hugging Face model hosting required.")
    top_k = st.slider("Top K results", min_value=1, max_value=8, value=3)
    rebuild = st.button("Rebuild vector store")

    # LLM enable toggle — default on when HUGGINGFACE_API_KEY exists
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    default_enable_llm = bool(hf_key)
    enable_llm = st.checkbox(
        "Enable remote LLM (Hugging Face)",
        value=default_enable_llm,
        help="Requires HUGGINGFACE_API_KEY set in environment or Colab Secrets",
    )
    if enable_llm and not hf_key:
        st.warning("HUGGINGFACE_API_KEY not found — set it in Colab Secrets or as an env var.")

    st.divider()
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Add .txt files",
        type=["txt"],
        accept_multiple_files=True,
        help="Upload plain text documents to index with FAISS.",
    )
    upload_clicked = st.button("Save uploads")


# Ensure demo documents exist before initializing the pipeline so
# the vector store can be built on first run.
default_doc_path = os.getenv("DOCUMENT_PATH", "./data/documents")
document_dir = Path(default_doc_path)
document_count = ensure_demo_documents(document_dir)

# Use the sidebar toggle to decide whether to enable the remote LLM
pipeline = load_pipeline(enable_llm)
# Update document_dir to the pipeline's configured path (if different)
document_dir = Path(pipeline.config["document_path"])
document_count = ensure_demo_documents(document_dir)

if upload_clicked:
    if not uploaded_files:
        st.warning("Choose one or more .txt files first.")
    else:
        saved_count = save_uploaded_documents(uploaded_files, document_dir)
        if saved_count == 0:
            st.warning("Only .txt files are accepted.")
        else:
            st.success(f"Saved {saved_count} uploaded file(s). Rebuild the vector store next.")
            st.rerun()

if rebuild or pipeline.vector_store is None:
    action_label = "Rebuilding" if rebuild else "Loading"
    with st.spinner(f"{action_label} vector store..."):
        pipeline.setup_pipeline(force_rebuild=rebuild)
    if rebuild:
        st.success("Vector store rebuilt.")

question = st.text_input("Ask a question about your documents")

if st.button("Search"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        pipeline.config["retrieval_k"] = top_k
        answer, sources = pipeline.query(question.strip())

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Answer / Retrieval Result")
            st.write(answer)

        with col2:
            st.subheader("Sources")
            st.write(f"{len(sources)} document(s) returned")
            for index, document in enumerate(sources, 1):
                filename = document.metadata.get("filename", f"doc_{index}")
                st.write(f"{index}. {filename}")
                st.caption(document.page_content[:180].replace("\n", " ") + "...")


st.divider()
st.subheader("Project Files")
st.write("Vector store:", Path(pipeline.config["vector_store_path"]).resolve())
st.write("Documents:", Path(pipeline.config["document_path"]).resolve())
st.write("Document count:", document_count)