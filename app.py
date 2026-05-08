"""Free deployment entry point for the retrieval-only RAG app.

This Streamlit app deploys the local embedding model + FAISS retrieval path.
The Hugging Face LLM is optional and not required for deployment.
"""

from __future__ import annotations

from pathlib import Path
import os

import streamlit as st
import requests

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


def resolve_secret_value(name: str) -> str | None:
    """Resolve a secret from Streamlit secrets or environment variables."""
    value = os.getenv(name)
    if value:
        return value

    try:
        if name in st.secrets:
            value = st.secrets[name]
            if value:
                return str(value)

        for alternate_name in ("HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"):
            if alternate_name in st.secrets:
                value = st.secrets[alternate_name]
                if value:
                    return str(value)
    except Exception:
        pass

    return None


@st.cache_resource(show_spinner=True)
def load_pipeline(enable_remote_llm: bool, hf_api_key: str | None, hf_model_id: str, llm_max_tokens: int) -> RAGPipeline:
    config = {
        "chunk_size": int(os.getenv("CHUNK_SIZE", 1000)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 200)),
        "retrieval_k": int(os.getenv("RETRIEVAL_K", 3)),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "vector_store_path": os.getenv("VECTOR_STORE_PATH", "./data/vector_store"),
        "document_path": os.getenv("DOCUMENT_PATH", "./data/documents"),
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", 0.7)),
        "llm_max_tokens": llm_max_tokens,
        "hf_model_id": hf_model_id,
        "hf_api_key": hf_api_key,
    }

    if hf_api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
        os.environ["HUGGINGFACE_API_KEY"] = hf_api_key

    pipeline = RAGPipeline(config=config, enable_remote_llm=enable_remote_llm)
    pipeline.setup_pipeline(force_rebuild=False)
    return pipeline


with st.sidebar:
    st.header("Deployment Mode")
    st.write("This app runs the embedding model locally and serves FAISS retrieval.")
    st.info("No Hugging Face model hosting required.")
    top_k = st.slider("Top K results", min_value=1, max_value=8, value=3)
    rebuild = st.button("Rebuild vector store")

    # LLM enable toggle — default on when HUGGINGFACE_API_KEY exists
    hf_key = resolve_secret_value("HUGGINGFACE_API_KEY") or resolve_secret_value("HUGGINGFACEHUB_API_TOKEN") or resolve_secret_value("HF_TOKEN")
    default_enable_llm = bool(hf_key)
    enable_llm = st.checkbox(
        "Enable remote LLM (Hugging Face)",
        value=default_enable_llm,
        help="Requires a Hugging Face token in Streamlit Cloud Secrets or environment variables",
    )
    if enable_llm and not hf_key:
        st.warning("Add your Hugging Face token to Streamlit Cloud Secrets or environment variables.")
    hf_model_id = resolve_secret_value("HF_MODEL_ID") or os.getenv("HF_MODEL_ID", "google/flan-t5-base")
    llm_max_tokens = st.slider("LLM max tokens", min_value=128, max_value=2048, value=int(os.getenv("LLM_MAX_TOKENS", 1024)), step=64)

    # Diagnostic: test HF token access to a list of models
    def test_hf_models(token: str, candidates: list[str]) -> dict:
        results = {}
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        for repo_id in candidates:
            repo = repo_id.split(":", 1)[0]
            url = f"https://huggingface.co/api/models/{repo}"
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                results[repo_id] = (resp.status_code, resp.reason)
            except Exception as exc:
                results[repo_id] = (None, str(exc))
        return results

    st.divider()
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Add .txt files",
        type=["txt"],
        accept_multiple_files=True,
        help="Upload plain text documents to index with FAISS.",
    )
    upload_clicked = st.button("Save uploads")

    # Run diagnostics button
    if st.button("Run Hugging Face diagnostics"):
        with st.spinner("Testing Hugging Face token and model access..."):
            hf_token = resolve_secret_value("HUGGINGFACE_API_KEY") or resolve_secret_value("HUGGINGFACEHUB_API_TOKEN") or resolve_secret_value("HF_TOKEN")
            candidates = [
                "deepseek-ai/DeepSeek-V4-Pro:novita",
                "meta-llama/Llama-3.1-8B-Instruct:cerebras",
                "Qwen/Qwen3.5-9B:ovhcloud",
                "Qwen/Qwen2.5-7B-Instruct:together",
                "meta-llama/Llama-4-Maverick-17B-128E-Instruct:sambanova",
                "google/flan-t5-base",
                "bigscience/bloom",
                "gpt2",
            ]
            results = test_hf_models(hf_token, candidates)
        st.subheader("Hugging Face diagnostics")
        for repo_id, (code, msg) in results.items():
            status = f"{code} {msg}" if code else f"Error: {msg}"
            st.write(f"- **{repo_id}**: {status}")


# Ensure demo documents exist before initializing the pipeline so
# the vector store can be built on first run.
default_doc_path = os.getenv("DOCUMENT_PATH", "./data/documents")
document_dir = Path(default_doc_path)
document_count = ensure_demo_documents(document_dir)

# Use the sidebar toggle to decide whether to enable the remote LLM
pipeline = load_pipeline(enable_llm, hf_key, hf_model_id, llm_max_tokens)
# Update document_dir to the pipeline's configured path (if different)
document_dir = Path(pipeline.config["document_path"])
document_count = ensure_demo_documents(document_dir)

st.sidebar.divider()
st.sidebar.subheader("Runtime Status")
st.sidebar.write(f"Embedding backend: **{pipeline.embedding_backend}**")
st.sidebar.write(f"Embedding model: `{pipeline.config['embedding_model']}`")
st.sidebar.write(f"LLM model: `{getattr(pipeline, 'active_llm_model', None) or pipeline.config.get('hf_model_id', 'disabled')}`")
st.sidebar.write(f"LLM max tokens: `{pipeline.config['llm_max_tokens']}`")
st.sidebar.write(f"LLM temperature: `{pipeline.config['llm_temperature']}`")
if getattr(pipeline, "embedding_init_error", None):
    st.sidebar.error(f"Embedding init error: {pipeline.embedding_init_error}")

if enable_llm and pipeline.llm is None and getattr(pipeline, "llm_init_error", None):
    st.warning(f"Remote LLM could not initialize: {pipeline.llm_init_error}")

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