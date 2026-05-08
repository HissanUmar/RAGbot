"""
RAG Pipeline Module

This module implements a deploy-friendly retrieval pipeline with local
embeddings and FAISS. The Hugging Face LLM is optional so the app can be
deployed for free without hosting a model.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks without extra dependencies."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


class RAGPipeline:
    """Retrieval-Augmented pipeline with optional remote LLM generation."""

    def __init__(self, config: dict = None, enable_remote_llm: bool = False):
        self.config = config or self._load_config()
        self.enable_remote_llm = enable_remote_llm
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None

        self._initialize_components()

    def _load_config(self) -> dict:
        return {
            "chunk_size": int(os.getenv("CHUNK_SIZE", 1000)),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 200)),
            "retrieval_k": int(os.getenv("RETRIEVAL_K", 3)),
            "embedding_model": os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "vector_store_path": os.getenv("VECTOR_STORE_PATH", "./data/vector_store"),
            "document_path": os.getenv("DOCUMENT_PATH", "./data/documents"),
            "llm_temperature": float(os.getenv("LLM_TEMPERATURE", 0.7)),
            "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", 512)),
            "hf_model_id": os.getenv("HF_MODEL_ID", "google/flan-t5-base"),
            "hf_api_key": os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HF_TOKEN"),
        }

    def _initialize_components(self):
        print(f"Loading embedding model: {self.config['embedding_model']}")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config["embedding_model"])
        self.llm = self._initialize_llm() if self.enable_remote_llm else None

    def _initialize_llm(self):
        if not self.config.get("hf_api_key"):
            print("HF API key not set; running in retrieval-only mode.")
            return None

        try:
            from langchain_community.llms import HuggingFaceHub

            print(f"Initializing LLM: {self.config['hf_model_id']}")
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.config["hf_api_key"]
            os.environ["HUGGINGFACE_API_KEY"] = self.config["hf_api_key"]
            return HuggingFaceHub(
                repo_id=self.config["hf_model_id"],
                model_kwargs={
                    "temperature": self.config["llm_temperature"],
                    "max_new_tokens": self.config["llm_max_tokens"],
                },
                huggingfacehub_api_token=self.config["hf_api_key"],
            )
        except Exception as exc:
            print(f"LLM initialization failed: {exc}")
            return None

    def load_documents(self) -> List[Document]:
        doc_path = Path(self.config["document_path"])
        if not doc_path.exists():
            doc_path.mkdir(parents=True, exist_ok=True)
            return []

        documents: List[Document] = []
        for file_path in sorted(doc_path.rglob("*.txt")):
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file_path.read_text(encoding="latin-1")

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "parent_directory": str(file_path.parent),
                    },
                )
            )

        print(f"Loaded {len(documents)} documents from {doc_path}")
        return documents

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        for document in documents:
            for index, chunk_text in enumerate(
                _chunk_text(
                    document.page_content,
                    self.config["chunk_size"],
                    self.config["chunk_overlap"],
                )
            ):
                metadata = dict(document.metadata)
                metadata.update({"chunk_index": index})
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
        return chunks

    def build_vector_store(self, documents: List[Document] = None, force_rebuild: bool = False) -> Optional[FAISS]:
        vector_store_path = Path(self.config["vector_store_path"])

        if vector_store_path.exists() and not force_rebuild:
            try:
                self.vector_store = FAISS.load_local(
                    str(vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"Loaded vector store from {vector_store_path}")
                return self.vector_store
            except Exception as exc:
                print(f"Could not load existing vector store: {exc}. Rebuilding.")

        if documents is None:
            documents = self.load_documents()

        if not documents:
            print("No documents found. Cannot build vector store.")
            return None

        print("Splitting documents into chunks...")
        splits = self._chunk_documents(documents)
        print(f"Created {len(splits)} document chunks")

        print("Creating vector store embeddings...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)

        vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(vector_store_path))
        print(f"Vector store saved to {vector_store_path}")

        return self.vector_store

    def create_qa_chain(self, vector_store: FAISS = None):
        if vector_store is None:
            vector_store = self.vector_store

        if vector_store is None:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")

        self.qa_chain = self._answer_with_retrieval
        return self.qa_chain

    def _answer_with_retrieval(self, question: str) -> Tuple[str, List[Document]]:
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")

        sources = self.vector_store.similarity_search(question, k=self.config["retrieval_k"])
        context = "\n\n".join(doc.page_content for doc in sources)

        if self.llm is not None:
            prompt = (
                "Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            try:
                answer = self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
            except Exception as exc:
                answer = f"Error generating LLM answer: {exc}"
        else:
            answer_lines = ["Retrieved context (LLM disabled):"]
            for index, document in enumerate(sources, 1):
                snippet = document.page_content[:220].replace("\n", " ")
                answer_lines.append(f"{index}. {document.metadata.get('filename', 'Unknown')}: {snippet}...")
            answer = "\n".join(answer_lines)

        return answer, sources

    def query(self, question: str) -> Tuple[str, List[Document]]:
        if self.qa_chain is None:
            self.create_qa_chain()

        print(f"\nQuery: {question}")
        return self.qa_chain(question)

    def setup_pipeline(self, force_rebuild: bool = False):
        print("\n" + "=" * 50)
        print("Setting up RAG Pipeline")
        print("=" * 50)

        documents = self.load_documents()
        self.build_vector_store(documents, force_rebuild=force_rebuild)
        self.create_qa_chain()

        print("=" * 50)
        print("RAG Pipeline Ready!")
        print("=" * 50 + "\n")

        return self.qa_chain
