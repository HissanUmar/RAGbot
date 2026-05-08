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
from huggingface_hub import InferenceClient

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
        self.llm_init_error = None
        self.qa_chain = None
        self.embedding_backend = "self-hosted-local"
        self.embedding_init_error = None
        self.active_llm_model = None

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
            "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", 1024)),
            "hf_model_id": os.getenv("HF_MODEL_ID", "google/flan-t5-base"),
            "hf_api_key": os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HF_TOKEN"),
        }

    def _initialize_components(self):
        try:
            print(f"Loading embedding model: {self.config['embedding_model']}")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.config["embedding_model"])
            self.embedding_backend = "self-hosted-local"
            self.embedding_init_error = None
        except Exception as exc:
            self.embedding_backend = "fallback-error"
            self.embedding_init_error = str(exc)
            raise RuntimeError(
                f"Embedding model failed to load: {exc}. "
                "This deployment is configured to use Hugging Face embeddings only."
            ) from exc
        self.llm = self._initialize_llm() if self.enable_remote_llm else None

    def _initialize_llm(self):
        token = (
            self.config.get("hf_api_key")
            or os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HF_TOKEN")
        )
        if not token:
            self.llm_init_error = "HUGGINGFACE_API_KEY is not set"
            print("HF API key not set; running in retrieval-only mode.")
            return None

        # persist token back into config so downstream code sees it
        self.config["hf_api_key"] = token

        requested_model = (self.config.get("hf_model_id") or "").strip()
        configured_candidates = [
            model.strip()
            for model in self.config.get("hf_model_candidates", [])
            if isinstance(model, str) and model.strip()
        ]

        # Candidate models to try, with the user-selected model first.
        env_list = os.getenv("HF_MODEL_TRY")
        env_candidates = [m.strip() for m in env_list.split(",") if m.strip()] if env_list else []

        public_fallbacks = ["google/flan-t5-base", "bigscience/bloom", "gpt2"]

        candidates: list[str] = []
        for model_id in [requested_model, *configured_candidates, *env_candidates, *public_fallbacks]:
            if model_id and model_id not in candidates:
                candidates.append(model_id)

        errors = {}
        for repo_id in candidates:
            try:
                print(f"Attempting LLM init with model: {repo_id}")
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
                os.environ["HUGGINGFACE_API_KEY"] = token
                client = self._create_hf_llm_client(repo_id, token)
                print(f"Initialized LLM with {repo_id}")
                # persist chosen model id
                self.config["hf_model_id"] = repo_id
                self.active_llm_model = repo_id
                return client
            except Exception as exc:
                errors[repo_id] = str(exc)
                print(f"Model {repo_id} failed: {exc}")
                continue

        # If we get here, all candidates failed
        combined = " | ".join([f"{k}: {v}" for k, v in errors.items()])
        self.llm_init_error = combined
        print("LLM initialization failed for all candidates:", combined)
        return None

    def _create_hf_llm_client(self, repo_id: str, token: str):
        """Create a simple callable wrapper around Hugging Face InferenceClient."""
        model_id, provider = self._split_provider(repo_id)
        client = InferenceClient(model=model_id, token=token, provider=provider)

        class _HFLLMWrapper:
            def __init__(self, inference_client: InferenceClient, model_name: str, temperature: float, max_new_tokens: int):
                self.client = inference_client
                self.model_name = model_name
                self.temperature = temperature
                self.max_new_tokens = max_new_tokens

            def invoke(self, prompt: str) -> str:
                # Prefer chat completion for instruction-tuned models.
                try:
                    response = self.client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                    )
                    choice = response.choices[0]
                    message = getattr(choice, "message", None)
                    if message and getattr(message, "content", None):
                        return message.content
                    if getattr(choice, "text", None):
                        return choice.text
                except Exception:
                    pass

                # Fall back to text generation.
                response = self.client.text_generation(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    return_full_text=False,
                )
                return response if isinstance(response, str) else str(response)

        return _HFLLMWrapper(
            client,
            model_id,
            self.config["llm_temperature"],
            self.config["llm_max_tokens"],
        )

    @staticmethod
    def _split_provider(repo_id: str) -> Tuple[str, Optional[str]]:
        if ":" in repo_id:
            model_id, provider = repo_id.rsplit(":", 1)
            return model_id, provider
        return repo_id, None

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
