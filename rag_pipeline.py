"""
RAG Pipeline Module

This module implements the Retrieval-Augmented Generation (RAG) pipeline
using LangChain with local vector embeddings and Hugging Face inference LLM.
"""

import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()


class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) Pipeline
    
    Combines local vector store with Hugging Face LLM for question answering
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the RAG pipeline
        
        Args:
            config (dict): Configuration dictionary with RAG parameters
        """
        self.config = config or self._load_config()
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        
        self._initialize_components()
    
    def _load_config(self) -> dict:
        """Load configuration from environment variables"""
        return {
            'chunk_size': int(os.getenv('CHUNK_SIZE', 1000)),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', 200)),
            'retrieval_k': int(os.getenv('RETRIEVAL_K', 3)),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            'vector_store_path': os.getenv('VECTOR_STORE_PATH', './data/vector_store'),
            'document_path': os.getenv('DOCUMENT_PATH', './data/documents'),
            'llm_temperature': float(os.getenv('LLM_TEMPERATURE', 0.7)),
            'llm_max_tokens': int(os.getenv('LLM_MAX_TOKENS', 512)),
            'hf_model_id': os.getenv('HF_MODEL_ID', 'meta-llama/Llama-2-7b-chat-hf'),
            'hf_api_key': os.getenv('HUGGINGFACE_API_KEY'),
        }
    
    def _initialize_components(self):
        """Initialize embeddings and LLM components"""
        # Initialize embeddings (runs locally)
        print(f"Loading embedding model: {self.config['embedding_model']}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embedding_model']
        )
        
        # Initialize LLM (Hugging Face Inference)
        print(f"Initializing LLM: {self.config['hf_model_id']}")
        self.llm = HuggingFaceHub(
            repo_id=self.config['hf_model_id'],
            model_kwargs={
                "temperature": self.config['llm_temperature'],
                "max_length": self.config['llm_max_tokens'],
            },
            huggingfacehub_api_token=self.config['hf_api_key'],
        )
    
    def load_documents(self) -> List:
        """
        Load documents from the document directory
        
        Returns:
            List of loaded documents
        """
        doc_path = Path(self.config['document_path'])
        
        if not doc_path.exists():
            print(f"Document path {doc_path} does not exist. Creating it...")
            doc_path.mkdir(parents=True, exist_ok=True)
            return []
        
        print(f"Loading documents from {doc_path}")
        loader = DirectoryLoader(str(doc_path), glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        
        return documents
    
    def build_vector_store(self, documents: List = None, force_rebuild: bool = False) -> FAISS:
        """
        Build or load vector store from documents
        
        Args:
            documents (List): List of documents to process
            force_rebuild (bool): Force rebuild vector store from documents
        
        Returns:
            FAISS vector store
        """
        vector_store_path = Path(self.config['vector_store_path'])
        
        # Load existing vector store if available
        if vector_store_path.exists() and not force_rebuild:
            print(f"Loading vector store from {vector_store_path}")
            self.vector_store = FAISS.load_local(
                str(vector_store_path),
                self.embeddings
            )
            return self.vector_store
        
        # Build new vector store
        if documents is None:
            documents = self.load_documents()
        
        if not documents:
            print("No documents found. Cannot build vector store.")
            return None
        
        # Split documents into chunks
        print(f"Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
        )
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} document chunks")
        
        # Create vector store
        print("Creating vector store embeddings...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        # Save vector store
        vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(vector_store_path))
        print(f"Vector store saved to {vector_store_path}")
        
        return self.vector_store
    
    def create_qa_chain(self, vector_store: FAISS = None):
        """
        Create QA chain from vector store and LLM
        
        Args:
            vector_store (FAISS): Vector store for retrieval
        """
        if vector_store is None:
            vector_store = self.vector_store
        
        if vector_store is None:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_kwargs={'k': self.config['retrieval_k']}
        )
        
        # Create custom prompt
        template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
        
        PROMPT = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("QA chain created successfully")
        return self.qa_chain
    
    def query(self, question: str) -> Tuple[str, List]:
        """
        Query the RAG pipeline
        
        Args:
            question (str): The question to answer
        
        Returns:
            Tuple of (answer, source_documents)
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain() first.")
        
        print(f"\nQuery: {question}")
        result = self.qa_chain({"query": question})
        
        answer = result.get('result', 'No answer found')
        sources = result.get('source_documents', [])
        
        return answer, sources
    
    def setup_pipeline(self, force_rebuild: bool = False) -> RetrievalQA:
        """
        Complete setup of RAG pipeline
        
        Args:
            force_rebuild (bool): Force rebuild vector store
        
        Returns:
            RetrievalQA chain
        """
        print("\n" + "="*50)
        print("Setting up RAG Pipeline")
        print("="*50)
        
        # Load and process documents
        documents = self.load_documents()
        
        # Build vector store
        self.build_vector_store(documents, force_rebuild=force_rebuild)
        
        # Create QA chain
        self.create_qa_chain()
        
        print("="*50)
        print("RAG Pipeline Ready!")
        print("="*50 + "\n")
        
        return self.qa_chain
