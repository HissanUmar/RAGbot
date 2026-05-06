"""
Advanced Examples for RAG Pipeline

This module demonstrates advanced usage patterns and configurations.
"""

from rag_pipeline import RAGPipeline
from utils import DocumentManager, VectorStoreManager
from pathlib import Path


def example_1_basic_rag():
    """Example 1: Basic RAG setup and query"""
    print("\n" + "="*60)
    print("Example 1: Basic RAG Setup and Query")
    print("="*60)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Setup with documents
    qa_chain = pipeline.setup_pipeline()
    
    # Single query
    question = "What is Python?"
    answer, sources = pipeline.query(question)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"\nSource Documents:")
    for i, doc in enumerate(sources, 1):
        print(f"  {i}. {doc.metadata.get('source', 'Unknown')}")


def example_2_batch_queries():
    """Example 2: Process multiple queries"""
    print("\n" + "="*60)
    print("Example 2: Batch Query Processing")
    print("="*60)
    
    pipeline = RAGPipeline()
    pipeline.setup_pipeline()
    
    queries = [
        "What is Python?",
        "Name three applications of Python",
        "What is machine learning?",
        "Explain supervised learning",
    ]
    
    results = []
    
    for query in queries:
        try:
            answer, sources = pipeline.query(query)
            results.append({
                'query': query,
                'answer': answer,
                'sources_count': len(sources)
            })
            print(f"\n✓ Processed: {query}")
        except Exception as e:
            print(f"\n✗ Error with query '{query}': {e}")
    
    # Summary
    print(f"\n\nProcessed {len(results)} queries successfully")


def example_3_custom_config():
    """Example 3: Custom configuration"""
    print("\n" + "="*60)
    print("Example 3: Custom Configuration")
    print("="*60)
    
    custom_config = {
        'chunk_size': 500,           # Smaller chunks
        'chunk_overlap': 100,
        'retrieval_k': 5,            # More context
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'llm_temperature': 0.3,      # More deterministic
        'llm_max_tokens': 256,
        'vector_store_path': './data/vector_store',
        'document_path': './data/documents',
        'hf_model_id': 'meta-llama/Llama-2-7b-chat-hf',
        'hf_api_key': None,  # Will be read from environment
    }
    
    pipeline = RAGPipeline(config=custom_config)
    
    print(f"Custom Configuration:")
    print(f"  - Chunk Size: {custom_config['chunk_size']}")
    print(f"  - Retrieval K: {custom_config['retrieval_k']}")
    print(f"  - Temperature: {custom_config['llm_temperature']}")
    
    pipeline.setup_pipeline()
    answer, sources = pipeline.query("What is Machine Learning?")
    print(f"\nAnswer: {answer}")


def example_4_document_management():
    """Example 4: Manage documents"""
    print("\n" + "="*60)
    print("Example 4: Document Management")
    print("="*60)
    
    # List current documents
    docs = DocumentManager.list_documents()
    print(f"Current documents: {len(docs)}")
    for doc in docs:
        print(f"  - {doc}")
    
    # Add a new document
    new_doc_content = """
    Transformers are a type of neural network architecture introduced in the "Attention is All You Need" paper.
    They use self-attention mechanisms to process input sequences in parallel.
    Transformers have become the foundation for modern large language models like GPT and BERT.
    The key components of transformers are: embeddings, positional encoding, attention layers, and feed-forward networks.
    """
    
    DocumentManager.add_document('transformers.txt', new_doc_content)
    
    # List updated documents
    docs = DocumentManager.list_documents()
    print(f"\nUpdated documents: {len(docs)}")
    for doc in docs:
        print(f"  - {doc}")


def example_5_vector_store_info():
    """Example 5: Vector store information and management"""
    print("\n" + "="*60)
    print("Example 5: Vector Store Management")
    print("="*60)
    
    info = VectorStoreManager.get_vector_store_info()
    
    print(f"Vector Store Status:")
    print(f"  - Status: {info.get('status', 'unknown')}")
    
    if info['status'] == 'initialized':
        print(f"  - Path: {info.get('path')}")
        print(f"  - Size: {info.get('size_mb')} MB")
        print(f"  - Files: {info.get('files')}")


def example_6_rebuild_vector_store():
    """Example 6: Force rebuild vector store"""
    print("\n" + "="*60)
    print("Example 6: Rebuild Vector Store")
    print("="*60)
    
    print("Clearing old vector store...")
    VectorStoreManager.clear_vector_store()
    
    print("Rebuilding from documents...")
    pipeline = RAGPipeline()
    documents = pipeline.load_documents()
    pipeline.build_vector_store(documents, force_rebuild=True)
    
    info = VectorStoreManager.get_vector_store_info()
    print(f"\nVector store rebuilt!")
    print(f"Size: {info.get('size_mb')} MB")


def example_7_error_handling():
    """Example 7: Error handling and edge cases"""
    print("\n" + "="*60)
    print("Example 7: Error Handling")
    print("="*60)
    
    try:
        pipeline = RAGPipeline()
        
        # Try to query without setup
        try:
            pipeline.query("Test question")
        except ValueError as e:
            print(f"✓ Caught expected error: {e}")
        
        # Setup pipeline
        pipeline.setup_pipeline()
        
        # Try various query patterns
        test_cases = [
            "",                    # Empty query
            "?",                   # Minimal query
            "What" * 100,         # Very long query
        ]
        
        for test_query in test_cases:
            if test_query:  # Skip empty
                try:
                    answer, _ = pipeline.query(test_query)
                    print(f"✓ Handled query of length {len(test_query)}")
                except Exception as e:
                    print(f"✗ Error with query: {e}")
    
    except Exception as e:
        print(f"Error in example: {e}")


def main():
    """Run all examples"""
    import os
    
    # Check if we can proceed
    if not Path('.env').exists():
        print("⚠ .env file not found. Please create one first.")
        print("Run: cp .env.example .env")
        print("Then add your Hugging Face API key")
        return
    
    # List of examples
    examples = [
        ("1", "Basic RAG Setup", example_1_basic_rag),
        ("2", "Batch Query Processing", example_2_batch_queries),
        ("3", "Custom Configuration", example_3_custom_config),
        ("4", "Document Management", example_4_document_management),
        ("5", "Vector Store Info", example_5_vector_store_info),
        ("6", "Rebuild Vector Store", example_6_rebuild_vector_store),
        ("7", "Error Handling", example_7_error_handling),
    ]
    
    print("\n" + "="*60)
    print("RAG Pipeline Advanced Examples")
    print("="*60)
    
    print("\nAvailable examples:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")
    
    print("\nTo run an example:")
    print("  python advanced_examples.py <number>")
    print("\nExample:")
    print("  python advanced_examples.py 1")
    
    # Check for command line argument
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        for num, name, func in examples:
            if num == choice:
                try:
                    func()
                except Exception as e:
                    print(f"Error running example: {e}")
                    import traceback
                    traceback.print_exc()
                return
        
        print(f"Unknown example: {choice}")
    else:
        print("\nRun with example number to execute")
        print("Example: python advanced_examples.py 1")


if __name__ == "__main__":
    main()
