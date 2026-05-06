"""
Main Application Entry Point

This script demonstrates how to use the RAG-based LLM application.
"""

import os
from pathlib import Path
from rag_pipeline import RAGPipeline


def create_sample_documents():
    """Create sample documents for testing"""
    doc_path = Path('./data/documents')
    doc_path.mkdir(parents=True, exist_ok=True)
    
    # Sample document 1: Information about Python
    python_doc = """Python is a high-level, interpreted programming language known for its simplicity and readability.
    Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability and simplicity.
    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    Common applications of Python include web development, data analysis, artificial intelligence, and automation.
    Popular Python libraries include NumPy, Pandas, Flask, Django, and TensorFlow."""
    
    # Sample document 2: Information about Machine Learning
    ml_doc = """Machine Learning is a subset of Artificial Intelligence that enables systems to learn and improve from experience.
    Machine Learning algorithms can be classified into three main categories: supervised learning, unsupervised learning, and reinforcement learning.
    Supervised learning involves training on labeled data, where the desired output is known.
    Unsupervised learning finds patterns in unlabeled data without predefined outputs.
    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.
    Common ML algorithms include Linear Regression, Logistic Regression, Decision Trees, Random Forests, and Neural Networks."""
    
    # Sample document 3: Information about LLMs
    llm_doc = """Large Language Models (LLMs) are neural networks trained on vast amounts of text data.
    LLMs can perform a wide range of natural language processing tasks including translation, summarization, and question answering.
    Popular LLMs include GPT-3, GPT-4, BERT, T5, and Llama models.
    LLMs use transformer architecture which relies on attention mechanisms to understand relationships between words.
    Retrieval-Augmented Generation (RAG) enhances LLMs by providing them with relevant context from external documents.
    This allows LLMs to provide more accurate and up-to-date responses based on specific knowledge bases."""
    
    # Write documents
    with open(doc_path / 'python.txt', 'w') as f:
        f.write(python_doc)
    
    with open(doc_path / 'machine_learning.txt', 'w') as f:
        f.write(ml_doc)
    
    with open(doc_path / 'llms.txt', 'w') as f:
        f.write(llm_doc)
    
    print(f"Sample documents created in {doc_path}")


def main():
    """Main application function"""
    
    # Check if .env file exists, if not copy from .env.example
    if not Path('.env').exists() and Path('.env.example').exists():
        print("Creating .env file from .env.example")
        import shutil
        shutil.copy('.env.example', '.env')
        print("Please update .env with your Hugging Face API key")
    
    # Create sample documents if they don't exist
    if not Path('./data/documents').exists() or not list(Path('./data/documents').glob('*.txt')):
        print("\nCreating sample documents...")
        create_sample_documents()
    
    # Initialize RAG pipeline
    print("\nInitializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # Setup the complete pipeline
    try:
        qa_chain = pipeline.setup_pipeline()
        
        # Example queries
        queries = [
            "What is Python and its main applications?",
            "Explain the difference between supervised and unsupervised learning",
            "What are Large Language Models and how do they work?",
        ]
        
        print("\n" + "="*50)
        print("Running Example Queries")
        print("="*50)
        
        for query in queries:
            try:
                answer, sources = pipeline.query(query)
                print(f"\nAnswer: {answer}")
                print(f"\nSource Documents: {len(sources)}")
                for i, doc in enumerate(sources, 1):
                    print(f"  {i}. {doc.metadata.get('source', 'Unknown')}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing query: {e}")
        
        # Interactive mode
        print("\n" + "="*50)
        print("Interactive Mode")
        print("="*50)
        print("Enter your questions (type 'exit' to quit):\n")
        
        while True:
            user_query = input("Your question: ").strip()
            if user_query.lower() == 'exit':
                break
            
            if not user_query:
                continue
            
            try:
                answer, sources = pipeline.query(user_query)
                print(f"\nAnswer: {answer}")
                print(f"\nSources: {len(sources)} document(s) used")
                for i, doc in enumerate(sources, 1):
                    print(f"  {i}. {doc.metadata.get('source', 'Unknown')}")
                print()
            except Exception as e:
                print(f"Error: {e}\n")
    
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        print("\nMake sure to:")
        print("1. Update .env with your Hugging Face API key")
        print("2. Have documents in ./data/documents/ directory")


if __name__ == "__main__":
    main()
