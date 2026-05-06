"""
CLI Interface for RAG Pipeline

Provides a command-line interface for interacting with the RAG pipeline.
"""

import argparse
import sys
from pathlib import Path
from rag_pipeline import RAGPipeline
from utils import DocumentManager, VectorStoreManager, setup_project_structure, print_project_stats


def cmd_setup(args):
    """Setup project structure"""
    setup_project_structure()
    print("✓ Project structure initialized")


def cmd_query(args):
    """Query the RAG pipeline"""
    question = args.question
    
    pipeline = RAGPipeline()
    try:
        pipeline.setup_pipeline()
        answer, sources = pipeline.query(question)
        
        print(f"\nQuestion: {question}")
        print(f"\nAnswer: {answer}")
        print(f"\nSources ({len(sources)}):")
        for i, doc in enumerate(sources, 1):
            print(f"  {i}. {doc.metadata.get('source', 'Unknown')}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_interactive(args):
    """Interactive mode"""
    pipeline = RAGPipeline()
    try:
        pipeline.setup_pipeline()
        
        print("\nInteractive RAG Pipeline")
        print("Type 'exit' to quit, 'help' for commands\n")
        
        while True:
            try:
                question = input("Q: ").strip()
                
                if question.lower() == 'exit':
                    break
                
                if question.lower() == 'help':
                    print("""
Commands:
  exit  - Exit interactive mode
  help  - Show this help message
  Or ask any question!
""")
                    continue
                
                if not question:
                    continue
                
                answer, sources = pipeline.query(question)
                print(f"\nA: {answer}\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}\n")
    
    except Exception as e:
        print(f"Error: {e}")


def cmd_documents(args):
    """Manage documents"""
    if args.action == 'list':
        docs = DocumentManager.list_documents()
        if docs:
            print("Documents:")
            for doc in docs:
                print(f"  - {doc}")
        else:
            print("No documents found")
    
    elif args.action == 'add':
        if args.file:
            DocumentManager.add_document(args.file)
        else:
            print("Please provide file path with --file")
    
    elif args.action == 'delete':
        if args.file:
            DocumentManager.delete_document(args.file)
        else:
            print("Please provide file name with --file")
    
    elif args.action == 'clear':
        if input("Are you sure? (yes/no): ").lower() == 'yes':
            DocumentManager.clear_all_documents()
        else:
            print("Cancelled")


def cmd_vector_store(args):
    """Manage vector store"""
    if args.action == 'info':
        info = VectorStoreManager.get_vector_store_info()
        print(f"Status: {info.get('status')}")
        if info.get('status') == 'initialized':
            print(f"Path: {info.get('path')}")
            print(f"Size: {info.get('size_mb')} MB")
            print(f"Files: {info.get('files')}")
    
    elif args.action == 'clear':
        if input("Are you sure? (yes/no): ").lower() == 'yes':
            VectorStoreManager.clear_vector_store()
        else:
            print("Cancelled")
    
    elif args.action == 'rebuild':
        print("Rebuilding vector store...")
        pipeline = RAGPipeline()
        try:
            documents = pipeline.load_documents()
            pipeline.build_vector_store(documents, force_rebuild=True)
            print("✓ Vector store rebuilt")
        except Exception as e:
            print(f"Error: {e}")


def cmd_status(args):
    """Show project status"""
    print_project_stats()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='RAG-based LLM Application CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py query "What is Python?"
  python cli.py interactive
  python cli.py documents list
  python cli.py vector-store info
  python cli.py setup
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    subparsers.add_parser('setup', help='Initialize project structure')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG pipeline')
    query_parser.add_argument('question', help='Question to ask')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Interactive mode')
    
    # Documents command
    doc_parser = subparsers.add_parser('documents', help='Manage documents')
    doc_parser.add_argument('action', choices=['list', 'add', 'delete', 'clear'])
    doc_parser.add_argument('--file', help='File path or name')
    
    # Vector store command
    vs_parser = subparsers.add_parser('vector-store', help='Manage vector store')
    vs_parser.add_argument('action', choices=['info', 'clear', 'rebuild'])
    
    # Status command
    subparsers.add_parser('status', help='Show project status')
    
    args = parser.parse_args()
    
    # Route commands
    if args.command == 'setup':
        cmd_setup(args)
    elif args.command == 'query':
        cmd_query(args)
    elif args.command == 'interactive':
        cmd_interactive(args)
    elif args.command == 'documents':
        cmd_documents(args)
    elif args.command == 'vector-store':
        cmd_vector_store(args)
    elif args.command == 'status':
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
