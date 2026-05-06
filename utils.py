"""
Utility functions for RAG Pipeline

This module contains helper functions for document processing,
vector store management, and other utilities.
"""

import os
from pathlib import Path
from typing import List
import json


class DocumentManager:
    """Manages document operations for RAG pipeline"""
    
    @staticmethod
    def add_document(file_path: str, content: str = None) -> bool:
        """
        Add a document to the documents directory
        
        Args:
            file_path (str): Path to save the document
            content (str): Content to write to the file
        
        Returns:
            bool: True if successful
        """
        doc_dir = Path('./data/documents')
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = doc_dir / Path(file_path).name
        
        if content is None:
            # Read from file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Document added: {target_path}")
        return True
    
    @staticmethod
    def list_documents() -> List[str]:
        """
        List all documents in the documents directory
        
        Returns:
            List of document file names
        """
        doc_dir = Path('./data/documents')
        
        if not doc_dir.exists():
            return []
        
        return [f.name for f in doc_dir.glob('*.txt')]
    
    @staticmethod
    def delete_document(file_name: str) -> bool:
        """
        Delete a document from the documents directory
        
        Args:
            file_name (str): Name of the document to delete
        
        Returns:
            bool: True if successful
        """
        doc_path = Path('./data/documents') / file_name
        
        if doc_path.exists():
            doc_path.unlink()
            print(f"Document deleted: {file_name}")
            return True
        
        return False
    
    @staticmethod
    def clear_all_documents() -> bool:
        """
        Clear all documents from the documents directory
        
        Returns:
            bool: True if successful
        """
        doc_dir = Path('./data/documents')
        
        if not doc_dir.exists():
            return True
        
        for file in doc_dir.glob('*.txt'):
            file.unlink()
        
        print("All documents cleared")
        return True


class VectorStoreManager:
    """Manages vector store operations"""
    
    @staticmethod
    def clear_vector_store() -> bool:
        """
        Clear the vector store
        
        Returns:
            bool: True if successful
        """
        vector_store_path = Path('./data/vector_store')
        
        if vector_store_path.exists():
            import shutil
            shutil.rmtree(vector_store_path)
            print("Vector store cleared")
            return True
        
        return False
    
    @staticmethod
    def get_vector_store_info() -> dict:
        """
        Get information about the vector store
        
        Returns:
            dict: Information about the vector store
        """
        vector_store_path = Path('./data/vector_store')
        
        if not vector_store_path.exists():
            return {"status": "not_initialized"}
        
        # Get size
        total_size = sum(f.stat().st_size for f in vector_store_path.rglob('*') if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        
        return {
            "status": "initialized",
            "path": str(vector_store_path),
            "size_mb": round(total_size_mb, 2),
            "files": len(list(vector_store_path.rglob('*')))
        }


class ConfigManager:
    """Manages configuration operations"""
    
    @staticmethod
    def save_config(config: dict, file_path: str = './config.json') -> bool:
        """
        Save configuration to JSON file
        
        Args:
            config (dict): Configuration dictionary
            file_path (str): Path to save the configuration
        
        Returns:
            bool: True if successful
        """
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {file_path}")
        return True
    
    @staticmethod
    def load_config(file_path: str = './config.json') -> dict:
        """
        Load configuration from JSON file
        
        Args:
            file_path (str): Path to the configuration file
        
        Returns:
            dict: Configuration dictionary
        """
        if not Path(file_path).exists():
            return {}
        
        with open(file_path, 'r') as f:
            return json.load(f)


def setup_project_structure():
    """Create necessary project directories"""
    directories = [
        './data/documents',
        './data/vector_store',
        './logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Project structure set up")


def print_project_stats():
    """Print project statistics"""
    doc_dir = Path('./data/documents')
    vector_store_dir = Path('./data/vector_store')
    
    print("\n" + "="*50)
    print("Project Statistics")
    print("="*50)
    
    # Documents stats
    if doc_dir.exists():
        doc_count = len(list(doc_dir.glob('*.txt')))
        print(f"Documents: {doc_count}")
    else:
        print("Documents: 0")
    
    # Vector store stats
    if vector_store_dir.exists():
        vs_info = VectorStoreManager.get_vector_store_info()
        print(f"Vector Store Status: {vs_info['status']}")
        if vs_info['status'] == 'initialized':
            print(f"Vector Store Size: {vs_info['size_mb']} MB")
    else:
        print("Vector Store: Not initialized")
    
    print("="*50 + "\n")
