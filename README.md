# RAG-Based LLM Application

A Retrieval-Augmented Generation (RAG) application using LangChain with local embeddings and Hugging Face Inference API for the LLM.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Query                             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │   Query Embedding      │
          │   (Local - Sentence    │
          │   Transformers)        │
          └────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Vector Store (FAISS)       │
        │   Retrieve K Most Similar    │
        │   Document Chunks           │
        └──────────────┬───────────────┘
                       │
                       ▼
      ┌──────────────────────────────────┐
      │  Context + Query + Prompt        │
      └──────────────┬───────────────────┘
                     │
                     ▼
      ┌──────────────────────────────────┐
      │   Hugging Face Inference API     │
      │   (Llama 3.1 / 3B Model)         │
      └──────────────┬───────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   LLM Response       │
          │   (Augmented with    │
          │   Retrieved Context) │
          └──────────────────────┘
```

## Features

- **Local RAG Pipeline**: Vector embeddings and document processing happen locally
- **Hugging Face LLM Integration**: Uses Llama 3.1 3B model via Hugging Face Inference API
- **FAISS Vector Store**: Efficient similarity search on local embeddings
- **Document Management**: Easy document loading and management
- **Source Attribution**: Returns source documents for generated answers
- **Configurable**: Adjust chunk size, retrieval count, and other parameters

## ☁️ Google Colab Quick Start

For the **easiest setup**, use the **Google Colab notebook** instead of local installation:

📓 **[RAG_LLM_Colab.ipynb](RAG_LLM_Colab.ipynb)** - Complete RAG pipeline in Google Colab

**Advantages of Colab:**
- ✅ No local installation required
- ✅ Pre-configured environment
- ✅ Free GPU/TPU access
- ✅ Persistent storage with Google Drive
- ✅ Automatic logging and result saving
- ✅ ~5-10 minutes total setup

**To use the Colab notebook:**
1. Open [RAG_LLM_Colab.ipynb](RAG_LLM_Colab.ipynb)
2. Click "Open in Colab" 
3. Add your Hugging Face API key to Secrets (🔑 icon)
4. Run all cells in order

See [COLAB_NOTEBOOK.md](COLAB_NOTEBOOK.md) for detailed Colab instructions.

## Requirements

- Python 3.8+
- Hugging Face API Key (for LLM inference)
- 4GB+ RAM recommended
- Internet connection for Hugging Face Inference API

## Free Deployment

If you only want to deploy the local embedding model and FAISS retrieval layer for free, use the Streamlit app in [app.py](app.py).

### What this deployment includes
- Local document chunking and embeddings
- FAISS vector store creation and loading
- Retrieval-only question answering by default
- Optional Hugging Face LLM integration if you later add an API key

### Recommended free host
- Streamlit Community Cloud

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deployment notes
1. Push the repo to GitHub.
2. Connect the repo to Streamlit Community Cloud.
3. Set the app entry point to `app.py`.
4. Leave `HUGGINGFACE_API_KEY` unset to keep it retrieval-only and free.

## Installation

### 1. Clone the Repository

```bash
cd /Users/misc/RAGbasedAgent
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and add your Hugging Face API key:

```
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxx
HF_MODEL_ID=meta-llama/Llama-2-7b-chat-hf
```

**To get your Hugging Face API key:**
1. Visit https://huggingface.co/settings/tokens
2. Create a new token with "read" access
3. Paste it in your `.env` file

**For Llama 3.1 3B Model:**
- Llama 3.1 8B: `meta-llama/Llama-3.1-8B-Instruct`
- Or use alternative small models for testing

## Project Structure

```
RAGbasedAgent/
├── .env.example              # Environment configuration template
├── requirements.txt          # Python dependencies
├── rag_pipeline.py          # Core RAG pipeline implementation
├── main.py                  # Main application entry point
├── utils.py                 # Utility functions and helpers
├── README.md                # This file
├── data/
│   ├── documents/           # Store your text documents here
│   └── vector_store/        # FAISS vector store (auto-created)
└── logs/                    # Application logs (auto-created)
```

## Usage

### Basic Usage (with Sample Documents)

Run the main application which creates sample documents and runs example queries:

```bash
python main.py
```

This will:
1. Create sample documents about Python, Machine Learning, and LLMs
2. Build a vector store from those documents
3. Run example queries
4. Enter interactive mode

### Add Custom Documents

Place your text files (`.txt`) in the `data/documents/` directory:

```bash
cp your_document.txt data/documents/
```

Then rebuild the vector store:

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.setup_pipeline(force_rebuild=True)
```

### Use as a Module

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Setup with your documents
qa_chain = pipeline.setup_pipeline()

# Query
answer, sources = pipeline.query("Your question here?")

print(f"Answer: {answer}")
print(f"Sources: {len(sources)} documents used")
```

### Use Document Manager

```python
from utils import DocumentManager

# List documents
docs = DocumentManager.list_documents()
print(docs)

# Add new document
DocumentManager.add_document('new_doc.txt', 'Document content here')

# Delete document
DocumentManager.delete_document('old_doc.txt')

# Clear all
DocumentManager.clear_all_documents()
```

## Configuration

Edit `.env` file to customize:

```env
# LLM Settings
HF_MODEL_ID=meta-llama/Llama-2-7b-chat-hf
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512

# RAG Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=3

# Embedding Model (runs locally)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Paths
VECTOR_STORE_PATH=./data/vector_store
DOCUMENT_PATH=./data/documents
```

### Configuration Parameters Explained

- **CHUNK_SIZE**: Size of document chunks (in characters) for embedding
- **CHUNK_OVERLAP**: Overlap between chunks to maintain context
- **RETRIEVAL_K**: Number of documents to retrieve for context
- **LLM_TEMPERATURE**: Creativity of responses (0.0 = deterministic, 1.0 = creative)
- **LLM_MAX_TOKENS**: Maximum response length
- **EMBEDDING_MODEL**: Sentence transformer model for local embeddings

## Performance Tips

1. **Reduce CHUNK_SIZE** for faster embedding but less context per chunk
2. **Reduce RETRIEVAL_K** for faster retrieval but potentially less context
3. **Use smaller embedding models** like `all-MiniLM-L6-v2` for faster inference
4. **Cache vector store** - it's reused if documents haven't changed
5. **Batch queries** - make multiple queries with same vector store

## Troubleshooting

### API Key Issues

```
Error: "Invalid API key"
```

Solution: Make sure your Hugging Face API key is correct and has read access.

### Out of Memory

```
Error: "CUDA out of memory" or "killed"
```

Solution:
- Reduce `CHUNK_SIZE` in `.env`
- Use a smaller embedding model
- Reduce `RETRIEVAL_K`

### No Documents Found

```
Error: "No documents found. Cannot build vector store."
```

Solution:
1. Make sure text files are in `data/documents/` directory
2. Files must have `.txt` extension
3. Run `main.py` first to create sample documents

### Slow Vector Store Building

This is normal for large document sets. The vector store is cached after first build, so subsequent runs are fast.

## API Costs

- **Hugging Face Inference API**: Free tier available with rate limits
  - Pro plan: $9/month for unlimited API access
  - Check https://huggingface.co/inference-api for current pricing

- **Local Embeddings**: FREE (runs on your machine)

## Advanced Features

### Custom Prompt Template

Modify the prompt in `rag_pipeline.py` `create_qa_chain()` method:

```python
template = """Your custom prompt template
{context}
{question}
"""
```

### Different Embedding Models

```python
pipeline.config['embedding_model'] = 'sentence-transformers/all-mpnet-base-v2'
pipeline.embeddings = HuggingFaceEmbeddings(
    model_name=pipeline.config['embedding_model']
)
```

### Use Different Vector Store

Replace FAISS with other options:
- Pinecone
- Weaviate
- Milvus
- Qdrant

## Example Queries

Try these with the sample documents:

1. "What is Python and what is it used for?"
2. "Explain the difference between supervised and unsupervised learning"
3. "What are Large Language Models?"
4. "How does Retrieval-Augmented Generation work?"

## Limitations

- Requires internet connection for Hugging Face Inference API
- LLM inference speed depends on Hugging Face API response time
- Local embeddings are limited to ~1M vectors on standard hardware (FAISS)
- Document processing is sequential

## Future Enhancements

- [ ] Add support for PDF, DOCX documents
- [ ] Implement query expansion
- [ ] Add semantic caching
- [ ] Support for local LLM inference (Ollama, LLaMA.cpp)
- [ ] Web UI interface
- [ ] Batch processing
- [ ] Async operations

## License

This project is open source. Feel free to modify and use as needed.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review LangChain documentation: https://python.langchain.com/
3. Check Hugging Face documentation: https://huggingface.co/docs

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Inference API](https://huggingface.co/inference-api)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Concept](https://arxiv.org/abs/2005.11401)
