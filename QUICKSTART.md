# Quick Start Guide

Get your RAG-based LLM application running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Internet connection
- Hugging Face account (free)

## Step 1: Get Hugging Face API Key (2 minutes)

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "RAG Pipeline")
4. Select "Read" access
5. Click "Create token"
6. Copy the token

## Step 2: Setup Environment (1 minute)

```bash
# Navigate to project directory
cd /Users/misc/RAGbasedAgent

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Configure (1 minute)

```bash
# Copy example config
cp .env.example .env

# Edit .env with your editor
# Add your Hugging Face API key to: HUGGINGFACE_API_KEY=hf_xxxx
```

## Step 4: Run (1 minute)

```bash
python main.py
```

That's it! The application will:
- Create sample documents
- Build a vector store locally
- Run example queries
- Enter interactive mode where you can ask questions

## Common Commands

### Run with sample documents
```bash
python main.py
```

### Run advanced examples
```bash
# Example 1: Basic RAG
python advanced_examples.py 1

# Example 2: Batch queries
python advanced_examples.py 2

# Example 3: Custom config
python advanced_examples.py 3
```

### Use as Python module
```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.setup_pipeline()
answer, sources = pipeline.query("Your question?")
print(answer)
```

## Next Steps

1. Read [README.md](README.md) for complete documentation
2. Add your own documents to `data/documents/` directory
3. Rebuild vector store: `pipeline.setup_pipeline(force_rebuild=True)`
4. Customize configuration in `.env` file
5. Check `advanced_examples.py` for more usage patterns

## Troubleshooting

### "Invalid API key"
- Make sure your Hugging Face token has "Read" access
- Verify the token is correctly pasted in `.env`

### "No documents found"
- Make sure text files are in `data/documents/` directory
- Check file extensions are `.txt`
- Run `main.py` first to create sample documents

### Application is slow
- First run is slow while building vector store
- Subsequent runs are much faster
- Check your internet connection for LLM inference latency

### Out of memory error
- Reduce `CHUNK_SIZE` in `.env` (try 500)
- Reduce `RETRIEVAL_K` in `.env` (try 2)
- Use a smaller embedding model

## Model Information

### Default Model: Llama 2 7B Chat
- Size: ~7 billion parameters
- Speed: Moderate (depends on Hugging Face API)
- Quality: Good
- Cost: Free tier available

### Alternative Models

For faster responses (smaller models):
```
gpt2 - 124M parameters (very fast, lower quality)
distilbert-base-uncased - Fast, good for encoding
```

For better quality (larger models):
```
meta-llama/Llama-2-13b-chat-hf - Better quality, slower
meta-llama/Llama-2-70b-chat-hf - Best quality, slowest
```

Change in `.env`:
```
HF_MODEL_ID=meta-llama/Llama-2-13b-chat-hf
```

## Architecture Overview

```
Your Question
    ↓
Local Embeddings (runs on your machine) ← Fast
    ↓
Vector Search (FAISS) ← Fast
    ↓
Find Similar Documents
    ↓
Send to Hugging Face LLM with Context
    ↓
Get Answer from Llama 3.1 3B
    ↓
Return Answer + Sources
```

## Performance Tips

1. **First time**: Vector store building takes 30 seconds
2. **Subsequent**: Each query takes 5-10 seconds (API latency)
3. **Reduce latency**: Use smaller chunks, fewer retrievals
4. **Batch queries**: Process multiple questions with same vector store

## Getting Help

- Check [README.md](README.md) troubleshooting section
- Review error messages carefully
- Check Hugging Face status: https://huggingface.co/status
- Read LangChain docs: https://python.langchain.com/

## What's Happening Under the Hood

1. **Local Embeddings**: Your documents are converted to vectors using Sentence Transformers (runs locally, private)
2. **Vector Store**: FAISS indexes these vectors for fast similarity search
3. **Retrieval**: When you ask a question, it's embedded locally and similar documents are found
4. **Augmentation**: Your question + similar documents are sent to Llama 3.1 via Hugging Face API
5. **Response**: The LLM generates an answer based on the provided context

## Security Notes

- Your documents stay on your machine (local embeddings)
- Only the question + retrieved documents go to Hugging Face API
- Hugging Face doesn't store your inference data
- Keep your API key safe (don't commit `.env` to git)

## Have Fun!

Your RAG pipeline is now ready. Try these questions with the sample documents:
- "What is Python used for?"
- "Explain machine learning"
- "How do LLMs work?"
- "What's the difference between supervised and unsupervised learning?"

Happy querying! 🚀
