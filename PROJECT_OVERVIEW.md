# PROJECT OVERVIEW

## RAG-Based LLM Application

A production-ready Retrieval-Augmented Generation (RAG) application that combines local vector embeddings with Llama 3.1 3B via Hugging Face Inference API.

### 🎯 What This Project Does

1. **Loads your documents** into a local knowledge base
2. **Converts them to embeddings** locally (private, no data sent to cloud)
3. **Indexes them** using FAISS for fast retrieval
4. **When you ask a question**:
   - Finds similar documents locally
   - Sends them + your question to Llama 3.1
   - Returns an informed answer with source attribution

### ✨ Key Features

- ✅ **Local RAG Pipeline** - Fast vector search on your machine
- ✅ **Cloud LLM Integration** - Uses Llama 3.1 via HF API (free tier available)
- ✅ **Privacy-First** - Only query + context sent to LLM, documents stay local
- ✅ **Easy to Use** - Simple API and CLI interface
- ✅ **Highly Configurable** - Adjust every parameter
- ✅ **Production Ready** - Error handling, logging, optimization

## 📁 Project Structure

```
RAGbasedAgent/
├── 📄 Core Files
│   ├── main.py                 # Main application & examples
│   ├── rag_pipeline.py        # RAG pipeline implementation
│   ├── utils.py               # Helper utilities
│   └── cli.py                 # Command-line interface
│
├── 📚 Documentation
│   ├── README.md              # Complete documentation
│   ├── QUICKSTART.md          # 5-minute setup guide
│   ├── CONFIGURATION.md       # Configuration parameters
│   ├── ARCHITECTURE.md        # Technical details
│   └── PROJECT_OVERVIEW.md    # This file
│
├── 📦 Configuration
│   ├── requirements.txt        # Python dependencies
│   └── .env.example           # Configuration template
│
├── 📁 Data Directories
│   ├── data/
│   │   ├── documents/         # Your documents go here
│   │   └── vector_store/      # FAISS vector store
│   └── logs/                  # Application logs
│
└── 🔧 Development Files
    └── .gitignore            # Git configuration
```

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Get API Key
1. Go to https://huggingface.co/settings/tokens
2. Create token with "read" access
3. Copy token

### Step 3: Configure
```bash
cp .env.example .env
# Edit .env and paste your token to HUGGINGFACE_API_KEY
```

### Step 4: Run
```bash
python main.py
```

That's it! You'll have:
- Sample documents automatically created
- Vector store built locally
- Example queries running
- Interactive mode for asking questions

## 📖 Documentation Guide

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[README.md](README.md)** | Complete guide | First read for full understanding |
| **[QUICKSTART.md](QUICKSTART.md)** | Fast setup | Get running in 5 minutes |
| **[CONFIGURATION.md](CONFIGURATION.md)** | All settings | When customizing parameters |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Technical deep dive | Understanding how it works |

## 🎓 Usage Examples

### Using Python API
```python
from rag_pipeline import RAGPipeline

# Setup
pipeline = RAGPipeline()
pipeline.setup_pipeline()

# Query
answer, sources = pipeline.query("What is machine learning?")
print(f"Answer: {answer}")
print(f"Sources: {len(sources)} documents used")
```

### Using CLI
```bash
# Interactive mode
python cli.py interactive

# Single query
python cli.py query "What is Python?"

# Manage documents
python cli.py documents list
python cli.py documents add --file mydoc.txt

# Vector store
python cli.py vector-store info
python cli.py vector-store rebuild
```

### Using main.py
```bash
# Runs with sample documents and interactive mode
python main.py
```

## 🔧 Configuration Profiles

### Default (Balanced)
- Good quality answers
- Reasonable speed
- Works for most use cases

### Fast Mode
```env
CHUNK_SIZE=500
RETRIEVAL_K=2
LLM_MODEL=gpt2
```

### Quality Mode
```env
CHUNK_SIZE=1500
RETRIEVAL_K=5
LLM_MODEL=meta-llama/Llama-2-13b-chat-hf
```

See [CONFIGURATION.md](CONFIGURATION.md) for all options.

## 📊 Architecture

```
┌─────────────┐
│  Documents  │
│ (in folder) │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Text Splitting   │ (Local)
│ Chunking         │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Embeddings       │ (Local - Sentence Transformers)
│ Generation       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  FAISS Vector   │ (Local - Fast Search)
│  Store          │
└──────┬───────────┘
       │
       ▼ (when querying)
┌─────────────────────────────┐
│ User Question Embedding     │ (Local)
│ Similarity Search           │
│ Retrieve Top-K Documents   │
└──────┬──────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Hugging Face Inference API   │
│ (Llama 3.1 3B Model)        │
│ Process with Context         │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────┐
│ Generated Answer │
│ + Attribution    │
└──────────────────┘
```

## 💡 Key Concepts

### RAG (Retrieval-Augmented Generation)
- Retrieves relevant documents for context
- Passes context + question to LLM
- Results in more accurate, grounded responses
- Better than pure LLM for specific knowledge domains

### Local Embeddings
- Sentence Transformers run on your machine
- Documents converted to vectors (384-768 dimensions)
- Indexed in FAISS for fast similarity search
- Completely private - nothing sent to cloud

### Vector Search
- FAISS (Facebook AI Similarity Search)
- Efficient nearest-neighbor search
- Handles millions of vectors
- Used for finding similar documents

### LLM Integration
- Llama 3.1 3B model via Hugging Face API
- Generates responses based on context + question
- Free tier available, paid options for heavy use
- Customizable parameters (temperature, max tokens)

## 🎯 Use Cases

### Information Retrieval
"Answer questions about my company documents"

### Knowledge Base
"Create a Q&A system for my documentation"

### Research Assistant
"Analyze and summarize research papers"

### Customer Support
"Automate support responses using knowledge base"

### Content Analysis
"Extract insights from document collection"

## ⚙️ System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for models + documents
- **Internet**: Required for Hugging Face API
- **OS**: Linux, macOS, Windows

## 📈 Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| First setup | 60-120s | Downloads models |
| Vector store build | Varies | ~100-500 docs/sec |
| Single query | 5-10s | API latency |
| Vector search | < 10ms | Local FAISS search |
| Embedding gen | 10-100ms | Local transformation |

## 🔐 Security & Privacy

- ✅ Documents stay on your machine
- ✅ Embeddings generated locally
- ✅ Only question + context sent to cloud
- ✅ API key managed via .env (excluded from git)
- ✅ No data stored by Hugging Face

## 📚 What's Included

### Core Modules
1. **rag_pipeline.py** - RAG implementation (300+ lines)
2. **main.py** - Example application (200+ lines)
3. **utils.py** - Helper functions (150+ lines)
4. **cli.py** - Command-line interface (250+ lines)

### Documentation
- README with full guide
- Quick start (5 minutes)
- Configuration guide (50+ options)
- Architecture documentation
- This overview

### Examples
- Sample documents generator
- Batch query processing
- Custom configuration
- Document management
- Error handling

## 🚦 Getting Started Path

1. **Read** [QUICKSTART.md](QUICKSTART.md) (5 min)
2. **Run** `python main.py` (2 min)
3. **Explore** sample queries (5 min)
4. **Read** [README.md](README.md) (15 min)
5. **Configure** for your use case (10 min)
6. **Add** your documents (5 min)
7. **Customize** as needed (ongoing)

## 🆘 Troubleshooting

### Common Issues & Quick Fixes

| Issue | Fix |
|-------|-----|
| "API key invalid" | Check token at huggingface.co/settings/tokens |
| "No documents found" | Put .txt files in data/documents/ directory |
| "Out of memory" | Reduce CHUNK_SIZE in .env |
| "Slow responses" | Reduce RETRIEVAL_K or use smaller LLM |
| "Low quality answers" | Increase RETRIEVAL_K or better documents |

See [README.md](README.md#troubleshooting) for more.

## 📞 Support Resources

- **Documentation**: Read the included .md files
- **Hugging Face**: https://huggingface.co/docs
- **LangChain**: https://python.langchain.com/
- **Issues**: Check error messages and logs

## 🎓 Learning Resources

- RAG Concept: https://arxiv.org/abs/2005.11401
- LangChain Tutorial: https://python.langchain.com/en/latest/getting_started/quickstart.html
- Sentence Transformers: https://www.sbert.net/
- FAISS: https://faiss.ai/

## 🔮 What's Next

After setup, you can:
1. Add your own documents
2. Experiment with different LLM models
3. Fine-tune embedding model
4. Deploy to cloud
5. Build a web interface
6. Optimize for your use case

## 📝 Configuration Quick Reference

```env
# Required
HUGGINGFACE_API_KEY=your_token

# Performance
CHUNK_SIZE=1000              # Document chunk size
RETRIEVAL_K=3               # Documents to retrieve
LLM_TEMPERATURE=0.7         # Response creativity

# Models
HF_MODEL_ID=meta-llama/Llama-2-7b-chat-hf  # LLM
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Local embeddings

# Paths
DOCUMENT_PATH=./data/documents
VECTOR_STORE_PATH=./data/vector_store
```

See [CONFIGURATION.md](CONFIGURATION.md) for 50+ configuration options.

## 🎯 Next Steps

1. **Start Here**: [QUICKSTART.md](QUICKSTART.md)
2. **Deep Dive**: [README.md](README.md)
3. **Configure**: [CONFIGURATION.md](CONFIGURATION.md)
4. **Understand**: [ARCHITECTURE.md](ARCHITECTURE.md)
5. **Advanced**: Run `python advanced_examples.py`

---

**Ready to build your RAG pipeline?** Start with [QUICKSTART.md](QUICKSTART.md) and you'll be running in 5 minutes! 🚀
