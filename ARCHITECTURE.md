# Technical Architecture

## System Overview

This RAG-based LLM application combines three key components:

### 1. Document Processing Pipeline
- **Local Processing**: Documents are loaded and split into chunks locally
- **Chunking Strategy**: Recursive text splitting with configurable overlap
- **Storage**: Processed documents are kept in memory during session

### 2. Vector Embedding Layer (Local)
- **Technology**: Sentence Transformers (HuggingFace)
- **Process**: All-MiniLM-L6-v2 by default
- **Advantages**:
  - Runs entirely on your machine
  - No data sent to external servers
  - Fast inference (< 100ms per chunk)
  - Privacy-preserving
- **Vector Dimension**: 384-768 depending on model
- **Storage**: FAISS index stored locally

### 3. LLM Inference Layer (Cloud)
- **Service**: Hugging Face Inference API
- **Model**: Llama 3.1 3B (or user-selected)
- **Communication**: Only retrieval results + question sent
- **Process**:
  1. Context (retrieved docs) sent to LLM
  2. LLM generates response
  3. Response returned to client

## Data Flow

```
User Input
    ↓
Local Tokenization
    ↓
Local Embedding Generation
    ↓
Vector Similarity Search (FAISS)
    ↓
Retrieve Top-K Documents
    ↓
Prepare Prompt with Context
    ↓
API Request to Hugging Face
    ↓
LLM Processing (Llama 3.1)
    ↓
Response Generation
    ↓
Return to User
```

## Component Details

### RAG Pipeline Module (`rag_pipeline.py`)

**Class: RAGPipeline**
- Orchestrates the entire RAG workflow
- Manages embeddings, vector store, and LLM
- Provides high-level API

**Key Methods:**
```python
setup_pipeline()           # Initialize everything
load_documents()          # Load from disk
build_vector_store()      # Create/load FAISS index
create_qa_chain()         # Setup LangChain QA chain
query()                   # Execute a query
```

**Configuration:**
- Loaded from environment variables
- Can be passed as custom dict
- Fallback to defaults

### Vector Store Implementation

**Technology: FAISS (Facebook AI Similarity Search)**
- Open-source vector similarity search library
- Optimized for CPU and GPU
- Supports various index types
- Used: Flat index (L2 distance)

**Operations:**
1. **Indexing**: Documents → Embeddings → FAISS Index
2. **Search**: Query → Embedding → Find K nearest neighbors
3. **Persistence**: Serialized to disk for reuse

**Performance:**
- Indexing: ~100-500 docs/second
- Search: 1-5ms for 10K documents
- Memory: ~4 bytes per dimension per vector

### LangChain Integration

**Components Used:**
- `DirectoryLoader`: Load documents from filesystem
- `RecursiveCharacterTextSplitter`: Split documents into chunks
- `HuggingFaceEmbeddings`: Generate embeddings
- `FAISS`: Vector similarity search
- `HuggingFaceHub`: LLM interface
- `RetrievalQA`: QA chain combining retrieval + generation

**Chain Architecture:**
```
RetrievalQA
    ├─ Retriever (FAISS)
    │   └─ Vector Store with embeddings
    ├─ LLM (HuggingFaceHub)
    └─ Prompt Template
```

## Embedding Models

### Sentence Transformers
- Pre-trained neural networks for semantic similarity
- Based on BERT/RoBERTa architectures
- Fine-tuned on semantic textual similarity

### Model Comparison

| Model | Dimension | Speed | Accuracy | Size |
|-------|-----------|-------|----------|------|
| MiniLM-L6 | 384 | Very Fast | Good | 22MB |
| MPNet-base | 768 | Fast | Very Good | 420MB |
| RoBERTa-large | 1024 | Slow | Excellent | 1.3GB |

## LLM Models Available

### Llama Models
- Open-source by Meta
- Instruction-tuned variants available
- Good balance of quality and speed

### Model Sizes and Performance
| Size | Speed | Quality | RAM | Notes |
|------|-------|---------|-----|-------|
| 3B | Fast | Fair | 6GB | Good for simple tasks |
| 7B | Moderate | Good | 14GB | Recommended balance |
| 13B | Slow | Very Good | 26GB | Better quality |
| 70B | Very Slow | Excellent | 140GB | Best quality |

## Prompt Engineering

### Current Prompt Strategy
```
Use context to answer question.
If you don't know, say so.

Context: [retrieved documents]
Question: [user question]
Answer:
```

### Customization Points
1. System prompt
2. Context formatting
3. Number of examples
4. Output format specification

## Error Handling

**Key Error Scenarios:**

1. **API Authentication**
   - Check token validity
   - Verify permissions
   - Retry with exponential backoff

2. **Vector Store Issues**
   - Rebuild if corrupted
   - Clear and restart if too large
   - Check disk space

3. **Document Processing**
   - Skip malformed documents
   - Handle encoding issues
   - Validate chunk sizes

4. **LLM Errors**
   - API timeout handling
   - Graceful degradation
   - Retry mechanisms

## Performance Optimization

### Vector Store Optimization
- Use FAISS IndexFlatL2 for accuracy
- Use IndexIVFFlat for speed on large sets
- Batch embedding generation

### LLM Optimization
- Reduce token generation
- Use temperature strategically
- Minimize context length

### Memory Optimization
- Streaming document loading
- Batch processing
- Cache management

## Security Considerations

### Data Privacy
- **Local Processing**: Embeddings generated locally, not sent to cloud
- **API Communication**: Only query + retrieved context sent
- **Credential Management**: API keys in .env file (gitignored)

### Best Practices
1. Never commit `.env` file
2. Use read-only API tokens
3. Monitor API usage
4. Validate user inputs
5. Implement rate limiting

## Scalability

### Current Architecture Limits
- Documents: Up to ~1M vectors in memory (depends on embedding model)
- Queries per second: Limited by Hugging Face API rate limits
- Context window: ~2000 tokens (LLM dependent)

### Scaling Options
1. **Vector Store**: Switch to production vector database (Pinecone, Weaviate)
2. **LLM**: Self-host using Ollama or vLLM
3. **Documents**: Implement sharding across multiple stores
4. **Caching**: Add Redis for query result caching

## Deployment Options

### Option 1: Local Development
- Run on personal machine
- Good for testing and prototyping
- Limited by local hardware

### Option 2: Cloud Instance
- EC2/VM with GPUs
- Self-managed infrastructure
- Full control over resources

### Option 3: Managed Services
- AWS SageMaker
- Azure ML
- Google Cloud Vertex AI

### Option 4: Containerized (Docker)
- Package as container
- Deploy anywhere (K8s, Fargate, etc.)
- Consistent environment

## Testing Strategy

### Unit Testing
- Individual component testing
- Mock LLM responses
- Vector store operations

### Integration Testing
- End-to-end pipeline
- Document loading and processing
- Query execution

### Performance Testing
- Query latency
- Vector store size
- Memory usage

## Monitoring and Logging

### Key Metrics
- Query latency
- Document processing time
- Vector store size
- API token usage
- Error rates

### Logging Strategy
- Application logs to `logs/` directory
- API call logging
- Error tracking
- Performance metrics

## Future Enhancements

### Planned Features
1. Multi-document retrieval
2. Query rewriting/expansion
3. Semantic caching
4. Local LLM support
5. Web UI interface
6. Async operations
7. Batch processing
8. Advanced RAG techniques (HyDE, MultiQuery)

### Research Areas
- Improved context selection
- Few-shot learning in prompts
- Retriever fine-tuning
- Generative model selection
- Evaluation frameworks

## References

### Key Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Attention Is All You Need" (Vaswani et al., 2017)
- "FAISS: A library for efficient similarity search" (Johnson et al., 2019)

### Resources
- [LangChain Docs](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Hub](https://huggingface.co/)

## Troubleshooting Advanced Issues

### Vector Store Corruption
```python
VectorStoreManager.clear_vector_store()
pipeline.setup_pipeline(force_rebuild=True)
```

### Memory Leaks
- Clear vector store periodically
- Implement batch processing
- Use generators for large datasets

### Embedding Quality Issues
- Experiment with different models
- Adjust chunk size
- Pre-process documents (remove noise)

### LLM Response Quality
- Improve prompt template
- Retrieve more context
- Use better LLM model
- Add few-shot examples
