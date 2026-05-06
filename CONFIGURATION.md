# Configuration Guide

Detailed explanation of all configuration parameters for the RAG pipeline.

## Environment Variables (.env)

### Hugging Face Configuration

#### HUGGINGFACE_API_KEY
- **Required**: Yes
- **Type**: String
- **Description**: Your Hugging Face API token for accessing the inference API
- **How to get**: https://huggingface.co/settings/tokens
- **Example**: `HUGGINGFACE_API_KEY=hf_abcdef123456`
- **Note**: Keep this secure, never commit to git

#### HF_MODEL_ID
- **Default**: `meta-llama/Llama-2-7b-chat-hf`
- **Type**: String
- **Description**: The model ID to use from Hugging Face Model Hub
- **Options**:
  - `meta-llama/Llama-2-7b-chat-hf` - 7B chat model (recommended)
  - `meta-llama/Llama-3.1-8B-Instruct` - Llama 3.1 8B (newer, better)
  - `gpt2` - Small model for testing (very fast)
  - `tiiuae/falcon-7b-instruct` - Falcon 7B alternative
- **Performance**:
  - Smaller models: Faster but lower quality
  - Larger models: Slower but higher quality

### Vector Store Configuration

#### VECTOR_STORE_PATH
- **Default**: `./data/vector_store`
- **Type**: Path
- **Description**: Directory where FAISS vector store is saved
- **Note**: Auto-created if doesn't exist

#### DOCUMENT_PATH
- **Default**: `./data/documents`
- **Type**: Path
- **Description**: Directory where source documents are stored
- **Requirements**:
  - Files must be `.txt` format (plain text)
  - UTF-8 encoding recommended
  - Can organize in subdirectories (recursively loaded)
- **Note**: Auto-created if doesn't exist

### Embedding Configuration

#### EMBEDDING_MODEL
- **Default**: `sentence-transformers/all-MiniLM-L6-v2`
- **Type**: String
- **Description**: Sentence transformer model for local embeddings
- **Options**:
  - `sentence-transformers/all-MiniLM-L6-v2` - Lightweight (recommended for speed)
  - `sentence-transformers/all-mpnet-base-v2` - Better quality, larger
  - `sentence-transformers/all-roberta-large-v1` - Highest quality, slowest
  - `distiluse-base-multilingual-cased-v2` - Multilingual support
- **Trade-offs**:
  - Smaller: Faster, lower memory, less accurate
  - Larger: Slower, more memory, more accurate
- **Note**: Downloaded and cached locally on first use

### RAG Pipeline Configuration

#### CHUNK_SIZE
- **Default**: `1000`
- **Type**: Integer (characters)
- **Description**: Size of document chunks for embedding
- **Considerations**:
  - Smaller (300-500): Better context boundaries, but more chunks
  - Larger (1500-2000): Fewer chunks, more context per chunk
  - 1000: Good balance
- **Impact**: Affects both quality and performance
- **Example**: `CHUNK_SIZE=1000`

#### CHUNK_OVERLAP
- **Default**: `200`
- **Type**: Integer (characters)
- **Description**: Overlap between consecutive chunks
- **Purpose**: Maintains context continuity between chunks
- **Recommendation**: 10-20% of chunk size
- **Example**: `CHUNK_OVERLAP=200` (20% of 1000)

#### RETRIEVAL_K
- **Default**: `3`
- **Type**: Integer
- **Description**: Number of document chunks to retrieve for context
- **Trade-offs**:
  - Lower (1-2): Faster, less context, faster API calls
  - Higher (5-10): Slower API calls, more context, better answers
  - 3: Good balance
- **Considerations**:
  - Each chunk increases prompt length
  - Longer prompts = slower and more expensive API calls
- **Recommendation**: Start at 3, adjust based on response quality

### LLM Configuration

#### LLM_TEMPERATURE
- **Default**: `0.7`
- **Type**: Float (0.0 - 1.0)
- **Description**: Controls randomness in LLM responses
- **Values**:
  - `0.0`: Deterministic, predictable (use for factual answers)
  - `0.3-0.5`: Conservative, focused responses
  - `0.7`: Balanced (recommended)
  - `1.0`: Maximum randomness, creative responses
- **Use cases**:
  - Question answering: 0.3-0.5 (factual)
  - Creative writing: 0.8-1.0
  - General: 0.7 (balanced)

#### LLM_MAX_TOKENS
- **Default**: `512`
- **Type**: Integer
- **Description**: Maximum length of generated response
- **Considerations**:
  - Shorter (128-256): Quick, concise responses
  - Longer (512-1024): More detailed, comprehensive
  - 512: Good for most use cases
- **Note**: Limits response length; answer may be cut off if too short

## Configuration Profiles

### Profile 1: Speed Optimized
For fast responses with lower quality:
```env
CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVAL_K=2
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=256
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_MODEL_ID=gpt2
```

### Profile 2: Quality Optimized
For best answers with slower responses:
```env
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
RETRIEVAL_K=5
LLM_TEMPERATURE=0.5
LLM_MAX_TOKENS=1024
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
HF_MODEL_ID=meta-llama/Llama-2-13b-chat-hf
```

### Profile 3: Balanced (Default)
Good balance of speed and quality:
```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=3
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_MODEL_ID=meta-llama/Llama-2-7b-chat-hf
```

### Profile 4: Memory Constrained
For limited RAM (< 4GB):
```env
CHUNK_SIZE=400
CHUNK_OVERLAP=50
RETRIEVAL_K=2
LLM_TEMPERATURE=0.5
LLM_MAX_TOKENS=256
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_MODEL_ID=gpt2
```

## How to Change Configuration

### Method 1: Edit .env File
```bash
# Copy from example
cp .env.example .env

# Edit with your editor
nano .env  # or vim, code, etc.
```

### Method 2: Programmatically
```python
from rag_pipeline import RAGPipeline

config = {
    'chunk_size': 500,
    'chunk_overlap': 100,
    'retrieval_k': 2,
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'llm_temperature': 0.3,
    'llm_max_tokens': 256,
    'vector_store_path': './data/vector_store',
    'document_path': './data/documents',
    'hf_model_id': 'gpt2',
    'hf_api_key': 'hf_xxxx',
}

pipeline = RAGPipeline(config=config)
```

## Common Configuration Scenarios

### Scenario 1: "Responses are too slow"
1. Reduce `RETRIEVAL_K` from 3 to 2
2. Reduce `CHUNK_SIZE` from 1000 to 500
3. Switch to smaller LLM: `HF_MODEL_ID=gpt2`
4. Reduce `LLM_MAX_TOKENS` from 512 to 256

### Scenario 2: "Responses are not accurate"
1. Increase `RETRIEVAL_K` from 3 to 5
2. Increase `CHUNK_SIZE` from 1000 to 1500
3. Switch to better LLM: `HF_MODEL_ID=meta-llama/Llama-2-13b-chat-hf`
4. Reduce `LLM_TEMPERATURE` from 0.7 to 0.3

### Scenario 3: "Running out of memory"
1. Use smaller embedding model: `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
2. Reduce `CHUNK_SIZE` from 1000 to 400
3. Clear vector store: `VectorStoreManager.clear_vector_store()`
4. Restart application

### Scenario 4: "Getting too long responses"
1. Reduce `LLM_MAX_TOKENS` from 512 to 256
2. Reduce `RETRIEVAL_K` to provide less context
3. Adjust prompt template in code

### Scenario 5: "Responses are too generic"
1. Increase `LLM_TEMPERATURE` from 0.7 to 0.9
2. Increase `RETRIEVAL_K` to get more specific context
3. Use better LLM model
4. Check document quality

## Performance Benchmarks

These are approximate (actual times vary based on hardware and API latency):

| Config | First Query | Subsequent | Token/sec | Quality |
|--------|------------|-----------|-----------|---------|
| Speed | 45s | 2-3s | 50+ | Low |
| Balanced | 60s | 5-8s | 30-40 | Medium |
| Quality | 120s | 10-15s | 20-30 | High |

*First query includes vector store building and model downloads*

## Model Comparison

| Model | Size | Speed | Quality | Cost |
|-------|------|-------|---------|------|
| GPT-2 | 124M | Very Fast | Low | Free |
| DistilBERT | 66M | Fast | Low | Free |
| Llama 2 7B | 7B | Moderate | Medium | Free |
| Llama 2 13B | 13B | Slow | High | Free |
| Llama 2 70B | 70B | Very Slow | Very High | Free |

## Advanced Configuration

### Custom Prompt Template
Edit `rag_pipeline.py` in the `create_qa_chain` method:

```python
template = """Answer based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
```

### Custom Vector Store Location
```env
VECTOR_STORE_PATH=/custom/path/to/vectorstore
```

### Multiple Vector Stores
Create different pipeline instances with different configs:
```python
pipeline1 = RAGPipeline(config={'vector_store_path': './stores/store1'})
pipeline2 = RAGPipeline(config={'vector_store_path': './stores/store2'})
```

## Troubleshooting Configuration Issues

### "ModuleNotFoundError"
- Missing dependencies: `pip install -r requirements.txt`

### "CUDA out of memory"
- Reduce embedding model size
- Reduce chunk size
- Clear vector store

### "Model not found on Model Hub"
- Check model ID spelling
- Ensure model exists on https://huggingface.co/models
- Use model ID with organization: `meta-llama/Llama-2-7b-chat-hf`

### "API key invalid"
- Get new token from https://huggingface.co/settings/tokens
- Token must have at least "read" access
- No quotes around token in .env

## Recommended Starting Configuration

For most use cases:
```env
HUGGINGFACE_API_KEY=your_token_here
HF_MODEL_ID=meta-llama/Llama-2-7b-chat-hf
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
VECTOR_STORE_PATH=./data/vector_store
DOCUMENT_PATH=./data/documents
```

Start with this, then adjust based on your needs!
