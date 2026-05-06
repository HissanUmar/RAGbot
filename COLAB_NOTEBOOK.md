# Google Colab RAG-Based LLM Notebook

## 📓 RAG_LLM_Colab.ipynb

This is a comprehensive Google Colab notebook that sets up and runs a complete Retrieval-Augmented Generation (RAG) pipeline using LangChain, FAISS, and Hugging Face models.

## 🚀 Quick Start

### Step 1: Open the Notebook in Google Colab
- Copy the notebook file `RAG_LLM_Colab.ipynb`
- Upload it to Google Drive
- Open with Google Colab

Or use this direct link format:
```
https://colab.research.google.com/drive/[FILE_ID]
```

### Step 2: Set Up Hugging Face API Key (Optional but Recommended)
1. Click the 🔑 **Secrets** icon in the left sidebar
2. Create a new secret named `HF_API_KEY`
3. Get your key from: https://huggingface.co/settings/tokens
4. The notebook will use `gpt2` model if no key is provided

### Step 3: Run Cells in Order
Run each cell sequentially from top to bottom. Total execution time: ~5-10 minutes

## 📋 Notebook Sections

1. **Section 1: Set Up Google Colab Runtime and Storage**
   - Verify Colab environment
   - Mount Google Drive
   - Create project directories

2. **Section 2: Install Required Dependencies**
   - Install LangChain, FAISS, Hugging Face libraries
   - ~20 packages total

3. **Section 3: Upload Project Files, Data, and Logs**
   - Create sample documents (Python, ML, LLM topics)
   - Prepare corpus for RAG pipeline

4. **Section 4: Configure Environment Variables**
   - Load API keys from secrets
   - Set up configuration parameters

5. **Section 5: Load Documents and Prepare the Dataset**
   - Load text documents
   - Split into chunks for embedding

6. **Section 6: Create Embeddings and Build the FAISS Index**
   - Generate embeddings locally
   - Create vector search index
   - Test retrieval

7. **Section 7: Initialize the RAG Pipeline and LLM**
   - Configure LangChain QA chain
   - Set up retriever and prompt template

8. **Section 8: Run Retrieval-Augmented Inference**
   - Execute example queries
   - Display answers with sources

9. **Section 9: Save Outputs and Runtime Logs**
   - Save results to JSON
   - Generate summary report
   - Backup to Google Drive

10. **Bonus: Interactive Query Interface**
    - Run custom queries
    - Edit examples for your questions

11. **Verification Cell**
    - Check all dependencies

## 💾 Output Files

All outputs are saved in the `logs` directory with timestamps:

- `rag_results_[timestamp].json` - Query results
- `config_[timestamp].json` - Configuration used
- `summary_[timestamp].txt` - Human-readable summary

## 🔧 Customization

### Add Your Own Documents
Replace the sample documents in Section 3 with your own corpus:

```python
# Instead of sample docs, load your own:
with open('your_document.txt', 'r') as f:
    content = f.read()
```

### Change the LLM Model
Modify in Section 4:
```python
'hf_model_id': 'meta-llama/Llama-2-7b-chat-hf'  # Change this
```

### Adjust RAG Parameters
In Section 4, modify the config dictionary:
- `chunk_size`: Size of document chunks
- `retrieval_k`: Number of documents to retrieve
- `llm_temperature`: Response creativity (0.0-1.0)

## ⚙️ Requirements

- **Google Colab Account** (free)
- **Hugging Face Account** (free, for API key)
- **Google Drive Access** (optional, for backup)
- **No GPU required** (CPU works fine for this use case)

## 📊 Performance

- **Installation**: ~2 minutes
- **Vector Store Build**: ~1 minute  
- **Sample Queries**: ~3-5 minutes (depends on LLM API latency)
- **Total**: ~5-10 minutes

## 🎓 What You'll Learn

1. How to set up a RAG pipeline in Google Colab
2. Local embeddings with Sentence Transformers
3. Vector search with FAISS
4. LLM integration via Hugging Face API
5. LangChain orchestration
6. Production-ready logging and error handling

## ❓ Troubleshooting

### "API Key not found" Warning
- Not all models require an API key (gpt2 doesn't)
- For premium models (Llama, GPT), you need a key
- Add it via Secrets (🔑 icon)

### "Out of Memory" Error
- Google Colab free tier has 12GB RAM
- This notebook uses ~2-3GB (safe)
- If error occurs, restart runtime and try again

### Slow LLM Responses
- Due to Hugging Face API latency
- Free tier has rate limits
- Pro tier available for faster access

### Document Loading Issues
- Ensure files are in `data/documents/` directory
- Only `.txt` files are loaded
- Use UTF-8 encoding

## 🔗 Useful Links

- [Google Colab](https://colab.research.google.com/)
- [Hugging Face Tokens](https://huggingface.co/settings/tokens)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)

## 💡 Tips

1. **Save Checkpoint**: After setup, save the Colab state to avoid re-running installation
2. **Use Google Drive**: Mount Drive for persistent storage of large vector stores
3. **Batch Queries**: Process multiple questions efficiently
4. **Experiment**: Try different embedding models and LLMs
5. **Monitor Costs**: Check Hugging Face API usage regularly

## 🎯 Next Steps

1. ✅ Run the notebook with sample documents
2. 📚 Replace with your own documents
3. ⚙️ Fine-tune configuration parameters
4. 🚀 Deploy to production
5. 🔄 Integrate with your application

## 📝 Notes

- All code is well-commented
- Error handling is comprehensive
- Results are automatically saved
- Backed up to Google Drive if available

## Support

For issues or questions:
- Check the Troubleshooting section above
- Review inline code comments
- Refer to documentation links
- Check GitHub issues for similar problems

---

**Happy RAG-ing in Google Colab!** 🚀
