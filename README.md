Step 1: Go to Your RAG Project Folder
Step 2: Create New Virtual Environment
Step 3: Activate the Environment
step4:pip install langchain langchain-community langchain-huggingface chromadb sentence-transformers pandas
Step : Add API Key(Optional but recomended)
Step 9: Save Dependencies

RAG Steps: Load Your Clinical CSV
👉 Load your CSV
👉 Convert it into documents (LangChain format)

# Load CSV
df = pd.read_csv("clinical_data.csv")

# Convert rows to documents

query.py:

1.Load the Chroma vector store
2.Initialize a local LLM
3.Create RetrievalQA chain
4.Ask a query