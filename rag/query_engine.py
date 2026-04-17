import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
#from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
import time
# -------------------------
# INIT
# -------------------------
load_dotenv()
print("🚀 Script started")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "../chroma_db")

# -------------------------
# EMBEDDINGS
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    cache_folder="./hf_cache"
)

print("✅ Embeddings loaded")

# -------------------------
# VECTOR STORE
# -------------------------
vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)
start= time.time()
print("Retrieving docs....")
#retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10
    }
)
docs = retriever.invoke(query)

for i, d in enumerate(docs):
    print(f"\n--- DOC {i+1} ---\n", d.page_content)
print(f"Retrieval time:{time.time()-start:.2f}s")

# -------------------------
# LLM
# -------------------------
response=None
llm = ChatOllama(model="phi", temperature=0.2)
mid=time.time()
print(" calling LLM")
try:
    response = llm.invoke(query)   # ✅ make sure this exists
except Exception as e:
    print("LLM Error:", e)
    response = "No response generated"

end = time.time()
print(f"LLM time: {end - mid:.2f}s")

print(response)
# -------------------------
# PROMPT
# -------------------------
prompt = PromptTemplate.from_template("""
("system", """
You are a strict clinical assistant.

RULES:
- Use ONLY provided context
- If context is irrelevant → say "No relevant clinical data found"
- Do NOT guess
- Do NOT use outside knowledge
- Do NOT fabricate ICD codes
- Be precise and short
"""),
    ("human", """
Context:
{context}

Question:
{question}

Answer:
""")
])
# -------------------------
# FORMAT DOCS
# -------------------------

def format_docs(docs):
    return "\n\n".join(
        [f"[Doc {i+1}]\n{d.page_content}" for i, d in enumerate(docs)]
)

# -------------------------
# RAG CHAIN
# -------------------------
llm = OllamaLLM(model="mistral")
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# -------------------------
# QUERY FUNCTION
# -------------------------
def ask_query(query: str):
    response = rag_chain.invoke(query)
    print("\n✅ ANSWER:\n", response.content)

# -------------------------
# MAIN LOOP (FIXED)
# -------------------------
if __name__ == "__main__":
    print("🔥 Entering main loop")
    print("🚀 RAG system ready\n")

    while True:
        query = input("Enter query (or 'exit'): ")

        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        ask_query(query)