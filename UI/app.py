import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
import os

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Clinical Assistant", page_icon="🩺", layout="wide")

st.title("🩺 Clinical Assistant")
st.markdown("Your AI-powered clinical RAG assistant")

# -------------------------
# LOAD MODELS (CACHE)
# -------------------------
@st.cache_resource
def load_rag():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PERSIST_DIR = os.path.join(BASE_DIR, "../chroma_db")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model="phi", temperature=0.2)

    prompt = PromptTemplate.from_template("""
    You are a clinical assistant.

    Use ONLY the context below to answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain, vector_store

rag_chain, vector_store = load_rag()

# -------------------------
# SESSION STATE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# USER INPUT
# -------------------------
query = st.chat_input("Ask a clinical question...")

if query:
    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke(query)
            answer = response.content
        except Exception as e:
            answer = f"Error: {e}"

    st.session_state.history.append((query, answer))
    st.subheader("💬 Chat History")



# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------
for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# -------------------------
# SIMILAR SEARCHES
# -------------------------
st.sidebar.title("🔍 Similar Searches")

if st.session_state.history:
    last_query = st.session_state.history[-1][0]
    docs = vector_store.similarity_search(last_query, k=5)

    for i, doc in enumerate(docs):
        st.sidebar.markdown(f"**{i+1}.** {doc.page_content[:80]}...")

# -------------------------
# FOOTER
# -------------------------
st.sidebar.markdown("---")
#st.sidebar.markdown("⚡ Built with RAG + Ollama")
