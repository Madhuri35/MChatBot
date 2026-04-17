from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_docs(docs):
    """
    Splits each Document into smaller chunks to improve embedding quality.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)