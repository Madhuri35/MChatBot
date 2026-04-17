
from langchain.docstore.document import Document

def build_documents(df):
    """
    Converts a pandas DataFrame into a list of LangChain Documents.
    """
    docs = []

    for _, row in df.iterrows():
        content = f"""
        Patient ID: {row.get('patient_id', '')}
        Category: {row.get('category', '')}
        Doctor: {row.get('doctor', '')}
        Notes: {row.get('notes', '')}
        """
        docs.append(
            Document(
                page_content=content.strip(),
                metadata={
                    "patient_id": str(row.get("patient_id", "")),
                    "category": row.get("category", ""),
                    "table": "clinical_data"
                }
            )
        )
    return docs