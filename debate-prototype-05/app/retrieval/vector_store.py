import streamlit as st
from langchain_community.vectorstores import FAISS
from typing import Optional
from workflow.config import get_embeddings
from retrieval.wikipedia_service import get_wikipedia_content


@st.cache_resource
def create_vector_store(topic: str) -> Optional[FAISS]:
    embeddings = get_embeddings()
    documents = []

    wiki_docs = get_wikipedia_content(topic, "ko")
    if wiki_docs:
        documents.extend(wiki_docs)

    if documents:
        try:
            vector_store = FAISS.from_documents(documents, embeddings)
            return vector_store
        except Exception as e:
            st.error(f"Vector DB 생성 중 오류 발생: {str(e)}")
            return None
    else:
        return None
