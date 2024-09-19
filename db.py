import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st

# Constants
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
VECTOR_STORE_DIR = "./vectorstore/"
COLLECTION_NAME = "counsel_chat_qa"

def display_vectordb_contents():
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # Initialize the Chroma client
    db = Chroma(
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
    )

    # Get all the documents from the collection
    results = db.get()

    # Display the results
    st.write(f"Total documents in the vector store: {len(results['ids'])}")

    for i, (id, document, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
        st.write(f"\nDocument {i+1}:")
        st.write(f"ID: {id}")
        st.write(f"Content: {document}")
        st.write(f"Metadata: {metadata}")

        # Optional: Display embeddings (Note: This can be very verbose)
        # st.write(f"Embedding: {results['embeddings'][i]}")

        st.write("---")  # Separator between documents

if __name__ == "__main__":
    st.set_page_config(page_title="Vector DB Contents", page_icon="📚", layout="wide")
    st.title("📚 Vector Database Contents")
    
    display_vectordb_contents()