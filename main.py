import os
from typing import List

import chromadb
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from datasets import load_dataset
import logging
import streamlit as st

# CONSTANTS
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_NAME = "mixtral-8x7b-32768"  # Groq model name
LLM_TEMPERATURE = 0.1

CHUNK_SIZE = 512
VECTOR_STORE_DIR = "./vectorstore/"
COLLECTION_NAME = "counsel_chat_qa"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class CounselorLLM:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        self.llm = ChatGroq(
            temperature=LLM_TEMPERATURE,
            model_name=LLM_NAME,
            max_tokens=1024,
        )
        self.retriever = self.get_vectorstore_retriever()
        self.chain = self.create_rag_chain()

    def get_vectorstore_retriever(self) -> VectorStoreRetriever:
        db = chromadb.PersistentClient(VECTOR_STORE_DIR)
        try:
            db.get_collection(COLLECTION_NAME)
            retriever = Chroma(
                embedding_function=self.embedding_model,
                collection_name=COLLECTION_NAME,
                persist_directory=VECTOR_STORE_DIR,
            ).as_retriever(search_kwargs={"k": 3})
        except:
            logger.info("Vectorstore doesn't exist. Creating it...")
            dataset = self.load_dataset()
            chunks = self.process_dataset(dataset)
            retriever = self.create_and_store_embeddings(chunks).as_retriever(search_kwargs={"k": 3})

        return retriever

    def load_dataset(self):
        logger.info("Loading the Counsel Chat dataset...")
        return load_dataset("nbertagnolli/counsel-chat")

    def process_dataset(self, dataset) -> List[Document]:
        logger.info("Processing the dataset...")
        questions = dataset['train']['questionText']
        answers = dataset['train']['answerText']

        documents = []
        for q, a in zip(questions, answers):
            if q is not None and a is not None:
                doc = Document(
                    page_content=f"Question: {q}\nAnswer: {a}",
                    metadata={"question": q, "answer": a}
                )
                documents.append(doc)

        logger.info(f"Processed {len(documents)} documents.")
        return documents

    def create_and_store_embeddings(self, chunks: List[Document]) -> Chroma:
        vectorstore = Chroma.from_documents(
            chunks,
            embedding=self.embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_STORE_DIR,
        )
        logger.info("Vectorstore created and embeddings stored.")
        return vectorstore

    def create_rag_chain(self) -> Runnable:
        template = """You are a professional counselor AI assistant. Your role is to provide empathetic, 
        supportive, and insightful responses to people seeking advice or support. Use the following context
        and question to formulate your response. Remember to:

        1. Show empathy and understanding
        2. Provide non-judgmental support
        3. Offer practical advice when appropriate
        4. Encourage seeking professional help for serious issues
        5. Maintain confidentiality and ethical standards

        Context:
        {context}

        Question: {input}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, document_chain)

        return retrieval_chain

    def generate_answer(self, query: str) -> str:
        logger.info(f"Generating answer for query: {query}")
        response = self.chain.invoke({"input": query})

        print("\nRetrieved documents:")
        for i, doc in enumerate(response["context"], 1):
            print(f"\nDocument {i}:")
            print(doc.page_content)

        return response["answer"]

def main():
    st.set_page_config(page_title="AI Counselor ChatBot", page_icon="🧠", layout="wide")

    st.title("🧠 AI Counselor ChatBot")
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f4f8;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    counselor_llm = CounselorLLM()


    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's on your mind today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            response = counselor_llm.generate_answer(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()