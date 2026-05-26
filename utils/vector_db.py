import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

class VectorDB:
    def __init__(self, persist_dir="./chroma_db", use_ollama=True):
        self.persist_dir = persist_dir
        self.use_ollama = use_ollama
        
        if self.use_ollama:
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        else:
            self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(name="nlp_docs")
    
    def split_text(self, text, chunk_size=1000, chunk_overlap=200):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = splitter.split_text(text)
        return chunks
    
    def add_documents(self, documents):
        for doc in documents:
            chunks = self.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                self.collection.add(
                    documents=[chunk],
                    metadatas=[{"filename": doc["filename"], "chunk_index": i}],
                    ids=[f"{doc['filename']}_{i}"]
                )
    
    def search(self, query, k=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results
    
    def get_retriever(self, k=3):
        vectorstore = Chroma(
            client=self.client,
            collection_name="nlp_docs",
            embedding_function=self.embeddings
        )
        return vectorstore.as_retriever(search_kwargs={"k": k})
    
    def get_collection_stats(self):
        return self.collection.count()