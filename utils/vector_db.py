import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def tokenize_chinese(text):
    words = re.findall(r'[\u4e00-\u9fa5]{1,2}|[a-zA-Z]+|\d+', text.lower())
    return set(words)

class SimpleVectorDB:
    def __init__(self):
        self.chunks = []
        self.chunk_tokens = []
    
    def add_texts(self, texts):
        self.chunks = texts
        self.chunk_tokens = [tokenize_chinese(text) for text in texts]
    
    def similarity_search(self, query, k=3):
        if not self.chunks:
            return []
        
        query_tokens = tokenize_chinese(query)
        if not query_tokens:
            return []
        
        scores = []
        for i, chunk_tokens in enumerate(self.chunk_tokens):
            if chunk_tokens:
                intersection = query_tokens & chunk_tokens
                if len(chunk_tokens) > 0:
                    score = len(intersection) / len(query_tokens)
                else:
                    score = 0
                scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:k]:
            if score > 0:
                results.append({"text": self.chunks[idx], "similarity": score})
        return results
    
    def as_retriever(self, search_kwargs=None):
        return SimpleRetriever(self, search_kwargs or {"k": 3})

class SimpleRetriever:
    def __init__(self, vectordb, search_kwargs):
        self.vectordb = vectordb
        self.k = search_kwargs.get("k", 3)
    
    def get_relevant_documents(self, query):
        results = self.vectordb.similarity_search(query, k=self.k)
        return [SimpleDocument(r["text"]) for r in results]

class SimpleDocument:
    def __init__(self, page_content):
        self.page_content = page_content

def create_vector_db(chunks, persist_directory="./chroma_db"):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectordb.persist()
        return vectordb
    except Exception as e:
        print(f"Failed to create Chroma DB: {e}")
        vectordb = SimpleVectorDB()
        vectordb.add_texts(chunks)
        return vectordb

def load_vector_db(persist_directory="./chroma_db"):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectordb
    except Exception as e:
        print(f"Failed to load Chroma DB: {e}")
        return SimpleVectorDB()