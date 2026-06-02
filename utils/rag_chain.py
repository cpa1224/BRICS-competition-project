import re
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class SimpleQA:
    def __init__(self, vectordb):
        self.vectordb = vectordb
        self.chat_history = []
    
    def get_answer(self, question):
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            return "鏂囨。涓湭鎵惧埌鐩稿叧绛旀"
        
        context = "\n\n".join([doc.page_content for doc in docs])
        answer = self.extract_answer(question, context)
        
        self.chat_history.append({"question": question, "answer": answer})
        return answer
    
    def extract_answer(self, question, context):
        question_lower = question.lower()
        
        if "浠€涔堟槸" in question or "瀹氫箟" in question or "姒傚康" in question:
            pattern = r"(浠€涔堟槸\s*[\u4e00-\u9fa5]+|[\u4e00-\u9fa5]+[\u7684]?瀹氫箟?[\uFF1A:]?\s*[^銆傦紵锛乚*[銆傦紵锛乚)"
            match = re.search(pattern, context)
            if match:
                return match.group(0).strip()
        
        if "鐗圭偣" in question or "鐗瑰緛" in question or "浼樺娍" in question:
            pattern = r"(鐗圭偣|鐗瑰緛|浼樺娍|涓昏鐗圭偣|涓昏鐗瑰緛)[\uFF1A:]?\s*[^銆傦紵锛乚*[銆傦紵锛乚"
            match = re.search(pattern, context)
            if match:
                return match.group(0).strip()
        
        if "鏂规硶" in question or "鎶€鏈? in question or "绠楁硶" in question:
            pattern = r"(鏂规硶|鎶€鏈瘄绠楁硶|姝ラ|娴佺▼)[\uFF1A:]?\s*[^銆傦紵锛乚*[銆傦紵锛乚"
            match = re.search(pattern, context)
            if match:
                return match.group(0).strip()
        
        sentences = re.split(r'[銆傦紒锛焆', context)
        keywords = ["鑷劧璇█澶勭悊", "NLP", "Transformer", "BERT", "璇嶅祵鍏?, "鏂囨湰鍒嗙被", "娣卞害瀛︿範", "鏈哄櫒瀛︿範"]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            question_words = question_lower.replace("浠€涔?, "").replace("鏄?, "").replace("锛?, "").strip()
            if question_words and question_words in sentence_lower:
                return sentence.strip() + "銆?
        
        for keyword in keywords:
            if keyword in question:
                for sentence in sentences:
                    if keyword in sentence:
                        return sentence.strip() + "銆?
        
        if len(context) > 500:
            return context[:500] + "..."
        return context

def create_rag_chain(vectordb, model_name="deepseek-r1:7b"):
    if OLLAMA_AVAILABLE:
        try:
            llm = OllamaLLM(model=model_name)
            
            template = """
            You are an assistant that answers questions based on the provided reference documents.

            Reference documents:
            {context}

            Chat history:
            {chat_history}

            Current question:
            {question}

            Guidelines:
            1. Answer the question strictly based on the information in the reference documents.
            2. If the documents do not contain relevant information to answer the question, clearly state "鏂囨。涓湭鎵惧埌鐩稿叧绛旀".
            3. Do not use external knowledge outside of the provided documents.
            4. Keep the answer concise and relevant.

            Answer:
            """
            
            prompt = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=template
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                verbose=True
            )
            return chain
        except Exception as e:
            print(f"Failed to create Ollama chain: {e}")
            return SimpleQA(vectordb)
    else:
        return SimpleQA(vectordb)

def ask_question(chain, question):
    if hasattr(chain, "get_answer"):
        return chain.get_answer(question)
    else:
        result = chain({"question": question})
        return result.get("answer", "鏂囨。涓湭鎵惧埌鐩稿叧绛旀")