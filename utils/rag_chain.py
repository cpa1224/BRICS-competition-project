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
            return "文档中未找到相关答案"
        
        context = "\n\n".join([doc.page_content for doc in docs])
        answer = self.extract_answer(question, context)
        
        self.chat_history.append({"question": question, "answer": answer})
        return answer
    
    def extract_answer(self, question, context):
        question_lower = question.lower()
        
        if "什么是" in question or "定义" in question or "概念" in question:
            pattern = r"(什么是\s*[\u4e00-\u9fa5]+|[\u4e00-\u9fa5]+[\u7684]?定义?[\uFF1A:]?\s*[^。！？]*[。！？])"
            match = re.search(pattern, context)
            if match:
                return match.group(0).strip()
        
        if "特点" in question or "特征" in question or "优势" in question:
            pattern = r"(特点|特征|优势|主要特点|主要特征)[\uFF1A:]?\s*[^。！？]*[。！？]"
            match = re.search(pattern, context)
            if match:
                return match.group(0).strip()
        
        if "方法" in question or "技术" in question or "算法" in question:
            pattern = r"(方法|技术|算法|步骤|流程)[\uFF1A:]?\s*[^。！？]*[。！？]"
            match = re.search(pattern, context)
            if match:
                return match.group(0).strip()
        
        sentences = re.split(r'[。！？]', context)
        keywords = ["自然语言处理", "NLP", "Transformer", "BERT", "词嵌入", "文本分类", "深度学习", "机器学习"]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            question_words = question_lower.replace("什么", "").replace("是", "").replace("？", "").strip()
            if question_words and question_words in sentence_lower:
                return sentence.strip() + "。"
        
        for keyword in keywords:
            if keyword in question:
                for sentence in sentences:
                    if keyword in sentence:
                        return sentence.strip() + "。"
        
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
            2. If the documents do not contain relevant information to answer the question, clearly state "文档中未找到相关答案".
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
        return result.get("answer", "文档中未找到相关答案")