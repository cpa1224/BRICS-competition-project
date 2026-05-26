from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class RAGChain:
    def __init__(self, retriever, model_name="deepseek-r1:7b"):
        self.llm = Ollama(model=model_name)
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.prompt_template = """
        你是一个专业的问答助手。请基于提供的参考文档回答用户问题。
        
        参考文档：
        {context}
        
        请严格按照以下规则回答：
        1. 只使用参考文档中的信息进行回答
        2. 如果文档中没有相关信息，请明确回答"文档中未找到相关答案"
        3. 如果文档中有相关信息，请基于文档内容给出详细回答
        4. 回答要简洁明了，不要添加无关内容
        
        问题：{question}
        
        回答：
        """
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt}
        )
    
    def ask(self, question):
        result = self.chain({"question": question})
        return result["answer"]
    
    def clear_memory(self):
        self.memory.clear()