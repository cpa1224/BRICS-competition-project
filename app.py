import streamlit as st
import os
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def load_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def load_document(file):
    filename = file.name
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pdf':
        return load_pdf(file)
    elif ext == '.docx':
        return load_docx(file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

def init_vector_db():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = "./chroma_db"
    
    if os.path.exists(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    return vectorstore

def init_rag_chain(retriever):
    llm = Ollama(model="deepseek-r1:7b")
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    prompt_template = """
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
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return chain

def main():
    st.set_page_config(page_title="RAG问答系统", page_icon="📚", layout="wide")
    
    st.title("📚 RAG问答系统")
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = init_vector_db()
    
    if "rag_chain" not in st.session_state:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        st.session_state.rag_chain = init_rag_chain(retriever)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("文档上传")
        uploaded_files = st.file_uploader(
            "选择PDF或DOCX文件",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(file.name)
        
        if st.button("🔄 构建知识库"):
            if uploaded_files:
                with st.spinner("正在处理文档..."):
                    for file in uploaded_files:
                        try:
                            text = load_document(file)
                            chunks = split_text(text)
                            st.session_state.vectorstore.add_texts(chunks)
                            st.success(f"✅ {file.name} 已成功处理")
                        except Exception as e:
                            st.error(f"❌ 处理 {file.name} 失败: {e}")
                    
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    st.session_state.rag_chain = init_rag_chain(retriever)
                    st.success("🎉 知识库构建完成！")
            else:
                st.warning("请先上传文档")
        
        st.header("知识库状态")
        try:
            collection_stats = st.session_state.vectorstore._collection.count()
            st.info(f"📊 当前知识库中文本块数量: {collection_stats}")
            st.info(f"📁 已上传文档数量: {len(st.session_state.uploaded_files)}")
        except Exception as e:
            st.info("📊 当前知识库中文本块数量: 0")
        
        if st.button("🗑️ 清空知识库"):
            import shutil
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
            st.session_state.vectorstore = init_vector_db()
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            st.session_state.rag_chain = init_rag_chain(retriever)
            st.session_state.chat_history = []
            st.session_state.uploaded_files = []
            st.success("✅ 知识库已清空")
    
    with col2:
        st.header("问答交互")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        user_input = st.text_input("请输入您的问题：")
        
        if st.button("提问"):
            if user_input.strip():
                with st.spinner("正在思考..."):
                    try:
                        answer = st.session_state.rag_chain({"question": user_input})
                        
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_input
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer["answer"]
                        })
                        
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"问答失败: {e}")
            else:
                st.warning("请输入问题")

if __name__ == "__main__":
    main()