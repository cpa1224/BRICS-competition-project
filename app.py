import streamlit as st
import os
import tempfile
from utils.document_loader import load_document, split_text
from utils.vector_db import create_vector_db, load_vector_db
from utils.rag_chain import create_rag_chain, ask_question, check_ollama_status, get_available_models

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "document_count" not in st.session_state:
        st.session_state.document_count = 0
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "ollama_status" not in st.session_state:
        st.session_state.ollama_status = (False, "未检查")
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "deepseek-r1:7b"
    if "model_status" not in st.session_state:
        st.session_state.model_status = ""

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name

def check_and_update_ollama_status():
    status, msg = check_ollama_status()
    st.session_state.ollama_status = (status, msg)
    if status:
        st.session_state.available_models = get_available_models()

def main():
    init_session_state()
    
    st.set_page_config(page_title="RAG智能问答系统", page_icon="🤖", layout="wide")
    
    st.title("🤖 RAG智能问答系统")
    
    with st.sidebar:
        st.title("Ollama大模型配置")
        
        if st.button("🔄 检查Ollama状态"):
            with st.spinner("检查中..."):
                check_and_update_ollama_status()
        
        ollama_available, ollama_msg = st.session_state.ollama_status
        
        if ollama_available:
            st.success(f"✅ Ollama服务正常运行")
            st.info(f"可用模型: {len(st.session_state.available_models)} 个")
            
            if st.session_state.available_models:
                selected_model = st.selectbox(
                    "选择大模型",
                    st.session_state.available_models,
                    index=st.session_state.available_models.index(st.session_state.selected_model) 
                    if st.session_state.selected_model in st.session_state.available_models else 0
                )
                st.session_state.selected_model = selected_model
            else:
                st.warning("暂无可用模型，请先下载模型")
                st.code("ollama pull deepseek-r1:7b")
        else:
            st.error(f"❌ Ollama服务未就绪")
            st.warning(ollama_msg)
            st.info("请按照以下步骤配置Ollama:\n1. 安装Ollama\n2. 启动Ollama服务\n3. 下载模型: ollama pull deepseek-r1:7b")
        
        st.divider()
        
        st.title("知识库管理")
        uploaded_files = st.file_uploader("上传文档", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        
        if st.button("构建知识库"):
            if not uploaded_files:
                st.warning("请先上传文档！")
            else:
                with st.spinner("正在处理文档..."):
                    all_text = ""
                    for uploaded_file in uploaded_files:
                        temp_path = save_uploaded_file(uploaded_file)
                        text = load_document(temp_path)
                        if text:
                            all_text += text + "\n\n"
                        os.unlink(temp_path)
                    
                    if not all_text:
                        st.error("未提取到有效文本内容！")
                    else:
                        chunks = split_text(all_text, chunk_size=1000, chunk_overlap=200)
                        st.session_state.chunk_count = len(chunks)
                        st.session_state.document_count = len(uploaded_files)
                        
                        persist_dir = "./chroma_db"
                        if os.path.exists(persist_dir):
                            import shutil
                            shutil.rmtree(persist_dir)
                        
                        st.session_state.vectordb = create_vector_db(chunks, persist_dir)
                        st.session_state.chain, status_msg = create_rag_chain(
                            st.session_state.vectordb, 
                            st.session_state.selected_model
                        )
                        st.session_state.model_status = status_msg
                        
                        if st.session_state.chain:
                            st.success(f"知识库构建完成！\n文档数: {st.session_state.document_count}\n文本块数: {st.session_state.chunk_count}")
                            st.info(status_msg)
    
    st.sidebar.subheader("知识库状态")
    st.sidebar.info(f"已加载文档: {st.session_state.document_count} 个\n文本块数: {st.session_state.chunk_count} 个")
    
    if st.session_state.model_status:
        st.sidebar.info(f"当前模型状态: {st.session_state.model_status}")
    
    st.subheader("问答交互")
    
    if st.session_state.chat_history:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(f"**用户:** {question}")
            with st.chat_message("assistant"):
                st.write(f"**助手:** {answer}")
    
    user_question = st.text_input("请输入您的问题", key="question_input")
    
    if st.button("提问"):
        if not user_question.strip():
            st.warning("请输入问题！")
        elif not st.session_state.chain:
            st.warning("请先构建知识库！")
        else:
            with st.spinner("正在思考..."):
                try:
                    answer = ask_question(st.session_state.chain, user_question)
                    st.session_state.chat_history.append((user_question, answer))
                    
                    with st.chat_message("user"):
                        st.write(f"**用户:** {user_question}")
                    with st.chat_message("assistant"):
                        st.write(f"**助手:** {answer}")
                except Exception as e:
                    st.error(f"回答错误: {e}")

if __name__ == "__main__":
    main()