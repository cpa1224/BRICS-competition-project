import streamlit as st
import os
import tempfile
from utils.document_loader import load_document, split_text
from utils.vector_db import create_vector_db, load_vector_db
from utils.rag_chain import create_rag_chain, ask_question

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

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name

def main():
    init_session_state()
    
    st.set_page_config(page_title="RAG智能问答系统", page_icon="🤖", layout="wide")
    
    st.title("🤖 RAG智能问答系统")
    st.sidebar.title("知识库管理")
    
    with st.sidebar:
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
                        st.session_state.chain = create_rag_chain(st.session_state.vectordb)
                        
                        if st.session_state.chain:
                            st.success(f"知识库构建完成！\n文档数: {st.session_state.document_count}\n文本块数: {st.session_state.chunk_count}")
    
    st.sidebar.subheader("知识库状态")
    st.sidebar.info(f"已加载文档: {st.session_state.document_count} 个\n文本块数: {st.session_state.chunk_count} 个")
    
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