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
    
    st.set_page_config(page_title="RAG鏅鸿兘闂瓟绯荤粺", page_icon="馃摎", layout="wide")
    
    st.title("馃摎 RAG鏅鸿兘闂瓟绯荤粺")
    st.sidebar.title("鐭ヨ瘑搴撶鐞?)
    
    with st.sidebar:
        uploaded_files = st.file_uploader("涓婁紶鏂囨。", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        
        if st.button("鏋勫缓鐭ヨ瘑搴?):
            if not uploaded_files:
                st.warning("璇峰厛涓婁紶鏂囨。锛?)
            else:
                with st.spinner("姝ｅ湪澶勭悊鏂囨。..."):
                    all_text = ""
                    for uploaded_file in uploaded_files:
                        temp_path = save_uploaded_file(uploaded_file)
                        text = load_document(temp_path)
                        if text:
                            all_text += text + "\n\n"
                        os.unlink(temp_path)
                    
                    if not all_text:
                        st.error("鏈彁鍙栧埌鏈夋晥鏂囨湰鍐呭锛?)
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
                            st.success(f"鐭ヨ瘑搴撴瀯寤哄畬鎴愶紒\n鏂囨。鏁? {st.session_state.document_count}\n鏂囨湰鍧楁暟: {st.session_state.chunk_count}")
    
    st.sidebar.subheader("鐭ヨ瘑搴撶姸鎬?)
    st.sidebar.info(f"宸插姞杞芥枃妗? {st.session_state.document_count} 涓猏n鏂囨湰鍧楁暟: {st.session_state.chunk_count} 涓?)
    
    st.subheader("闂瓟浜や簰")
    
    if st.session_state.chat_history:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(f"**鐢ㄦ埛:** {question}")
            with st.chat_message("assistant"):
                st.write(f"**鍔╂墜:** {answer}")
    
    user_question = st.text_input("璇疯緭鍏ユ偍鐨勯棶棰?", key="question_input")
    
    if st.button("鎻愰棶"):
        if not user_question.strip():
            st.warning("璇疯緭鍏ラ棶棰橈紒")
        elif not st.session_state.chain:
            st.warning("璇峰厛鏋勫缓鐭ヨ瘑搴擄紒")
        else:
            with st.spinner("姝ｅ湪鎬濊€?.."):
                try:
                    answer = ask_question(st.session_state.chain, user_question)
                    st.session_state.chat_history.append((user_question, answer))
                    
                    with st.chat_message("user"):
                        st.write(f"**鐢ㄦ埛:** {user_question}")
                    with st.chat_message("assistant"):
                        st.write(f"**鍔╂墜:** {answer}")
                except Exception as e:
                    st.error(f"鍥炵瓟澶辫触: {e}")

if __name__ == "__main__":
    main()