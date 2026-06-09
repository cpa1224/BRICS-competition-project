import os
import sys
from utils.document_loader import process_documents, split_text
from utils.vector_db import create_vector_db
from utils.rag_chain import create_rag_chain, ask_question

def main():
    print("=== RAG问答系统测试 ===")
    
    documents_folder = "./docs"
    persist_directory = "./chroma_db"
    
    if not os.path.exists(documents_folder):
        print(f"错误：文档文件夹 {documents_folder} 不存在！")
        sys.exit(1)
    
    print(f"正在处理文档...")
    all_text, file_count = process_documents(documents_folder)
    print(f"已处理 {file_count} 个文档")
    
    if not all_text:
        print("错误：未找到任何有效文档内容！")
        sys.exit(1)
    
    print(f"\n正在分割文本...")
    chunks = split_text(all_text, chunk_size=1000, chunk_overlap=200)
    print(f"文本已分割为 {len(chunks)} 个块")
    
    print(f"\n正在创建向量数据库...")
    vectordb = create_vector_db(chunks, persist_directory)
    print("向量数据库创建完成")
    
    print(f"\n正在初始化RAG问答链...")
    chain = create_rag_chain(vectordb)
    print("RAG问答链初始化完成")
    
    print("\n=== 测试问答 ===")
    test_questions = [
        "什么是自然语言处理？",
        "Transformer模型的主要特点是什么？",
        "什么是词嵌入？",
        "BERT模型有什么创新？",
        "文本分类的常用方法有哪些？",
        "人工智能的未来发展趋势是什么？",
        "量子计算的原理是什么？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        try:
            answer = ask_question(chain, question)
            print(f"回答: {answer}")
        except Exception as e:
            print(f"回答错误: {e}")
    
    print("\n=== 交互式问答 ===")
    print("输入 'quit' 或 'exit' 退出")
    while True:
        try:
            question = input("\n请输入问题: ")
            if question.lower() in ["quit", "exit"]:
                break
            answer = ask_question(chain, question)
            print(f"回答: {answer}")
        except Exception as e:
            print(f"回答错误: {e}")

if __name__ == "__main__":
    main()