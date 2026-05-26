from utils.document_loader import load_documents_from_folder
from utils.vector_db import VectorDB
from utils.rag_chain import RAGChain

def main():
    print("加载文档...")
    documents = load_documents_from_folder("./docs")
    
    if not documents:
        print("未找到任何文档，请先在docs目录中添加PDF或DOCX文件")
        return
    
    print(f"成功加载 {len(documents)} 个文档")
    
    print("初始化向量数据库...")
    vector_db = VectorDB(use_ollama=False)
    vector_db.add_documents(documents)
    
    print(f"向量数据库中共有 {vector_db.get_collection_stats()} 个文本块")
    
    print("初始化RAG问答链...")
    retriever = vector_db.get_retriever(k=3)
    rag_chain = RAGChain(retriever, model_name="deepseek-r1:7b")
    
    test_questions = [
        "什么是自然语言处理？",
        "Transformer模型的主要特点是什么？",
        "BERT模型是如何预训练的？",
        "词嵌入技术有哪些应用？",
        "文本分类的常用方法有哪些？",
        "今天天气怎么样？",
        "如何制作蛋糕？"
    ]
    
    print("\n开始测试问答效果：")
    for question in test_questions:
        print(f"\n问题：{question}")
        try:
            answer = rag_chain.ask(question)
            print(f"回答：{answer}")
        except Exception as e:
            print(f"回答失败：{e}")

if __name__ == "__main__":
    main()