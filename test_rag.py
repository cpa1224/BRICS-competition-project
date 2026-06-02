import os
import sys
from utils.document_loader import process_documents, split_text
from utils.vector_db import create_vector_db
from utils.rag_chain import create_rag_chain, ask_question

def main():
    print("=== RAG闂瓟绯荤粺娴嬭瘯 ===")
    
    documents_folder = "./docs"
    persist_directory = "./chroma_db"
    
    if not os.path.exists(documents_folder):
        print(f"閿欒锛氭枃妗ｆ枃浠跺す {documents_folder} 涓嶅瓨鍦紒")
        sys.exit(1)
    
    print(f"姝ｅ湪澶勭悊鏂囨。...")
    all_text, file_count = process_documents(documents_folder)
    print(f"宸插鐞?{file_count} 涓枃妗?)
    
    if not all_text:
        print("閿欒锛氭湭鎵惧埌浠讳綍鏈夋晥鏂囨。鍐呭锛?)
        sys.exit(1)
    
    print(f"\n姝ｅ湪鍒嗗壊鏂囨湰...")
    chunks = split_text(all_text, chunk_size=1000, chunk_overlap=200)
    print(f"鏂囨湰宸插垎鍓蹭负 {len(chunks)} 涓潡")
    
    print(f"\n姝ｅ湪鍒涘缓鍚戦噺鏁版嵁搴?..")
    vectordb = create_vector_db(chunks, persist_directory)
    print("鍚戦噺鏁版嵁搴撳垱寤哄畬鎴?)
    
    print(f"\n姝ｅ湪鍒濆鍖朢AG闂瓟閾?..")
    chain = create_rag_chain(vectordb)
    print("RAG闂瓟閾惧垵濮嬪寲瀹屾垚")
    
    print("\n=== 娴嬭瘯闂瓟 ===")
    test_questions = [
        "浠€涔堟槸鑷劧璇█澶勭悊锛?,
        "Transformer妯″瀷鐨勪富瑕佺壒鐐规槸浠€涔堬紵",
        "浠€涔堟槸璇嶅祵鍏ワ紵",
        "BERT妯″瀷鏈変粈涔堝垱鏂帮紵",
        "鏂囨湰鍒嗙被鐨勫父鐢ㄦ柟娉曟湁鍝簺锛?,
        "浜哄伐鏅鸿兘鐨勬湭鏉ュ彂灞曡秼鍔挎槸浠€涔堬紵",
        "閲忓瓙璁＄畻鐨勫師鐞嗘槸浠€涔堬紵"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n闂 {i}: {question}")
        try:
            answer = ask_question(chain, question)
            print(f"鍥炵瓟: {answer}")
        except Exception as e:
            print(f"鍥炵瓟澶辫触: {e}")
    
    print("\n=== 浜や簰寮忛棶绛?===")
    print("杈撳叆 'quit' 鎴?'exit' 閫€鍑?)
    while True:
        try:
            question = input("\n璇疯緭鍏ラ棶棰? ")
            if question.lower() in ["quit", "exit"]:
                break
            answer = ask_question(chain, question)
            print(f"鍥炵瓟: {answer}")
        except Exception as e:
            print(f"鍥炵瓟澶辫触: {e}")

if __name__ == "__main__":
    main()