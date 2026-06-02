try:
    from langchain_ollama import OllamaLLM
    
    print("=== Ollama API娴嬭瘯 ===")
    
    llm = OllamaLLM(model="deepseek-r1:7b")
    
    test_prompt = "璇风敤涓€鍙ヨ瘽浠嬬粛鑷劧璇█澶勭悊銆?
    print(f"娴嬭瘯鎻愮ず璇? {test_prompt}")
    
    response = llm.invoke(test_prompt)
    print(f"妯″瀷鍝嶅簲: {response}")
    
except ImportError:
    print("閿欒锛歭angchain-ollama 妯″潡鏈畨瑁?)
except Exception as e:
    print(f"閿欒: {e}")
    print("璇风‘淇漁llama鏈嶅姟宸插惎鍔紝骞朵笖宸蹭笅杞絛eepseek-r1:7b妯″瀷")