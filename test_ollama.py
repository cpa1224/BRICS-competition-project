try:
    from langchain_ollama import OllamaLLM
    
    print("=== Ollama API测试 ===")
    
    llm = OllamaLLM(model="deepseek-r1:7b")
    
    test_prompt = "请用一句话介绍自然语言处理。"
    print(f"测试提示: {test_prompt}")
    
    response = llm.invoke(test_prompt)
    print(f"模型响应: {response}")
    
except ImportError:
    print("错误：langchain-ollama 模块未安装")
except Exception as e:
    print(f"错误: {e}")
    print("请确保Ollama服务已启动，并且已下载deepseek-r1:7b模型")