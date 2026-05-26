import ollama

def test_ollama_connection():
    try:
        response = ollama.chat(model="deepseek-r1:7b", messages=[
            {"role": "user", "content": "Hello, what is natural language processing?"}
        ])
        print("Ollama API测试成功！")
        print("响应内容:", response['message']['content'][:200], "...")
        return True
    except Exception as e:
        print(f"Ollama连接失败: {e}")
        print("请确保Ollama已安装并运行，且已下载deepseek-r1:7b模型")
        return False

if __name__ == "__main__":
    test_ollama_connection()