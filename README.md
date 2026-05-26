# RAG问答系统

## 项目简介

基于LangChain和Ollama构建的本地知识库问答系统，支持PDF/DOCX文档上传、文本向量化存储、智能问答等功能。

## 环境要求与安装步骤

### 1. 安装Ollama

访问 [Ollama官方网站](https://ollama.com/) 下载并安装Ollama。

### 2. 下载模型

```bash
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

### 3. 安装Python依赖

```bash
pip install -r requirements.txt
```

## 使用说明

### 运行Web应用

```bash
streamlit run app.py
```

### 上传文档

1. 在左侧面板点击"浏览文件"选择PDF或DOCX文件
2. 点击"构建知识库"按钮处理文档

### 提问

1. 在右侧问答交互区输入问题
2. 点击"提问"按钮获取答案

## 关键技术点

### RAG流程

1. **文档加载**：支持PDF和DOCX格式文档读取
2. **文本分块**：使用RecursiveCharacterTextSplitter，chunk_size=1000，chunk_overlap=200
3. **向量化**：使用SentenceTransformer的all-MiniLM-L6-v2模型
4. **向量存储**：使用Chroma向量数据库
5. **检索**：基于相似度检索相关文本块
6. **生成**：使用Ollama的deepseek-r1:7b模型生成回答

### 模型配置

- **嵌入模型**：all-MiniLM-L6-v2 (SentenceTransformer)
- **LLM模型**：deepseek-r1:7b (Ollama)
- **向量数据库**：Chroma

## 项目结构

```
├── app.py                 # Streamlit Web应用
├── requirements.txt       # 依赖包列表
├── .gitignore            # Git忽略配置
├── test_ollama.py        # Ollama API测试脚本
├── test_rag.py           # RAG问答测试脚本
├── docs/                 # 文档存放目录
│   └── sample_nlp_doc.txt # 示例文档
└── utils/                # 工具函数目录
    ├── document_loader.py # 文档加载工具
    ├── vector_db.py      # 向量数据库操作
    └── rag_chain.py      # RAG问答链
```

## 已知问题与改进方向

### 已知问题

1. Ollama服务需要单独启动
2. 模型下载需要网络连接
3. 大文档处理时间较长

### 改进方向

1. 支持更多文档格式（如TXT、Markdown）
2. 添加文档预处理功能（表格提取、图片OCR等）
3. 优化检索算法
4. 添加模型选择功能

## 许可证

MIT License