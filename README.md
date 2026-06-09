# RAG智能问答系统

基于Ollama本地大模型、LangChain框架和Streamlit构建的智能问答系统，能够"学习"指定文档并回答相关问题。

## 项目功能

- 支持上传PDF、DOCX、TXT等格式文档
- 自动进行文档解析、文本分割和向量索引
- 使用Chroma向量数据库存储文档向量
- 基于检索增强生成(RAG)技术进行问答
- 支持多轮对话，具有会话记忆功能
- 对无关问题能正确拒答

## 环境要求

- Python 3.8+
- Ollama（用于运行本地大模型）
- 至少8GB内存（推荐16GB以上）

## 安装步骤

### 1. 安装Ollama

访问 [Ollama官方网站](https://ollama.com/) 下载并安装Ollama。

### 2. 下载大模型

```bash
# 下载deepseek-r1:7b模型（推荐）
ollama pull deepseek-r1:7b

# 或者下载qwen2:7b模型
ollama pull qwen2:7b

# 下载嵌入模型
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

### 使用流程

1. 在浏览器中打开应用（通常是 http://localhost:8501）
2. 在左侧面板上传PDF、DOCX或TXT格式的文档
3. 点击"构建知识库"按钮，等待文档处理完成
4. 在问答交互区输入问题，点击"提问"按钮获取答案
5. 支持多轮对话，系统会保存对话历史

### 运行测试脚本

```bash
python test_rag.py
```

## 核心技术点

### RAG流程

1. **文档加载**：支持PDF、DOCX、TXT等格式文档的读取
2. **文本分割**：使用RecursiveCharacterTextSplitter进行分割（chunk_size=1000, chunk_overlap=200）
3. **向量索引**：使用llama的nomic-embed-text模型将文本块转换为向量
4. **向量存储**：使用Chroma向量数据库存储和检索向量
5. **问答生成**：使用ConversationalRetrievalChain连接检索器和大模型

### 所用模型

- **大语言模型**：deepseek-r1:7b 或 qwen2:7b
- **嵌入模型**：nomic-embed-text

### 系统提示词设计要求

系统提示词要求模型：
- 基于提供的参考文档回答问题
- 若文档中没有相关信息，明确说"文档中未找到相关答案"
- 不使用文档外的知识
- 保持回答简洁相关

## 项目结构

```
BRICS-competition-project/
├── app.py                 # Streamlit Web应用主文件
├── test_rag.py           # 命令行测试脚本
├── test_ollama.py        # Ollama连接测试
├── requirements.txt      # 依赖列表
├── .gitignore            # Git忽略配置
├── docs/                 # 示例文档目录
│   ├── nlp_introduction.txt
│   ├── transformer.txt
│   ├── word_embedding.txt
│   ├── bert.txt
│   └── text_classification.txt
└── utils/                # 工具模块
    ├── document_loader.py # 文档处理模块
    ├── vector_db.py       # 向量数据库模块
    └── rag_chain.py       # RAG问答链模块
```

## 已知问题与改进方向

### 已知问题

- Ollama服务需要预先启动
- 模型下载需要一定时间和网络带宽
- 首次构建知识库可能需要较长时间

### 改进方向

- 支持更多文档格式（如PPT、Excel）
- 添加文档管理功能（删除、更新文档）
- 支持批量上传文档
- 添加多语言支持
- 支持导出问答记录
- 优化长文档处理能力

## 许可证

MIT License