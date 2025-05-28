## 🧠 大语言模型学习路径（LLM Learning Map）

### 🧩 1. 基础知识（Foundation）

- 语言模型原理（LM, MLM, CLM）
- Transformer 架构（Attention, Position Encoding 等）
- Token 与 Tokenizer ✅ **← 放在此处**
  - 什么是 Tokenizer
  - 常见分词算法（BPE、WordPiece、SentencePiece）
  - Tokenizer 的训练方法
  - 不同模型的 Tokenizer 差异（如 GPT vs T5）

------

### 🏗️ 2. 模型训练（Model Training）

- 预训练 vs 微调（Pretraining vs Fine-tuning）✅ **← Fine-tuning 放这里**
- 自监督学习目标（Causal LM, Masked LM）
- 多语言/多任务训练
- 数据清洗与构造
- 分布式训练（DeepSpeed, FSDP）

------

### 🧪 3. 模型增强与优化（Enhancement & Alignment）

- RLHF（人类反馈强化学习）
- Prompt Engineering
- Instruction Tuning / SFT
- 模型对齐（Alignment）
- MoE / LoRA / QLoRA（参数高效微调）

------

### 🧠 4. 知识增强（Retrieval-Augmented Generation, RAG）✅ **← RAG 放这里**

- 向量数据库基础（Faiss, Milvus, Chroma）
- Embedding 技术（OpenAI, BGE, E5）
- 检索+生成架构（RAG, HyDE, Fusion-in-Decoder）
- 多文档问答、知识库问答

------

### 🚀 5. 推理与部署（Inference & Deployment）

- 推理优化（KV Cache, Quantization, FlashAttention）
- 模型压缩与量化（GPTQ, AWQ）
- 多 GPU 推理框架（vLLM, TGI, vLLM + Hugging Face）
- Triton、ONNX、TensorRT

------

### 📦 6. 应用开发（LLM Application）

- LangChain / LlamaIndex 构建流程
- Agent（AutoGPT、CrewAI）
- 多模态应用（文本+图像）
- API 对接（OpenAI, Claude, Ollama）

------

### 🛡️ 7. 安全与伦理（Safety & Alignment）

- 有害内容检测
- 红队测试（Red teaming）
- 模型幻觉与纠错机制

------

### 📚 8. 工具与社区资源

- Hugging Face Transformers / Datasets / Accelerate
- OpenLLM / LMDeploy
- 学术论文追踪（arXiv / PapersWithCode）

------

## 🗂️ 总结归类示意

| 模块     | 内容示例                          |
| -------- | --------------------------------- |
| 基础原理 | Transformer, Tokenizer, Attention |
| 模型训练 | Fine-tuning, LoRA, 数据构造       |
| 增强方法 | RAG, RLHF, Prompt Tuning          |
| 推理部署 | vLLM, 量化, Triton, 多卡部署      |
| 应用场景 | LangChain, 多模态, Agent          |
| 安全伦理 | 幻觉检测, 内容过滤                |