import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
from datetime import datetime

# 导入必要的库
try:
    import ollama
    import requests
    import gradio as gr
except ImportError:
    print("请安装必要的库：pip install ollama requests gradio")


class LocalEmbeddingModel:
    """完全本地的嵌入模型"""

    def __init__(self, model_name: str = "bge-small-zh-v1.5"):
        """
        初始化本地嵌入模型

        Args:
            model_name: 嵌入模型名称
                - 'bge-small-zh-v1.5': 中文小模型 (推荐)
                - 'text2vec-base-chinese': 中文基础模型
                - 'all-MiniLM-L6-v2': 英文小模型
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        # 模型下载路径
        # self.models_dir = "./models"
        self.models_dir = "G:\PycharmProjects\education_llm\conductive_edu\local_rag\embed_model"
        os.makedirs(self.models_dir, exist_ok=True)

        print(f"初始化本地嵌入模型: {model_name}")
        self._load_model()

    def _download_model_locally(self):
        """从本地或备用源下载模型"""
        print("正在准备嵌入模型...")

        # 检查是否已有本地模型
        model_path = os.path.join(self.models_dir, self.model_name)
        if os.path.exists(model_path):
            print(f"找到本地模型: {model_path}")
            return model_path

        # 如果没有本地模型，提供手动下载指引
        print(f"\n⚠️  未找到本地模型: {self.model_name}")
        print("请按以下步骤操作：")
        print("1. 下载模型文件：")

        model_urls = {
            "bge-small-zh-v1.5": [
                "https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/pytorch_model.bin",
                "https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/config.json",
                "https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/tokenizer.json",
                "https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/vocab.txt"
            ],
            "text2vec-base-chinese": [
                "https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/pytorch_model.bin",
                "https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/config.json",
                "https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/tokenizer.json"
            ]
        }

        if self.model_name in model_urls:
            print(f"模型下载链接：")
            for url in model_urls[self.model_name]:
                print(f"  {url}")

        print("\n2. 将下载的文件放入：")
        print(f"   {model_path}/")
        print("\n3. 或者使用 transformers 自动下载（需要网络）：")
        print("   from transformers import AutoModel, AutoTokenizer")
        print(f"   model = AutoModel.from_pretrained('{self.model_name}')")

        # 创建目录结构
        os.makedirs(model_path, exist_ok=True)

        # 尝试使用 transformers 下载（如果可用且有网络）
        try:
            print("\n尝试自动下载模型...")
            from transformers import AutoModel, AutoTokenizer

            print(f"下载模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=model_path)
            self.model = AutoModel.from_pretrained(self.model_name, cache_dir=model_path)

            # 保存到本地
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            print(f"✅ 模型已下载并保存到: {model_path}")

            return model_path

        except Exception as e:
            print(f"自动下载失败: {e}")
            print("请手动下载模型文件")
            return None

    def _load_model(self):
        """加载模型"""
        try:
            # 方法1：尝试使用 transformers 加载
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                model_path = os.path.join(self.models_dir, self.model_name)

                if os.path.exists(model_path):
                    print(f"从本地加载模型: {model_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self.model = AutoModel.from_pretrained(model_path)
                else:
                    print("本地模型不存在，尝试在线下载...")
                    self._download_model_locally()

                # 设置为评估模式
                if self.model:
                    self.model.eval()
                    print(f"✅ 模型加载成功: {self.model_name}")

            except ImportError:
                print("未安装 transformers，尝试方法2...")

            # 方法2：尝试使用 sentence-transformers（如果可用）
            try:
                if not self.model:
                    from sentence_transformers import SentenceTransformer

                    model_path = os.path.join(self.models_dir, f"{self.model_name}-sentence-transformers")

                    if os.path.exists(model_path):
                        print(f"从本地加载 sentence-transformers 模型: {model_path}")
                        self.model = SentenceTransformer(model_path)
                    else:
                        print(f"下载 sentence-transformers 模型: {self.model_name}")
                        self.model = SentenceTransformer(self.model_name)

                        # 保存到本地
                        self.model.save(model_path)
                        print(f"模型已保存到: {model_path}")

                    print(f"✅ sentence-transformers 模型加载成功")

            except ImportError:
                print("未安装 sentence-transformers，尝试方法3...")

            # 方法3：使用轻量级替代方案
            if not self.model:
                print("使用轻量级词向量替代方案...")
                self._setup_lightweight_embedding()

        except Exception as e:
            print(f"模型加载失败: {e}")
            print("使用简易嵌入方法作为后备方案...")
            self._setup_simple_embedding()

    def _setup_lightweight_embedding(self):
        """设置轻量级嵌入方法"""
        print("初始化轻量级词向量...")

        # 简单的词频向量化
        self.vocab = {}
        self.vocab_size = 10000  # 限制词汇表大小

        # 简单的词向量（随机初始化）
        np.random.seed(42)
        self.word_vectors = np.random.randn(self.vocab_size, 128).astype(np.float32)

        print("✅ 轻量级嵌入模型就绪")

    def _setup_simple_embedding(self):
        """设置最简单的嵌入方法"""
        print("使用简易词袋模型...")
        self.simple_mode = True
        self.word_index = {}
        self.vector_size = 100

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]

        # 方法1：使用 transformers 模型
        if hasattr(self, 'model') and hasattr(self.model, 'encode'):
            # sentence-transformers 风格
            embeddings = self.model.encode(texts, **kwargs)
            return embeddings

        elif hasattr(self, 'model') and hasattr(self.model, 'eval'):
            # transformers 风格
            import torch
            from transformers import AutoTokenizer, AutoModel

            if not self.tokenizer:
                return np.random.randn(len(texts), 384)

            # 编码文本
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()

            return embeddings

        # 方法2：使用轻量级词向量
        elif hasattr(self, 'word_vectors'):
            return self._lightweight_encode(texts)

        # 方法3：使用最简单的词袋模型
        elif hasattr(self, 'simple_mode'):
            return self._simple_encode(texts)

        # 方法4：随机向量作为后备
        else:
            print("使用随机向量作为嵌入")
            return np.random.randn(len(texts), 384).astype(np.float32)

    def _lightweight_encode(self, texts: List[str]) -> np.ndarray:
        """轻量级编码"""
        embeddings = []

        for text in texts:
            # 简单的词向量平均
            words = text.split()
            word_vecs = []

            for word in words:
                # 简单的哈希函数映射到向量
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab) % self.vocab_size

                idx = self.vocab[word]
                word_vecs.append(self.word_vectors[idx])

            if word_vecs:
                embedding = np.mean(word_vecs, axis=0)
            else:
                embedding = np.zeros(128)

            embeddings.append(embedding)

        return np.array(embeddings)

    def _simple_encode(self, texts: List[str]) -> np.ndarray:
        """最简单的词袋编码"""
        embeddings = []

        for text in texts:
            vector = np.zeros(self.vector_size)
            words = text.split()

            for word in words:
                # 简单的哈希函数
                hash_val = hash(word) % self.vector_size
                vector[hash_val] += 1

            # 归一化
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)

            embeddings.append(vector)

        return np.array(embeddings)


class OfflineRAGSystem:
    """完全离线的 RAG 系统"""

    def __init__(self,
                 embedding_model: str = "bge-small-zh-v1.5",
                 # persist_directory: str = "./offline_rag_db",
                 persist_directory: str = "G:\PycharmProjects\education_llm\conductive_edu\local_rag\offline_rag_db",
                 chunk_size: int = 500,
                 chunk_overlap: int = 100):
        """
        初始化离线 RAG 系统
        """
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化完全本地的嵌入模型
        print("初始化本地嵌入模型...")
        self.embeddings = LocalEmbeddingModel(embedding_model)

        # 初始化文本分割器（使用简单的实现）
        self.text_splitter = self._create_text_splitter()

        # 初始化向量存储（使用简单的实现）
        self.vector_store = {}
        self.documents = []

        # 创建存储目录
        os.makedirs(persist_directory, exist_ok=True)
        os.makedirs("./offline_docs", exist_ok=True)

        # 尝试加载现有的向量存储
        self._load_vector_store()

    def _create_text_splitter(self):
        """创建简单的文本分割器"""

        class SimpleTextSplitter:
            def __init__(self, chunk_size=500, chunk_overlap=100):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text: str) -> List[str]:
                """简单的文本分割"""
                chunks = []
                start = 0
                text_length = len(text)

                while start < text_length:
                    end = start + self.chunk_size

                    # 如果还没到文本末尾，尝试在句子边界处分割
                    if end < text_length:
                        # 找最近的句子结束符
                        for split_char in ['。', '！', '？', '.', '!', '?', '\n']:
                            split_pos = text.rfind(split_char, start, end)
                            if split_pos != -1:
                                end = split_pos + 1
                                break

                    chunk = text[start:end].strip()
                    if chunk:
                        chunks.append(chunk)

                    # 移动起始位置，考虑重叠
                    start = end - self.chunk_overlap
                    if start < 0:
                        start = 0

                return chunks

        return SimpleTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def _load_vector_store(self):
        """加载向量存储"""
        try:
            vector_store_path = os.path.join(self.persist_directory, "vector_store.json")
            documents_path = os.path.join(self.persist_directory, "documents.json")

            if os.path.exists(vector_store_path) and os.path.exists(documents_path):
                print("加载现有的向量存储...")

                with open(documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)

                with open(vector_store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.vector_store = data

                print(f"加载了 {len(self.documents)} 个文档")

        except Exception as e:
            print(f"加载向量存储失败: {e}")
            self.vector_store = {}
            self.documents = []

    def _save_vector_store(self):
        """保存向量存储"""
        try:
            vector_store_path = os.path.join(self.persist_directory, "vector_store.json")
            documents_path = os.path.join(self.persist_directory, "documents.json")

            # 只保存必要的元数据，不保存向量（向量可以重新计算）
            with open(documents_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)

            # 保存向量索引
            vector_data = {
                "metadata": {
                    "total_documents": len(self.documents),
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "last_updated": datetime.now().isoformat()
                }
            }

            with open(vector_store_path, 'w', encoding='utf-8') as f:
                json.dump(vector_data, f, ensure_ascii=False, indent=2)

            print(f"向量存储已保存，共 {len(self.documents)} 个文档")

        except Exception as e:
            print(f"保存向量存储失败: {e}")

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> bool:
        """
        添加文档到系统

        Args:
            text: 文档文本
            metadata: 文档元数据

        Returns:
            是否成功
        """
        if not text or not text.strip():
            return False

        try:
            # 文本分割
            chunks = self.text_splitter.split_text(text)

            # 生成文档ID
            doc_id = len(self.documents)
            doc_hash = hashlib.md5(text.encode()).hexdigest()[:8]

            doc_metadata = {
                "id": doc_id,
                "hash": doc_hash,
                "chunks_count": len(chunks),
                "original_length": len(text),
                "added_time": datetime.now().isoformat()
            }

            if metadata:
                doc_metadata.update(metadata)

            # 添加文档和分块
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                chunk_metadata = {
                    **doc_metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "text": chunk
                }

                self.documents.append(chunk_metadata)

                # 计算嵌入并存储（在实际使用中才计算）
                # 这里我们延迟计算嵌入，只在搜索时计算

            print(f"添加文档成功，分割为 {len(chunks)} 个块")

            # 保存向量存储
            self._save_vector_store()

            return True

        except Exception as e:
            print(f"添加文档失败: {e}")
            return False

    def add_document_from_file(self, file_path: str, metadata: Optional[Dict] = None) -> bool:
        """从文件添加文档"""
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return False

        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                print(f"文件内容为空: {file_path}")
                return False

            # 添加元数据
            file_metadata = {
                "source": os.path.basename(file_path),
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "file_type": os.path.splitext(file_path)[1]
            }

            if metadata:
                file_metadata.update(metadata)

            return self.add_document(text, file_metadata)

        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    text = f.read()
                return self.add_document(text, metadata)
            except:
                print(f"无法读取文件编码: {file_path}")
                return False
        except Exception as e:
            print(f"读取文件失败: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索相关文档

        Args:
            query: 查询文本
            top_k: 返回的文档数量

        Returns:
            相关文档列表
        """
        if not self.documents:
            print("文档库为空")
            return []

        try:
            # 计算查询向量
            print(f"搜索查询: {query}")
            query_vector = self.embeddings.encode([query])[0]

            # 计算每个文档块与查询的相似度
            similarities = []

            for i, doc in enumerate(self.documents):
                # 计算文档向量（延迟计算）
                if "vector" not in doc:
                    # 第一次计算并缓存
                    doc_vector = self.embeddings.encode([doc["text"]])[0]
                    doc["vector"] = doc_vector.tolist() if isinstance(doc_vector, np.ndarray) else doc_vector

                doc_vector = np.array(doc["vector"])

                # 计算余弦相似度
                similarity = self._cosine_similarity(query_vector, doc_vector)
                similarities.append((i, similarity, doc))

            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)

            # 返回 top_k 个结果
            results = []
            for i, (doc_idx, similarity, doc) in enumerate(similarities[:top_k]):
                result = {
                    "rank": i + 1,
                    "similarity": float(similarity),
                    "text": doc["text"],
                    "source": doc.get("source", "unknown"),
                    "chunk_id": doc.get("chunk_id", ""),
                    "metadata": {k: v for k, v in doc.items() if k not in ["text", "vector"]}
                }
                results.append(result)

            print(f"找到 {len(results)} 个相关文档")
            return results

        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except:
            return 0.0

    def query(self,
              question: str,
              top_k: int = 3,
              model: str = "deepseek-r1:8b") -> Dict:
        """
        执行 RAG 查询

        Args:
            question: 用户问题
            top_k: 检索文档数量
            model: Ollama 模型名称

        Returns:
            查询结果
        """
        print(f"\n{'=' * 60}")
        print(f"RAG 查询: {question}")
        print(f"{'=' * 60}")

        # 1. 检索相关文档
        context_docs = self.search(question, top_k=top_k)

        if not context_docs:
            return {
                "question": question,
                "context": [],
                "answer": "未找到相关文档信息。",
                "sources": []
            }

        # 2. 构建上下文
        context_text = "\n\n".join([
            f"[文档 {doc['rank']} - 相似度: {doc['similarity']:.3f}]\n"
            f"来源: {doc['source']}\n"
            f"内容: {doc['text']}"
            for doc in context_docs
        ])

        # 3. 构建提示词
        prompt = f"""基于以下文档内容回答问题：

{context_text}

问题：{question}

要求：
1. 只基于提供的文档内容回答问题
2. 如果文档内容不足以回答问题，请说明"根据提供的文档，无法回答这个问题"
3. 在回答中引用相关的文档编号
4. 请使用中文回答

请基于以上文档内容回答："""

        # 4. 调用本地 Ollama 模型
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )

            answer = response['message']['content']

        except Exception as e:
            print(f"生成答案失败: {e}")
            answer = f"生成答案时出错: {str(e)}"

        # 5. 整理结果
        sources = []
        for doc in context_docs:
            source_info = {
                "rank": doc["rank"],
                "source": doc["source"],
                "similarity": doc["similarity"],
                "content_preview": doc["text"][:100] + "..." if len(doc["text"]) > 100 else doc["text"]
            }
            sources.append(source_info)

        return {
            "question": question,
            "context": context_docs,
            "answer": answer,
            "sources": sources
        }

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        total_chunks = len(self.documents)

        # 统计来源
        sources = {}
        for doc in self.documents:
            source = doc.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1

        return {
            "total_documents": len(set(doc.get("source", "") for doc in self.documents)),
            "total_chunks": total_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "sources": sources
        }

    def clear(self) -> bool:
        """清空系统"""
        try:
            self.vector_store = {}
            self.documents = []

            # 删除存储文件
            vector_store_path = os.path.join(self.persist_directory, "vector_store.json")
            documents_path = os.path.join(self.persist_directory, "documents.json")

            if os.path.exists(vector_store_path):
                os.remove(vector_store_path)
            if os.path.exists(documents_path):
                os.remove(documents_path)

            print("系统已清空")
            return True

        except Exception as e:
            print(f"清空系统失败: {e}")
            return False


# 创建 Gradio 界面
def create_offline_rag_interface():
    """创建离线 RAG 界面"""

    # 初始化离线 RAG 系统
    rag_system = OfflineRAGSystem(
        embedding_model="bge-small-zh-v1.5",
        persist_directory="./offline_rag_db",
        chunk_size=500,
        chunk_overlap=100
    )

    with gr.Blocks(title="离线 RAG 系统") as demo:
        gr.Markdown("""
        # 🔒 离线 RAG 智能问答系统
        **完全本地运行，无需网络连接！**
        """)

        with gr.Row():
            # 左侧：文档管理
            with gr.Column(scale=1):
                gr.Markdown("### 📄 文档管理")

                # 文本输入
                text_input = gr.Textbox(
                    label="输入文本",
                    placeholder="直接粘贴文本内容...",
                    lines=5
                )

                text_submit = gr.Button("添加文本", variant="primary")

                # 文件上传
                file_input = gr.File(
                    label="上传文件",
                    file_count="single",
                    file_types=[".txt", ".md"]
                )

                file_submit = gr.Button("上传文件", variant="primary")

                # 控制按钮
                with gr.Row():
                    stats_btn = gr.Button("📊 统计", variant="secondary")
                    clear_btn = gr.Button("🗑️ 清空", variant="secondary")

                # 统计信息
                stats_output = gr.JSON(label="系统统计", value={})

            # 右侧：问答
            with gr.Column(scale=2):
                gr.Markdown("### 💬 智能问答")

                # 查询输入
                query_input = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入您的问题...",
                    lines=2
                )

                with gr.Row():
                    submit_btn = gr.Button("🔍 搜索并回答", variant="primary")
                    retry_btn = gr.Button("🔄 重新生成", variant="secondary")

                # 答案显示
                answer_output = gr.Markdown(
                    label="AI 答案",
                    value="答案将在这里显示..."
                )

                # 相关文档
                with gr.Accordion("📖 相关文档", open=False):
                    docs_output = gr.JSON(label="检索结果", value=[])

                # 状态
                status_display = gr.Textbox(
                    label="状态",
                    value="就绪",
                    interactive=False
                )

        # ========== 事件处理 ==========

        def add_text(text):
            """添加文本"""
            if not text or not text.strip():
                return "请输入文本内容", {}

            success = rag_system.add_document(text, {"source": "手动输入"})

            if success:
                stats = rag_system.get_stats()
                return "✅ 文本添加成功", stats
            else:
                return "❌ 添加失败", {}

        def add_file(file):
            """添加文件"""
            if not file:
                return "请选择文件", {}

            try:
                # 保存上传的文件
                save_path = f"./offline_docs/{file.name}"
                with open(save_path, "wb") as f:
                    f.write(file)

                success = rag_system.add_document_from_file(save_path)

                if success:
                    stats = rag_system.get_stats()
                    return "✅ 文件添加成功", stats
                else:
                    return "❌ 添加失败", {}

            except Exception as e:
                return f"❌ 错误: {str(e)}", {}

        def get_stats():
            """获取统计"""
            stats = rag_system.get_stats()
            return stats

        def clear_system():
            """清空系统"""
            success = rag_system.clear()
            if success:
                return "✅ 系统已清空", {}
            else:
                return "❌ 清空失败", {}

        def process_query(question):
            """处理查询"""
            if not question or not question.strip():
                return "请输入问题", "答案将在这里显示...", []

            stats = rag_system.get_stats()
            if stats["total_chunks"] == 0:
                return "❌ 请先添加文档", "文档库为空，请先添加文档。", []

            yield "🔍 正在搜索相关文档...", "正在搜索...", []

            result = rag_system.query(question, top_k=3)

            # 格式化结果
            formatted_docs = []
            for doc in result.get("context", []):
                formatted_doc = {
                    "排名": doc["rank"],
                    "来源": doc["source"],
                    "相似度": f"{doc['similarity']:.3f}",
                    "内容": doc["text"]
                }
                formatted_docs.append(formatted_doc)

            yield "✅ 查询完成", result["answer"], formatted_docs

        # ========== 绑定事件 ==========

        # 文档管理
        text_submit.click(
            add_text,
            inputs=[text_input],
            outputs=[status_display, stats_output]
        ).then(
            lambda: "",  # 清空输入
            outputs=[text_input]
        )

        file_submit.click(
            add_file,
            inputs=[file_input],
            outputs=[status_display, stats_output]
        )

        stats_btn.click(
            get_stats,
            outputs=[stats_output]
        )

        clear_btn.click(
            clear_system,
            outputs=[status_display, stats_output]
        )

        # 问答
        def handle_query(question):
            for status, answer, docs in process_query(question):
                yield status, answer, docs

        query_input.submit(
            handle_query,
            inputs=[query_input],
            outputs=[status_display, answer_output, docs_output]
        )

        submit_btn.click(
            handle_query,
            inputs=[query_input],
            outputs=[status_display, answer_output, docs_output]
        )

        # 重新生成
        retry_btn.click(
            lambda q: handle_query(q) if q else (status_display.value, answer_output.value, docs_output.value),
            inputs=[query_input],
            outputs=[status_display, answer_output, docs_output]
        )

    return demo


# 运行系统
if __name__ == "__main__":
    print("=" * 60)
    print("离线 RAG 系统启动")
    print("=" * 60)

    # 检查 Ollama
    try:
        response = requests.get("http://localhost:11434/", timeout=5)
        print("✅ Ollama 服务正在运行")
    except:
        print("⚠️  Ollama 服务未运行")
        print("请先运行: ollama serve")

    print("\n提示：")
    print("1. 首次使用需要下载嵌入模型")
    print("2. 可以手动下载模型到 ./models/ 目录")
    print("3. 系统会自动使用简易嵌入作为后备方案")
    print("=" * 60)

    # 创建并启动界面
    demo = create_offline_rag_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        show_error=True
    )