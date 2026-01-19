import os
import json
import hashlib
import chromadb
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime
import gradio as gr
import ollama
import requests
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader
)

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from conductive_edu.config import Config


class LocalRAGSystem:
    """本地 RAG 系统"""

    def __init__(self,
                 # embedding_model: str = "all-MiniLM-L6-v2",
                 embedding_model: str = Config.EMBEDDING_MODEL_NAME,
                 # persist_directory: str = "./chroma_db",
                 persist_directory: str = Config.PERSIST_DIR,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        初始化 RAG 系统

        Args:
            embedding_model: 嵌入模型名称
            persist_directory: 向量数据库存储目录
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
        """
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化嵌入模型
        print(f"加载嵌入模型: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""]
        )

        # 初始化向量数据库
        self.vectorstore = None
        self.collection_name = "rag_documents"

        # 初始化 Ollama 模型
        self.llm_model = "deepseek-r1:8b"

        # 创建存储目录
        os.makedirs(persist_directory, exist_ok=True)
        # os.makedirs("./uploaded_docs", exist_ok=True)

        # 加载现有的向量数据库
        self._load_vectorstore()

    def _load_vectorstore(self):
        """加载向量数据库"""
        try:
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                print("加载现有的向量数据库...")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                print(f"向量数据库加载成功，包含 {self.vectorstore._collection.count()} 个文档")
            else:
                print("创建新的向量数据库...")
                self.vectorstore = Chroma.from_documents(
                    # documents=
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
        except Exception as e:
            print(f"加载向量数据库失败: {e}")
            self.vectorstore = Chroma.from_documents(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )

    def _get_file_loader(self, file_path: str):
        """根据文件类型获取相应的加载器"""
        ext = os.path.splitext(file_path)[1].lower()

        loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader,
            '.csv': CSVLoader
        }

        if ext in loaders:
            return loaders[ext](file_path)
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

    def process_document(self, file_path: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        处理单个文档

        Args:
            file_path: 文档路径
            metadata: 文档元数据

        Returns:
            处理后的文档块列表
        """
        print(f"处理文档: {file_path}")

        try:
            # 加载文档
            loader = self._get_file_loader(file_path)
            documents = loader.load()

            # 添加元数据
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)

            # 文本分割
            texts = self.text_splitter.split_documents(documents)

            print(f"文档分割为 {len(texts)} 个块")
            return texts

        except Exception as e:
            print(f"处理文档失败: {e}")
            raise

    def add_local_documents(self, file_paths: List[str], metadata: Optional[Dict] = None) -> bool:
        """
        添加文档到向量数据库

        Args:
            file_paths: 文档路径列表
            metadata: 文档元数据

        Returns:
            是否成功
        """
        try:
            all_texts = []

            for file_path in file_paths:
                if not os.path.exists(file_path):
                    print(f"文件不存在: {file_path}")
                    continue

                # 生成文档ID
                file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]

                doc_metadata = {
                    "source": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_type": os.path.splitext(file_path)[1],
                    "file_size": os.path.getsize(file_path),
                    "hash": file_hash,
                    "upload_time": datetime.now().isoformat()
                }

                if metadata:
                    doc_metadata.update(metadata)

                # 处理文档
                texts = self.process_document(file_path, doc_metadata)
                all_texts.extend(texts)

            if all_texts:
                # 添加到向量数据库
                self.vectorstore.add_documents(all_texts)
                # self.vectorstore.persist()

                print(f"成功添加 {len(all_texts)} 个文档块到向量数据库")
                return True
            else:
                print("没有有效的文档可以添加")
                return False

        except Exception as e:
            print(f"添加文档失败: {e}")
            return False

    def search_documents(self, query: str, k: int = 4) -> List[Dict]:
        """
        搜索相关文档

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            相关文档列表
        """
        if not self.vectorstore:
            print("向量数据库未初始化")
            return []

        try:
            # 相似度搜索
            docs = self.vectorstore.similarity_search_with_score(query, k=k)

            results = []
            for doc, score in docs:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "source": doc.metadata.get("source", "unknown")
                }
                results.append(result)

            print(f"搜索到 {len(results)} 个相关文档")
            return results

        except Exception as e:
            print(f"搜索文档失败: {e}")
            return []

    def generate_answer(self,
                        query: str,
                        context_docs: List[Dict],
                        model: Optional[str] = None,
                        stream: bool = True) -> Generator[str, None, None]:
        """
        基于检索到的文档生成答案

        Args:
            query: 用户问题
            context_docs: 相关文档列表
            model: Ollama 模型名称
            stream: 是否流式输出

        Returns:
            生成器，逐步返回答案
        """
        if not model:
            model = self.llm_model

        # 构建上下文
        context = "\n\n".join([
            f"[文档 {i + 1} - {doc['source']}]\n{doc['content']}"
            for i, doc in enumerate(context_docs)
        ])

        # 构建提示词
        prompt = f"""基于以下上下文信息，请回答问题：

            上下文信息：
            {context}
            
            问题：{query}
            
            要求：
            1. 只基于提供的上下文信息回答问题
            
            3. 在答案末尾，用【来源】标签注明参考的文档编号
            4. 请使用中文回答
            
            请基于以上信息回答："""
# 2.如果上下文信息不足以回答问题，请说明"根据提供的信息，无法回答这个问题"
        print(f"生成答案，使用模型: {model}")
        print(f"查询: {query}")
        print(f"使用 {len(context_docs)} 个文档作为上下文")

        # 调用 Ollama
        try:
            messages = [
                {"role": "system", "content": "你是一个基于文档内容回答问题的助手。"},
                {"role": "user", "content": prompt}
            ]

            if stream:
                # 流式生成
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    stream=True
                )

                full_response = ""
                for chunk in response:
                    if 'message' in chunk and 'content' in chunk['message']:
                        token = chunk['message']['content']
                        full_response += token
                        yield full_response
            else:
                # 非流式生成
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    stream=False
                )
                yield response['message']['content']

        except Exception as e:
            print(f"生成答案失败: {e}")
            yield f"生成答案时出错: {str(e)}"

    def rag_query(self, query: str, k: int = 4, model: Optional[str] = None) -> Dict:
        """
        完整的 RAG 查询流程

        Args:
            query: 用户问题
            k: 检索文档数量
            model: Ollama 模型名称

        Returns:
            RAG 查询结果
        """
        print(f"\n{'=' * 60}")
        print(f"RAG 查询: {query}")
        print(f"{'=' * 60}")

        # 1. 检索相关文档
        context_docs = self.search_documents(query, k=k)

        if not context_docs:
            return {
                "query": query,
                "context_docs": [],
                "answer": "未找到相关文档信息。",
                "sources": []
            }

        # 2. 生成答案
        answer = ""
        for response_chunk in self.generate_answer(query, context_docs, model, stream=False):
            answer = response_chunk

        # 3. 提取来源信息
        sources = []
        for doc in context_docs:
            source_info = {
                "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                "source": doc["source"],
                "score": doc["score"],
                "metadata": doc["metadata"]
            }
            sources.append(source_info)

        return {
            "query": query,
            "context_docs": context_docs,
            "answer": answer,
            "sources": sources
        }

    def get_document_stats(self) -> Dict:
        """获取文档统计信息"""
        if not self.vectorstore:
            return {"total_documents": 0}

        try:
            count = self.vectorstore._collection.count()
            return {"total_documents": count}
        except:
            return {"total_documents": 0}

    def clear_database(self) -> bool:
        """清空向量数据库"""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                self.vectorstore = None

            # 重新创建
            self._load_vectorstore()
            print("向量数据库已清空")
            return True

        except Exception as e:
            print(f"清空数据库失败: {e}")
            return False


class RAGGradioInterface:
    """RAG 系统的 Gradio 界面"""

    def __init__(self, rag_system: LocalRAGSystem):
        self.rag_system = rag_system
        self.current_model = rag_system.llm_model

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                return [model['model'] for model in models]
        except Exception as e:
            print(f"获取模型列表失败: {e}")
        return ["deepseek-r1:8b"]

    def create_interface(self):
        """创建 Gradio 界面"""

        # 自定义 CSS
        custom_css = """
        .rag-container { max-width: 1200px; margin: 0 auto; }
        .document-card { 
            background: #f8f9fa; 
            border-radius: 10px; 
            padding: 15px; 
            margin: 10px 0; 
            border-left: 4px solid #667eea; 
        }
        .source-badge { 
            background: #e9ecef; 
            padding: 3px 8px; 
            border-radius: 12px; 
            font-size: 12px; 
            margin-right: 5px; 
        }
        .similarity-score { 
            color: #6c757d; 
            font-size: 12px; 
            float: right; 
        }
        .answer-container {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        """

        with gr.Blocks(css=custom_css, title="本地 RAG 系统") as demo:
            gr.Markdown("""
            # 📚 本地 RAG 智能问答系统
            基于 Ollama 本地大模型的检索增强生成系统
            """)

            with gr.Row():
                # 左侧：文档管理区域
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 文档管理")

                    # 文件上传
                    file_output = gr.File(
                        label="上传文档",
                        file_count="multiple",
                        file_types=[".txt", ".pdf", ".docx", ".md", ".csv"]
                    )

                    upload_btn = gr.Button("上传并处理文档", variant="primary")
                    clear_db_btn = gr.Button("清空数据库", variant="secondary")

                    # 文档统计
                    with gr.Group():
                        stats_btn = gr.Button("📊 查看统计", variant="secondary")
                        stats_output = gr.JSON(label="文档统计", value={})

                    # 模型选择
                    available_models = self.get_available_models()
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        value=available_models[0] if available_models else "deepseek-r1:8b",
                        label="选择 AI 模型"
                    )

                    # 检索参数
                    with gr.Accordion("⚙️ 检索参数", open=False):
                        k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="检索文档数量"
                        )

                    gr.Markdown("---")
                    gr.Markdown("**支持的文档格式:**")
                    gr.Markdown("- 📝 TXT 文本文件")
                    gr.Markdown("- 📄 PDF 文档")
                    gr.Markdown("- 📘 Word 文档 (.docx)")
                    gr.Markdown("- 📋 Markdown 文件")
                    gr.Markdown("- 📊 CSV 表格")

                # 右侧：问答区域
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 智能问答")

                    # 问答输入
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="输入问题",
                            placeholder="请输入您的问题...",
                            lines=2,
                            scale=4
                        )
                        submit_btn = gr.Button("🔍 搜索并回答", variant="primary", scale=1)

                    # 状态显示
                    status_display = gr.Textbox(
                        label="状态",
                        value="就绪",
                        interactive=False
                    )

                    # 答案显示
                    answer_output = gr.Markdown(
                        label="AI 答案",
                        value="答案将在这里显示..."
                    )

                    # 检索到的文档
                    with gr.Accordion("📖 检索到的相关文档", open=False):
                        docs_output = gr.JSON(label="相关文档", value=[])

                    # 示例问题
                    gr.Examples(
                        examples=[
                            ["文档的主要内容是什么？"],
                            ["总结一下文档的关键点"],
                            ["文档中提到了哪些重要概念？"],
                            ["根据文档，最核心的观点是什么？"]
                        ],
                        inputs=query_input,
                        label="💡 示例问题"
                    )

            # ========== 事件处理 ==========

            def upload_and_process(files):
                """上传并处理文档"""
                if not files:
                    return "请选择要上传的文件", {}

                try:
                    # 保存上传的文件
                    file_paths = []
                    for file in files:
                        # save_path = f"./uploaded_docs/{file.name}"
                        save_path = f"{file.name}"
                        # with open(save_path, "wb") as f:
                        #     f.write(file)
                        file_paths.append(save_path)

                    # 添加到向量数据库
                    success = self.rag_system.add_local_documents(file_paths)

                    if success:
                        stats = self.rag_system.get_document_stats()
                        return f"✅ 成功处理 {len(files)} 个文档", stats
                    else:
                        return "❌ 处理文档失败", {}

                except Exception as e:
                    return f"❌ 上传失败: {str(e)}", {}

            def get_stats():
                """获取文档统计"""
                stats = self.rag_system.get_document_stats()
                return stats

            def clear_database():
                """清空数据库"""
                success = self.rag_system.clear_database()
                if success:
                    return "✅ 数据库已清空", {}
                else:
                    return "❌ 清空数据库失败", {}

            def process_query(query, k, model):
                """处理查询"""
                if not query.strip():
                    yield "请输入问题", "答案将在这里显示...", []
                    return

                if not self.rag_system.get_document_stats()["total_documents"]:
                    yield "❌ 请先上传文档", "数据库中没有文档，请先上传文档。", []
                    return

                # 执行 RAG 查询
                yield "🔍 正在检索相关文档...", "正在检索相关文档...", []

                result = self.rag_system.rag_query(query, k=k, model=model)

                # 格式化答案
                answer = result["answer"]

                # 格式化文档显示
                formatted_docs = []
                for i, doc in enumerate(result["context_docs"], 1):
                    formatted_doc = {
                        "序号": i,
                        "来源": doc["source"],
                        "相似度": f"{doc['score']:.4f}",
                        "内容预览": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
                    }
                    formatted_docs.append(formatted_doc)

                yield "✅ 查询完成", answer, formatted_docs

            # ========== 绑定事件 ==========

            # 文档管理事件
            upload_btn.click(
                upload_and_process,
                inputs=[file_output],
                outputs=[status_display, stats_output]
            )

            stats_btn.click(
                get_stats,
                outputs=[stats_output]
            )

            clear_db_btn.click(
                clear_database,
                outputs=[status_display, stats_output]
            )

            # 问答事件
            def handle_query(query, k, model):
                for status, answer, docs in process_query(query, k, model):
                    yield status, answer, docs

            query_input.submit(
                handle_query,
                inputs=[query_input, k_slider, model_dropdown],
                outputs=[status_display, answer_output, docs_output]
            )

            submit_btn.click(
                handle_query,
                inputs=[query_input, k_slider, model_dropdown],
                outputs=[status_display, answer_output, docs_output]
            )

            # 模型切换
            model_dropdown.change(
                lambda m: f"已切换到模型: {m}",
                inputs=[model_dropdown],
                outputs=[status_display]
            )

        return demo


def main():
    """主函数"""
    print("=" * 60)
    print("本地 RAG 系统启动")
    print("=" * 60)

    # 检查 Ollama 服务
    try:
        response = requests.get("http://localhost:11434/", timeout=5)
        print("✅ Ollama 服务正在运行")
    except Exception as e:
        print(f"❌ 无法连接到 Ollama: {e}")
        print("\n请确保:")
        print("1. Ollama 已安装 (https://ollama.ai)")
        print("2. 运行: ollama serve")
        print("3. 下载模型: ollama pull deepseek-r1:8b")
        return

    # 初始化 RAG 系统
    print("\n初始化 RAG 系统...")
    rag_system = LocalRAGSystem(
        embedding_model="all-MiniLM-L6-v2",
        persist_directory="./chroma_db",
        chunk_size=800,
        chunk_overlap=150
    )

    # 创建 Gradio 界面
    print("创建 Gradio 界面...")
    interface = RAGGradioInterface(rag_system)
    demo = interface.create_interface()

    # 启动界面
    print("\n启动 Web 界面...")
    print("访问地址: http://127.0.0.1:7860")
    print("=" * 60)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    # 安装必要的依赖
    # pip install chromadb langchain sentence-transformers pypdf docx2txt markdown unstructured[md] gradio ollama requests

    main()