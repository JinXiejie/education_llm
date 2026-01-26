
import gradio as gr
import ollama
from typing import Generator, List, Dict, Optional, Any
import os, requests, hashlib, json
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from conductive_edu.backend_serve.ollama_streaming import format_history_for_ollama
from conductive_edu.backend_serve.utils.doc_processor import DocumentProcessor
from conductive_edu.backend_serve.utils.db_processor import DBProcessor
from conductive_edu.config import Config
from conductive_edu.frontend_controller.gradio_streaming_ui import GradioStreamingUI


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
        self.system_prompt = Config.SYSTEM_PROMPT
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap



        # 初始化文件处理器
        self.doc_processor = DocumentProcessor()

        # 初始化 Ollama 模型
        self.llm_model = "deepseek-r1:8b"
        # self.llm_model = "deepseek-r1:1.5b"

        # 创建存储目录
        os.makedirs(persist_directory, exist_ok=True)
        # os.makedirs("./uploaded_docs", exist_ok=True)

        # 加载现有的向量数据库
        self.db_processor = DBProcessor()
        # self._load_vectorstore()



    def generate_answer(self,
                        message: str,
                        history: List,
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
        print(f"发送给 Ollama 的上下文消息: {json.dumps(context_docs, indent=2, ensure_ascii=False)}")
        context = "\n\n".join([
            f"[文档 {i + 1} - {doc['source']}]\n{doc['content']}"
            for i, doc in enumerate(context_docs)
        ])

        """
        流式生成响应
        """


        # 2.如果上下文信息不足以回答问题，请说明"根据提供的信息，无法回答这个问题"
        print(f"生成答案，使用模型: {model}")
        print(f"查询: {message}")
        print(f"使用 {len(context_docs)} 个文档作为上下文")

        # 添加系统提示
        # 转换历史记录格式
        system_messages = [{"role": "system", "content": self.system_prompt}]
        ollama_messages = format_history_for_ollama(history, system_messages)

        # 添加当前用户消息
        message = str(message).strip()
        if not message:
            yield "错误：消息为空"
            return

        # 构建提示词
        current_msg = f"""基于以下上下文信息，请回答问题：
                上下文信息：{context}
                问题：{message}
                
                要求：
                1. 只基于提供的上下文信息回答问题
                2. 在答案末尾，用【来源】标签注明参考的文档编号
                3. 请使用中文回答

                请基于以上信息回答："""

        ollama_messages.append({"role": "user", "content": current_msg})

        print(f"发送给 Ollama 的消息: {json.dumps(ollama_messages, indent=2, ensure_ascii=False)}")

        # 调用 Ollama
        try:
            # messages = [
            #     {"role": "system", "content": "你是一个基于文档内容回答问题的助手。"},
            #     {"role": "user", "content": prompt}
            # ]

            if stream:
                # 流式生成
                stream = ollama.chat(
                    model=model,
                    messages=ollama_messages,
                    stream=True
                )
                full_response = ""
                for chunk in stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        token = chunk['message']['content']
                        full_response += token
                        yield full_response
            else:
                # 非流式生成
                response = ollama.chat(
                    model=model,
                    messages=ollama_messages,
                    stream=False
                )
                yield response['message']['content']

        except Exception as e:
            print(f"生成答案失败: {e}")
            yield f"生成答案时出错: {str(e)}"

    def rag_query(self, query: str, history: List, k: int = 4, model: Optional[str] = None) -> Generator[List, None, None]:

        """
                处理用户消息 - 返回 Gradio Chatbot 期望的格式

                Gradio Chatbot 期望: [[user_msg1, assistant_msg1], [user_msg2, assistant_msg2], ...]
                """
        print(f"respond: 收到消息: {query}")
        print(f"respond: 当前历史: {history}")

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
        context_docs = self.doc_processor.search_documents(query, k=k)

        # if not context_docs:
        #     return {
        #         "query": query,
        #         "context_docs": [],
        #         "answer": "未找到相关文档信息。",
        #         "sources": []
        #     }


        if not history:
            history.append({"role": "assistant", "content": ""})

        if not query or not str(query).strip():
            # 返回当前历史（不添加新消息）
            yield history
            return

        # 创建一个新的历史记录副本
        new_history = history.copy() if history else []

        # 添加用户消息（助手消息暂时为空）
        new_history.append({"role": "user", "content": query})

        # 格式化文档显示
        formatted_docs = []
        for i, doc in enumerate(context_docs, 1):
            formatted_doc = {
                "序号": i,
                "来源": doc["source"],
                "相似度": f"{doc['score']:.4f}",
                "内容预览": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
            }
            formatted_docs.append(formatted_doc)


        # 2. 生成答案
        try:
            previous_history = history if history else []
            new_history.append({"role": "assistant", "content": ""})
            for response_chunk in self.generate_answer(query, previous_history, context_docs, model, stream=False):
                full_response = response_chunk
                print(full_response)
                # 更新最后一条消息的助手回复
                new_history[-1] = {"role": "assistant", "content": full_response + str(formatted_docs)}
                yield new_history
        except Exception as e:
            print(f"响应生成错误: {e}")
            new_history[-1][1] = f"错误: {str(e)}"
            yield new_history
        # for response_chunk in self.generate_answer(query, context_docs, previous_history, model, stream=False):
        #     answer = response_chunk
        #     yield answer
        #     # print(answer)

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



        # rag_docs = {"role": "assistant", "content": formatted_docs}
        # yield new_history, rag_docs

        # return {
        #     "query": query,
        #     "context_docs": context_docs,
        #     "answer": answer,
        #     "sources": sources
        # }



class RAGGradioInterface:
    """RAG 系统的 Gradio 界面"""

    def __init__(self):
        # 初始化 RAG 系统
        print("\n初始化 RAG 系统...")
        rag_system = LocalRAGSystem(
            embedding_model=Config.EMBEDDING_MODEL_NAME,
            persist_directory=Config.PERSIST_DIR,
            chunk_size=800,
            chunk_overlap=150
        )
        self.rag_system = rag_system
        self.current_model = rag_system.llm_model
        self.available_models = self.get_available_models()
        print("可用的模型列表:" + str(self.available_models))
        self.default_model = "deepseek-r1:8b"
        self.llm_model = self.get_llm_model(Config.LLM_MODEL_NAME)
        # 初始化文件处理器
        self.doc_processor = DocumentProcessor()
        self.db_processor = DBProcessor()

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            models = ollama.list()
            return [model['model'] for model in models['models']]
        except:
            print("未获取有效模型列表")
            return ["llama2", "mistral", "codellama"]  # 默认模型

    def get_llm_model(self, llm_model_name):
        print('-'*8 + '初始化教育大模型：' + '-'*8)
        for model in self.available_models:
            if model == llm_model_name:
                return model
        print('未找到教育大模型：')
        print('采用默认模型：' + self.default_model)
        return self.default_model
    # def get_available_models(self) -> List[str]:
    #     """获取可用模型列表"""
    #     try:
    #         response = requests.get("http://localhost:11434/api/tags", timeout=5)
    #         if response.status_code == 200:
    #             data = response.json()
    #             models = data.get('models', [])
    #             return [model['model'] for model in models]
    #     except Exception as e:
    #         print(f"获取模型列表失败: {e}")
    #     return ["deepseek-r1:8b"]

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

                    model_dropdown = gr.Dropdown(
                        choices=self.available_models,
                        value=self.llm_model,
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

                    # 聊天界面
                    chatbot = gr.Chatbot(
                        label="聊天记录",
                        height=400,
                        avatar_images=(
                            "👤",  # 用户头像
                            "🤖"  # 助手头像
                        )
                    )

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
                    success = self.doc_processor.add_local_documents(file_paths)

                    if success:
                        stats = self.doc_processor.get_document_stats()
                        return f"✅ 成功处理 {len(files)} 个文档", stats
                    else:
                        return "❌ 处理文档失败", {}

                except Exception as e:
                    return f"❌ 上传失败: {str(e)}", {}

            def get_stats():
                """获取文档统计"""
                stats = self.doc_processor.get_document_stats()
                return stats

            def clear_database():
                """清空数据库"""
                success = self.db_processor.clear_database()
                if success:
                    return "✅ 数据库已清空", {}
                else:
                    return "❌ 清空数据库失败", {}

            def process_query(query, k, model):
                """处理查询"""
                if not query.strip():
                    yield "请输入问题", "答案将在这里显示...", []
                    return

                if not self.doc_processor.get_document_stats()["total_documents"]:
                    yield "❌ 请先上传文档", "数据库中没有文档，请先上传文档。", []
                    return

                # 执行 RAG 查询
                yield "🔍 正在检索相关文档...", "正在检索相关文档...", []

                result = self.rag_system.rag_query(query, [], k=k, model=model)

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
            # def handle_query(query, k, model):
            #     for status, answer, docs in process_query(query, k, model):
            #         yield status, answer, docs

            query_input.submit(
                self.rag_system.rag_query,
                inputs=[query_input, chatbot, k_slider, model_dropdown],
                outputs=[chatbot]
                # outputs=[status_display, chatbot]
            )

            submit_btn.click(
                self.rag_system.rag_query,
                inputs=[query_input, chatbot, k_slider, model_dropdown],
                outputs=[chatbot]
                # outputs=[status_display, chatbot]
            )

            # query_input.submit(
            #     handle_query,
            #     inputs=[query_input, k_slider, model_dropdown],
            #     outputs=[status_display, answer_output, docs_output]
            # )

            # submit_btn.click(
            #     handle_query,
            #     inputs=[query_input, k_slider, model_dropdown],
            #     outputs=[status_display, answer_output, docs_output]
            # )

            # 模型切换
            model_dropdown.change(
                lambda m: f"已切换到模型: {m}",
                inputs=[model_dropdown],
                outputs=[status_display]
            )

        return demo


class OllamaDocumentQA:
    """Ollama 文档问答系统"""

    def __init__(self):
        self.document_text = ""
        self.chat_history = []

    def stream_document_qa(self, question: str, document: str,
                           model_name: str, history: List) -> Generator[str, None, None]:
        """
        基于文档的流式问答
        """
        if not document.strip():
            yield "请先上传或输入文档内容！"
            return

        # 构建提示
        prompt = f"""
        基于以下文档内容回答问题：

        文档内容：
        {document}

        问题：{question}

        请根据文档内容直接回答：
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        # 流式调用
        stream = ollama.chat(
            model=model_name,
            messages=messages,
            stream=True
        )

        response = ""
        for chunk in stream:
            content = chunk['message']['content']
            response += content
            yield response

    def create_document_qa_interface(self):
        """创建文档问答界面"""

        with gr.Blocks(title="Ollama 文档问答系统") as demo:
            gr.Markdown("""
            # 📄 Ollama 文档问答系统
            上传文档或输入文本内容，AI 将基于文档内容回答问题
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        # choices=["llama2", "mistral", "codellama"],
                        choices=get_available_models(),
                        value="deepseek-r1:1.5b",
                        label="选择模型"
                    )

                    gr.Markdown("### 文档输入")
                    document_input = gr.Textbox(
                        label="文档内容",
                        placeholder="在此粘贴或输入文档内容...",
                        lines=15
                    )

                    upload_btn = gr.UploadButton(
                        "📎 上传文档文件",
                        file_types=[".txt", ".pdf", ".docx", ".md"]
                    )

                    clear_doc_btn = gr.Button("清空文档", variant="secondary")

                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(
                        label="问答记录",
                        height=400
                    )

                    question_input = gr.Textbox(
                        label="输入问题",
                        placeholder="基于文档内容提问...",
                        lines=2
                    )

                    with gr.Row():
                        submit_btn = gr.Button("提问", variant="primary")
                        clear_chat_btn = gr.Button("清空对话", variant="secondary")

                    gr.Examples(
                        examples=[
                            ["文档的主要内容是什么？"],
                            ["总结一下文档的关键点"],
                            ["根据文档，最核心的观点是什么？"]
                        ],
                        inputs=question_input,
                        label="示例问题"
                    )

            # 绑定事件
            def handle_upload(file):
                if file:
                    try:
                        # 简单的文本文件读取
                        with open(file.name, 'r', encoding='utf-8') as f:
                            content = f.read()
                        return content
                    except:
                        return "无法读取文件，请确保是文本文件"
                return ""

            upload_btn.upload(
                handle_upload,
                [upload_btn],
                [document_input]
            )

            submit_btn.click(
                self.stream_document_qa,
                [question_input, document_input, model_dropdown, chatbot],
                [chatbot],
                show_progress=True
            ).then(
                lambda: "",
                None,
                [question_input]
            )

            clear_doc_btn.click(
                lambda: "",
                None,
                [document_input]
            )

            clear_chat_btn.click(
                lambda: [],
                None,
                [chatbot]
            )

        return demo
