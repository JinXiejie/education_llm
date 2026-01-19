import gradio as gr
import ollama
import json
import time
from typing import Generator, List, Dict, Any
import threading
import queue


class OllamaStreamingChat:
    """Ollama 流式聊天类"""

    def __init__(self, model_name: str = "deepseek-r1:8b"):
        self.model_name = model_name
        self.history = []

    def stream_response(self, prompt: str, history: List = None,
                        system_prompt: str = None) -> Generator[str, None, None]:
        """
        流式生成响应
        """
        messages = []

        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 添加历史对话
        if history:
            for h in history:
                messages.append({"role": "user", "content": h[0]})
                messages.append({"role": "assistant", "content": h[1]})
                # messages.extend([
                #     {"role": "user", "content": h[0]},
                #     {"role": "assistant", "content": h[1]}
                # ])

        # 添加当前用户输入
        messages.append({"role": "user", "content": prompt})

        # 流式调用 Ollama
        stream = ollama.chat(
            model=self.model_name,
            messages=messages,
            stream=True
        )

        # 流式生成响应
        full_response = ""
        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            yield content

        # 保存到历史记录
        self.history.append((prompt, full_response))

    def clear_history(self):
        """清空历史记录"""
        self.history = []


class GradioStreamingUI:
    """Gradio 流式界面"""

    def __init__(self):
        self.chatbot = OllamaStreamingChat()
        self.available_models = self.get_available_models()

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            models = ollama.list()
            return [model['model'] for model in models['models']]
        except:
            return ["llama2", "mistral", "codellama"]  # 默认模型

    def predict_stream(self, message: str, history: List,
                       model_name: str, temperature: float,
                       max_tokens: int, system_prompt: str):
        """
        Gradio 流式预测函数
        """
        # 更新模型
        if model_name != self.chatbot.model_name:
            self.chatbot = OllamaStreamingChat(model_name)

        # 生成响应
        response = ""
        for chunk in self.chatbot.stream_response(message, history, system_prompt ):
            response += chunk
            yield response

    def format_history_for_ollama(self, gradio_history: List) -> List[Dict[str, str]]:
        """
        将 Gradio 的历史记录格式转换为 Ollama 格式
        """
        messages = []

        if gradio_history and isinstance(gradio_history, list):
            for turn in gradio_history:
                if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    user_msg = turn[0]
                    assistant_msg = turn[1]

                    # 处理用户消息
                    if user_msg is not None:
                        user_msg_str = str(user_msg).strip()
                        if user_msg_str:
                            messages.append({
                                "role": "user",
                                "content": user_msg_str
                            })

                    # 处理助手消息
                    if assistant_msg is not None:
                        assistant_msg_str = str(assistant_msg).strip()
                        if assistant_msg_str:
                            messages.append({
                                "role": "assistant",
                                "content": assistant_msg_str
                            })

        return messages

    def stream_response(self, message: str, history: List, model: str = None) -> Generator[str, None, None]:
        """
        流式生成响应
        """
        if model:
            self.model = model

        # 转换历史记录格式
        ollama_messages = self.format_history_for_ollama(history)

        # 添加当前用户消息
        current_msg = str(message).strip()
        if not current_msg:
            yield "错误：消息为空"
            return

        ollama_messages.append({
            "role": "user",
            "content": current_msg
        })

        print(f"发送给 Ollama 的消息: {json.dumps(ollama_messages, indent=2, ensure_ascii=False)}")

        # 流式生成响应
        try:
            stream = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    token = chunk['message']['content']
                    full_response += token
                    yield full_response

        except Exception as e:
            print(f"Ollama 调用错误: {e}")
            yield f"错误: {str(e)}"

    def respond(self, message: str, history: List,
        model: str, temperature: float,
        max_tokens: int, system_prompt: str) -> Generator[List, None, None]:

        """
        处理用户消息 - 返回 Gradio Chatbot 期望的格式

        Gradio Chatbot 期望: [[user_msg1, assistant_msg1], [user_msg2, assistant_msg2], ...]
        """
        print(f"respond: 收到消息: {message}")
        print(f"respond: 当前历史: {history}")
        if not history:
            history.append({"role": "assistant", "content": ""})
        print(f"respond: 当前历史: {history}")
        if not message or not str(message).strip():
            # 返回当前历史（不添加新消息）
            yield history
            return

        # 创建一个新的历史记录副本
        new_history = history.copy() if history else []

        # 添加用户消息（助手消息暂时为空）
        # new_history.append([message, None])
        new_history.append({"role": "user", "content": message})

        # 生成响应
        full_response = ""
        try:
            # 注意：传递给 stream_response 的是之前的历史（不包含当前消息）
            previous_history = history if history else []
            new_history.append({"role": "assistant", "content": ""})
            for response_chunk in self.stream_response(message, previous_history, model):
                full_response = response_chunk
                # 更新最后一条消息的助手回复
                # new_history[-1][1] = full_response
                new_history[-1] = {"role": "assistant", "content": full_response}
                # print(f"响应结果full_response: {full_response}")
                # new_history.append({"role": "assistant", "content": full_response})
                yield new_history

        except Exception as e:
            print(f"响应生成错误: {e}")
            new_history[-1][1] = f"错误: {str(e)}"
            yield new_history

    def clear_chat(self):
        """清空聊天"""
        self.chatbot.clear_history()
        return [], []  # 返回空的历史和当前状态

    def create_interface(self):
        """创建 Gradio 界面"""

        # 自定义 CSS 样式
        css = """
        .gradio-container {
            max-width: 800px;
            margin: auto;
        }
        .chatbot {
            min-height: 400px;
            border-radius: 10px;
        }
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
        .assistant-message {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
        """

        # 创建主题
        theme = gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="pink",
        )

        with gr.Blocks(theme=theme, css=css, title="Ollama 本地大模型聊天") as demo:
            gr.Markdown("""
            # 🚀 Ollama 本地大模型聊天界面
            与本地部署的 Ollama 大模型进行实时对话，支持流式输出！
            """)

            # 模型选择和参数设置
            with gr.Row():
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=self.available_models,
                        value=self.available_models[3] if self.available_models else "llama2",
                        label="选择模型",
                        interactive=True
                    )

                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="温度 (Temperature)",
                        info="值越高回答越随机"
                    )

                    max_tokens_slider = gr.Slider(
                        minimum=100,
                        maximum=4000,
                        value=2048,
                        step=100,
                        label="最大生成长度"
                    )

                    clear_btn = gr.Button("🧹 清空对话", variant="secondary")

                with gr.Column(scale=3):
                    # 聊天界面
                    chatbot = gr.Chatbot(
                        label="聊天记录",
                        height=400,
                        avatar_images=(
                            "👤",  # 用户头像
                            "🤖"  # 助手头像
                        )
                    )

                    # 系统提示
                    system_prompt_input = gr.Textbox(
                        label="系统提示词",
                        placeholder="例如：你是一个专业的AI助手，请用简洁明了的语言回答问题...",
                        lines=2
                    )

                    # 输入区域
                    with gr.Row():
                        msg = gr.Textbox(
                            label="输入消息",
                            placeholder="请输入您的问题...",
                            scale=4,
                            lines=2
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)

                    # 示例问题
                    gr.Examples(
                        examples=[
                            ["请用简单的语言解释什么是机器学习"],
                            ["写一个Python函数计算斐波那契数列"],
                            ["帮我写一份工作日报的模板"],
                            ["用中文讲一个有趣的笑话"]
                        ],
                        inputs=msg,
                        label="示例问题"
                    )

            # 状态信息
            status_text = gr.Textbox(
                label="状态",
                value="就绪",
                interactive=False
            )

            # 绑定事件
            msg.submit(
                self.respond,
                [msg, chatbot, model_dropdown, temperature_slider, max_tokens_slider, system_prompt_input],
                [chatbot],
                show_progress=True
            ).then(
                lambda: "",
                None,
                [msg]
            ).then(
                lambda: "响应完成",
                None,
                [status_text]
            )

            submit_btn.click(
                self.respond,
                [msg, chatbot, model_dropdown, temperature_slider, max_tokens_slider, system_prompt_input],
                [chatbot],
                show_progress=True
            ).then(
                lambda: "",
                None,
                [msg]
            ).then(
                lambda: "响应完成",
                None,
                [status_text]
            )

            clear_btn.click(
                self.clear_chat,
                None,
                [chatbot, msg],
                show_progress=False
            ).then(
                lambda: "对话已清空",
                None,
                [status_text]
            )

            # 模型切换时更新状态
            model_dropdown.change(
                lambda model: f"已切换到模型: {model}",
                [model_dropdown],
                [status_text]
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
                        choices=["llama2", "mistral", "codellama"],
                        value="llama2",
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


class OllamaMultiModelDashboard:
    """Ollama 多模型控制面板"""

    def __init__(self):
        self.models_info = {}
        self.update_models()

    def update_models(self):
        """更新模型信息"""
        try:
            models = ollama.list()
            for model in models['models']:
                self.models_info[model['name']] = {
                    'size': model.get('size', 0),
                    'modified': model.get('modified_at', '')
                }
        except:
            pass

    def pull_model(self, model_name: str):
        """拉取模型"""
        try:
            response = ollama.pull(model_name, stream=True)
            progress = ""
            for line in response:
                if 'status' in line:
                    progress += f"{line['status']}\n"
                    yield progress
            yield f"✅ 模型 {model_name} 下载完成！"
        except Exception as e:
            yield f"❌ 下载失败: {str(e)}"

    def delete_model(self, model_name: str):
        """删除模型"""
        try:
            ollama.delete(model_name)
            self.update_models()
            return f"✅ 已删除模型: {model_name}"
        except Exception as e:
            return f"❌ 删除失败: {str(e)}"

    def create_dashboard(self):
        """创建控制面板"""

        with gr.Blocks(title="Ollama 模型管理面板") as demo:
            gr.Markdown("""
            # 🛠️ Ollama 模型管理面板
            管理本地的大语言模型
            """)

            with gr.Tabs():
                with gr.TabItem("📊 模型列表"):
                    gr.Markdown("### 本地已安装模型")
                    models_table = gr.Dataframe(
                        headers=["模型名称", "大小", "修改时间"],
                        value=self.get_models_table_data(),
                        interactive=False
                    )

                    refresh_btn = gr.Button("🔄 刷新列表")

                    refresh_btn.click(
                        self.update_models_and_table,
                        None,
                        [models_table]
                    )

                with gr.TabItem("⬇️ 下载模型"):
                    gr.Markdown("### 下载新模型")

                    available_models = [
                        "llama2", "llama2:13b", "llama2:70b",
                        "mistral", "codellama", "neural-chat",
                        "starling-lm", "orca-mini"
                    ]

                    with gr.Row():
                        model_to_pull = gr.Dropdown(
                            choices=available_models,
                            label="选择要下载的模型"
                        )
                        pull_btn = gr.Button("开始下载", variant="primary")

                    pull_output = gr.Textbox(
                        label="下载进度",
                        lines=10,
                        interactive=False
                    )

                    pull_btn.click(
                        self.pull_model,
                        [model_to_pull],
                        [pull_output]
                    )

                with gr.TabItem("🗑️ 删除模型"):
                    gr.Markdown("### 删除本地模型")

                    with gr.Row():
                        model_to_delete = gr.Dropdown(
                            choices=list(self.models_info.keys()),
                            label="选择要删除的模型"
                        )
                        delete_btn = gr.Button("删除模型", variant="stop")

                    delete_output = gr.Textbox(
                        label="操作结果",
                        interactive=False
                    )

                    delete_btn.click(
                        self.delete_model,
                        [model_to_delete],
                        [delete_output]
                    ).then(
                        self.update_models_and_table,
                        None,
                        [models_table]
                    )

        return demo

    def get_models_table_data(self):
        """获取模型表格数据"""
        data = []
        for name, info in self.models_info.items():
            size_mb = info.get('size', 0) / 1024 / 1024
            data.append([
                name,
                f"{size_mb:.1f} MB",
                info.get('modified', '')
            ])
        return data

    def update_models_and_table(self):
        """更新模型信息并刷新表格"""
        self.update_models()
        return self.get_models_table_data()


def main():
    """主函数"""

    # 创建应用实例
    streaming_ui = GradioStreamingUI()
    document_qa = OllamaDocumentQA()
    model_dashboard = OllamaMultiModelDashboard()

    # 创建标签页界面
    with gr.Blocks(title="Ollama AI 助手套件", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Ollama 本地大模型 AI 助手套件
        完全本地运行，无需网络，保护隐私！
        """)

        with gr.Tabs():
            with gr.TabItem("💬 智能聊天"):
                chat_interface = streaming_ui.create_interface()

            with gr.TabItem("📄 文档问答"):
                doc_qa_interface = document_qa.create_document_qa_interface()

            with gr.TabItem("🛠️ 模型管理"):
                dashboard_interface = model_dashboard.create_dashboard()

            with gr.TabItem("📖 使用说明"):
                gr.Markdown("""
                ## 使用说明

                ### 💬 智能聊天
                1. 选择模型（默认为 llama2）
                2. 调整参数（温度、生成长度等）
                3. 在输入框输入问题
                4. 点击发送或按回车

                ### 📄 文档问答
                1. 在左侧输入或上传文档
                2. 在右侧输入基于文档的问题
                3. AI 会根据文档内容回答

                ### 🛠️ 模型管理
                1. 查看已安装的模型
                2. 下载新的模型
                3. 删除不需要的模型

                ### 📊 系统要求
                - Python 3.8+
                - Ollama 已安装并运行
                - 足够的内存运行大模型

                ### 🔧 常见问题
                **Q: 如何安装 Ollama？**
                A: 访问 https://ollama.ai 下载并安装

                **Q: 如何下载更多模型？**
                A: 在模型管理页面选择并下载

                **Q: 为什么响应很慢？**
                A: 大模型需要更多计算资源，请确保有足够的 RAM

                ### 🐛 问题反馈
                如遇问题，请检查：
                1. Ollama 服务是否运行（终端运行 `ollama serve`）
                2. 模型是否已下载
                3. 内存是否充足
                """)

    # 启动界面
    demo.launch(
        share=False,  # 不创建公开链接
        debug=True
    )


if __name__ == "__main__":
    # 检查 Ollama 是否可用
    try:
        import ollama

        # 测试连接
        ollama.list()
        print("✅ Ollama 连接成功！")
        print("🚀 正在启动 Gradio 界面...")
        main()
    except Exception as e:
        print(f"❌ 无法连接到 Ollama: {e}")
        print("请确保：")
        print("1. Ollama 已安装 (https://ollama.ai)")
        print("2. 终端运行: ollama serve")
        print("3. 已下载模型: ollama pull llama2")
        input("按回车键退出...")