
from conductive_edu.backend_serve.ollama_streaming import OllamaStreamingChat
import gradio as gr
import ollama
from typing import Generator, List, Dict, Any



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

    def respond(self, message: str, history: List, model: str, temperature: float, max_tokens: int,
                system_prompt: str) -> Generator[List, None, None]:
        """
        处理用户消息 - 返回 Gradio Chatbot 期望的格式

        Gradio Chatbot 期望: [[user_msg1, assistant_msg1], [user_msg2, assistant_msg2], ...]
        """
        print(f"respond: 收到消息: {message}")
        print(f"respond: 当前历史: {history}")
        if not history:
            history.append({"role": "assistant", "content": ""})

        if not message or not str(message).strip():
            # 返回当前历史（不添加新消息）
            yield history
            return

        # 创建一个新的历史记录副本
        new_history = history.copy() if history else []

        # 添加用户消息（助手消息暂时为空）
        new_history.append({"role": "user", "content": message})

        # 生成响应
        ollama_stream_chat = OllamaStreamingChat()
        try:
            # 注意：传递给 stream_response 的是之前的历史（不包含当前消息）
            previous_history = history if history else []
            new_history.append({"role": "assistant", "content": ""})
            for response_chunk in ollama_stream_chat.stream_response(message, previous_history, model):
                full_response = response_chunk
                # 更新最后一条消息的助手回复
                new_history[-1] = {"role": "assistant", "content": full_response}
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
        theme = gr.themes.Soft(primary_hue="purple",secondary_hue="pink",)

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
