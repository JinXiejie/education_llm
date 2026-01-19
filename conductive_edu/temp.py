import gradio as gr
import ollama
import json
import requests
from typing import List, Tuple, Dict, Any, Generator


class OllamaChat:
    """Ollama 聊天处理器"""

    def __init__(self):
        self.model = "deepseek-r1:8b"

    def get_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            # 使用 HTTP API 获取模型
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                model_names = [model.get('model') for model in models if model.get('model')]
                return model_names
        except Exception as e:
            print(f"HTTP API 获取模型失败: {e}")

        # 备选方案
        return ["deepseek-r1:8b", "deepseek-r1:1.5b", "qwen3:0.6b", "nomic-embed-text:latest"]

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


def main():
    """创建 Gradio 界面"""
    # 创建聊天处理器
    chat_processor = OllamaChat()

    # 获取可用模型
    available_models = chat_processor.get_models()
    print(f"可用模型: {available_models}")

    if not available_models:
        available_models = ["deepseek-r1:8b", "deepseek-r1:1.5b", "qwen3:0.6b", "nomic-embed-text:latest"]

    # 选择默认模型
    default_model = available_models[0]
    for model in available_models:
        if "deepseek-r1:8b" in model:
            default_model = model
            break

    # 创建 Gradio 界面
    with gr.Blocks(title="Ollama 聊天助手") as demo:
        gr.Markdown("""
        # 🦙 Ollama 本地大模型聊天
        完全本地运行，保护隐私！
        """)

        # 状态显示
        status_display = gr.Textbox(
            label="状态",
            value="就绪",
            interactive=False
        )

        # 模型选择
        model_dropdown = gr.Dropdown(
            choices=available_models,
            value=default_model,
            label="选择模型"
        )

        # 聊天界面 - 使用正确的格式
        chatbot = gr.Chatbot(
            label="对话记录",
            height=400,
            # 注意：Chatbot 期望的格式是 [[user_msg, assistant_msg], ...]
        )

        # 输入区域
        msg = gr.Textbox(
            label="输入消息",
            placeholder="请输入您的问题...",
            lines=2
        )

        submit_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空对话", variant="secondary")

        # ========== 事件处理函数 ==========

        def test_connection(model_name: str) -> str:
            """测试 Ollama 连接"""
            try:
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model_name,
                        "messages": {"role": "user", "content": "hello"},
                        "stream": False
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    return f"✅ 连接成功！模型: {model_name}"
                else:
                    return f"❌ 连接失败: HTTP {response.status_code}"

            except Exception as e:
                return f"❌ 连接失败: {str(e)}"

        def respond(message: str, history: List, model: str) -> Generator[List, None, None]:
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
                for response_chunk in chat_processor.stream_response(message, previous_history, model):
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

        def clear_chat():
            """清空聊天"""
            return [], "对话已清空"

        # ========== 绑定事件 ==========

        # 测试连接按钮
        test_btn = gr.Button("测试连接", variant="secondary")
        test_btn.click(
            test_connection,
            [model_dropdown],
            [status_display]
        )

        # 清空按钮
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, status_display]
        )

        # 提交消息
        def handle_submit(message, history, model):
            """处理消息提交"""
            # 调用 respond 函数
            for updated_history in respond(message, history, model):
                # 返回更新后的历史记录和清空输入框
                yield updated_history, ""

        # 绑定提交事件
        msg.submit(
            handle_submit,
            inputs=[msg, chatbot, model_dropdown],
            outputs=[chatbot, msg]
        )

        submit_btn.click(
            handle_submit,
            inputs=[msg, chatbot, model_dropdown],
            outputs=[chatbot, msg]
        )

        # 模型切换
        model_dropdown.change(
            lambda m: f"已切换到模型: {m}",
            inputs=[model_dropdown],
            outputs=[status_display]
        )

        # 示例问题
        gr.Examples(
            examples=[
                ["你好"],
                ["用 Python 写一个 Hello World 程序"],
                ["你是谁？"],
                ["讲一个笑话"]
            ],
            inputs=msg,
            label="示例问题"
        )

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("Ollama 聊天界面")
    print("=" * 60)

    # 检查 Ollama 服务
    try:
        response = requests.get("http://localhost:11434/", timeout=5)
        print(f"✅ Ollama 服务正在运行")
    except Exception as e:
        print(f"⚠️  无法连接到 Ollama: {e}")
        print("请确保已运行: ollama serve")

    # 创建并启动界面
    demo = main()

    # 启动界面 - 使用正确的参数
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )