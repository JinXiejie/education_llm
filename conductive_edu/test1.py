import gradio as gr
import ollama
import json
from typing import List, Tuple, Dict, Any, Generator


class OllamaChat:
    """Ollama 聊天处理器"""

    def __init__(self):
        self.model = "llama2"

    def get_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            result = ollama.list()
            models = result.get('models', [])
            return [model['model'] for model in models]
        except Exception as e:
            print(f"获取模型列表失败: {e}")
            return ["llama2"]

    def format_history_for_ollama(self, gradio_history: List) -> List[Dict[str, str]]:
        """
        将 Gradio 的历史记录格式转换为 Ollama 格式

        Gradio 格式: [[user_message, assistant_message], ...]
        Ollama 格式: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]
        """
        messages = []

        # 调试：打印输入格式
        print(f"Gradio 历史记录类型: {type(gradio_history)}")
        print(f"Gradio 历史记录内容: {json.dumps(gradio_history, indent=2, ensure_ascii=False)}")

        if gradio_history:
            for turn in gradio_history:
                if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    user_msg = turn[0]
                    assistant_msg = turn[1]

                    # 添加用户消息
                    if user_msg and isinstance(user_msg, str):
                        messages.append({"role": "user", "content": user_msg})

                    # 添加助手消息
                    if assistant_msg and isinstance(assistant_msg, str):
                        messages.append({"role": "assistant", "content": assistant_msg})

        return messages

    def stream_response(self, message: str, history: List, model: str = None) -> Generator[str | Any, Any, None]:
        """
        流式生成响应

        参数:
            message: 当前用户消息
            history: Gradio 格式的历史记录
            model: 模型名称

        返回:
            完整的响应文本
        """
        if model:
            self.model = model

        # 转换历史记录格式
        ollama_messages = self.format_history_for_ollama(history)

        # 添加当前用户消息
        ollama_messages.append({"role": "user", "content": message})

        # 调试：打印发送给 Ollama 的消息
        print("=" * 50)
        print("发送给 Ollama 的消息:")
        for i, msg in enumerate(ollama_messages):
            print(f"[{i}] role: {msg['role']}")
            print(f"    content: {msg['content'][:50]}..." if len(
                msg['content']) > 50 else f"    content: {msg['content']}")
        print("=" * 50)

        # 流式生成响应
        full_response = ""
        try:
            stream = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True
            )

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

    # 创建 Gradio 界面
    with gr.Blocks(title="Ollama 聊天助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🦙 Ollama 本地大模型聊天
        完全本地运行，保护隐私！
        """)

        # 状态显示
        with gr.Row():
            status_display = gr.Textbox(
                label="状态",
                value="就绪",
                interactive=False
            )

        # 模型选择
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[0] if available_models else "llama2",
                label="选择模型"
            )

            test_btn = gr.Button("测试连接", variant="secondary")

        # 聊天界面
        chatbot = gr.Chatbot(
            label="对话记录",
            height=400,
            type="messages",  # 使用 messages 类型
            avatar_images=(
                "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",  # 用户
                "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"  # 助手
            )
        )

        # 输入区域
        with gr.Row():
            message_input = gr.Textbox(
                placeholder="请输入您的问题...",
                show_label=False,
                scale=4,
                lines=2
            )
            submit_btn = gr.Button("发送", variant="primary", scale=1)

        # 控制按钮
        with gr.Row():
            clear_btn = gr.Button("清空对话", variant="secondary")
            retry_btn = gr.Button("重新生成", variant="secondary")

        # 示例问题
        gr.Examples(
            examples=[
                ["你好，请介绍一下你自己"],
                ["用 Python 写一个 Hello World 程序"],
                ["解释一下什么是机器学习"],
                ["讲一个有趣的笑话"]
            ],
            inputs=message_input,
            label="示例问题"
        )

        # ========== 事件处理函数 ==========

        def test_connection(model_name: str) -> str:
            """测试 Ollama 连接"""
            try:
                test_messages = [{"role": "user", "content": "请回复'测试成功'"}]
                response = ollama.chat(model=model_name, messages=test_messages)
                return f"✅ 连接成功！模型: {model_name}"
            except Exception as e:
                return f"❌ 连接失败: {str(e)}"

        def process_message(
                message: str,
                history: List,
                model: str
        ) -> Tuple[List, str]:
            """
            处理用户消息

            返回:
                (更新后的历史记录, 状态信息)
            """
            if not message.strip():
                return history, "请输入有效消息"

            # 添加用户消息到历史记录（先添加一个空的助手回复）
            history.append([message, None])

            # 生成响应
            response_text = ""
            for response_chunk in chat_processor.stream_response(
                    message,
                    history[:-1],  # 排除当前正在处理的消息
                    model
            ):
                response_text = response_chunk
                # 更新最后一条消息的助手回复
                history[-1][1] = response_text
                yield history, "正在生成..."

            yield history, "完成"

        def clear_chat() -> Tuple[List, str]:
            """清空聊天"""
            return [], "对话已清空"

        def retry_last_message(history: List, model: str) -> Tuple[List, str]:
            """重新生成最后一条消息的回复"""
            if not history:
                return history, "没有消息可以重新生成"

            # 获取最后一条用户消息
            last_message = history[-1][0]

            # 移除最后一条消息
            history = history[:-1]

            # 重新处理
            for updated_history, status in process_message(last_message, history, model):
                yield updated_history, status

        # ========== 绑定事件 ==========

        # 测试连接按钮
        test_btn.click(
            test_connection,
            [model_dropdown],
            [status_display]
        )

        # 清空按钮
        clear_btn.click(
            clear_chat,
            None,
            [chatbot, status_display]
        )

        # 重新生成按钮
        retry_btn.click(
            retry_last_message,
            [chatbot, model_dropdown],
            [chatbot, status_display]
        )

        # 提交消息（按回车或点击发送）
        def handle_submit(message, history, model):
            """处理消息提交"""
            for updated_history, status in process_message(message, history, model):
                yield updated_history, status, ""

        message_input.submit(
            handle_submit,
            [message_input, chatbot, model_dropdown],
            [chatbot, status_display, message_input]
        )

        submit_btn.click(
            handle_submit,
            [message_input, chatbot, model_dropdown],
            [chatbot, status_display, message_input]
        )

        # 模型切换
        model_dropdown.change(
            lambda m: f"已切换到模型: {m}",
            [model_dropdown],
            [status_display]
        )

    return demo


if __name__ == "__main__":
    # 先验证 Ollama 是否正常工作
    print("正在检查 Ollama 连接...")
    try:
        # 测试最简单的调用
        response = ollama.chat(
            model='llama2',
            messages=[{'role': 'user', 'content': 'ping'}]
        )
        print(f"✅ Ollama 连接成功！")
        print(f"测试响应: {response['message']['content'][:100]}...")
    except Exception as e:
        print(f"❌ Ollama 连接失败: {e}")
        print("\n请确保：")
        print("1. Ollama 已安装 (https://ollama.ai)")
        print("2. 在终端运行: ollama serve")
        print("3. 已下载模型: ollama pull llama2")
        input("\n按回车键退出...")
        exit(1)

    # 创建并启动界面
    demo = main()
    demo.launch(
        share=False,
        show_error=True
    )