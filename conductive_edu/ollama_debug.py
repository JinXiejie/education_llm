import gradio as gr
import ollama
import json
from typing import List, Dict, Any


class OllamaChat:
    def __init__(self):
        self.model = "deepseek-r1:8b"

    def get_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            result = ollama.list()
            if 'models' in result:
                return [model['model'] for model in result['models']]
            return ["llama2"]
        except Exception as e:
            print(f"获取模型列表失败: {e}")
            return ["llama2"]

    def create_message(self, role: str, content: Any) -> Dict[str, str]:
        """创建标准的消息字典"""
        # 确保 content 是字符串
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)

        # 清理内容
        content = content.strip()

        return {
            "role": role.strip().lower(),
            "content": content
        }

    def format_messages_for_ollama(self, history: List, current_message: str) -> List[Dict[str, str]]:
        """格式化消息供 Ollama 使用"""
        messages = []

        print(f"[DEBUG] 历史记录类型: {type(history)}")
        print(f"[DEBUG] 历史记录内容: {history}")

        # 处理历史记录
        if history:
            for turn in history:
                if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    user_msg = turn[0]
                    assistant_msg = turn[1]

                    # 添加用户消息
                    if user_msg is not None and str(user_msg).strip():
                        messages.append(self.create_message("user", user_msg))

                    # 添加助手消息
                    if assistant_msg is not None and str(assistant_msg).strip():
                        messages.append(self.create_message("assistant", assistant_msg))

        # 添加当前消息
        if current_message and str(current_message).strip():
            messages.append(self.create_message("user", current_message))

        print(f"[DEBUG] 格式化后的消息数量: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"[DEBUG] 消息 {i}: role={msg['role']}, content_preview={msg['content'][:50]}...")

        return messages

    def test_ollama_connection(self):
        """测试 Ollama 连接"""
        try:
            test_message = self.create_message("user", "Hello, please respond with 'OK'")
            response = ollama.chat(
                model=self.model,
                messages=[test_message]
            )
            return True, response['message']['content']
        except Exception as e:
            return False, str(e)

    def stream_chat(self, message: str, history: List, model: str = None) -> str:
        """流式聊天"""
        if model:
            self.model = model

        # 格式化消息
        messages = self.format_messages_for_ollama(history, message)

        # 验证消息格式
        if not messages:
            yield "错误：消息为空"
            return

        # 检查每条消息
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                yield f"错误：消息 {i} 不是字典，而是 {type(msg)}"
                return

            if 'role' not in msg or 'content' not in msg:
                yield f"错误：消息 {i} 缺少必要的键"
                return

            if not isinstance(msg['role'], str) or not isinstance(msg['content'], str):
                yield f"错误：消息 {i} 的 role 或 content 不是字符串"
                return

        # 尝试调用 Ollama
        try:
            print(f"[DEBUG] 调用 Ollama，模型: {self.model}")
            print(f"[DEBUG] 消息: {json.dumps(messages, indent=2, ensure_ascii=False)}")

            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True
            )

            response = ""
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    token = chunk['message']['content']
                    response += token
                    yield response

        except Exception as e:
            print(f"[ERROR] Ollama 调用失败: {e}")
            yield f"调用失败: {str(e)}"


def main():
    # 创建聊天实例
    chat = OllamaChat()

    # 测试连接
    print("测试 Ollama 连接...")
    success, result = chat.test_ollama_connection()
    if success:
        print(f"✅ 连接成功: {result}")
    else:
        print(f"❌ 连接失败: {result}")
        return

    # 获取可用模型
    available_models = chat.get_models()
    print(f"可用模型: {available_models}")

    # 创建 Gradio 界面
    with gr.Blocks(title="Ollama Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🦙 Ollama 本地大模型聊天")

        # 模型选择
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[3] if available_models else "llama2",
                label="选择模型"
            )

        # 聊天界面
        chatbot = gr.Chatbot(
            label="对话",
            height=400,
            # type="messages"
        )

        # 输入区域
        msg = gr.Textbox(
            label="输入消息",
            placeholder="请输入您的问题...",
            lines=2
        )

        submit_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空", variant="secondary")

        # 状态显示
        status = gr.Textbox(label="状态", value="就绪", interactive=False)

        def process_message(message: str, history: List, model: str):
            """处理消息"""
            if not message.strip():
                yield history, "请输入有效消息"
                return

            # 添加用户消息（先不添加助手回复）
            history.append([message, None])

            # 生成响应
            try:
                full_response = ""
                for response_chunk in chat.stream_chat(message, history[:-1], model):
                    full_response = response_chunk
                    # 更新助手回复
                    history[-1][1] = full_response
                    yield history, "正在生成..."

                yield history, "完成"

            except Exception as e:
                print(f"处理消息错误: {e}")
                history[-1][1] = f"错误: {str(e)}"
                yield history, "出错"

        def clear_chat():
            return [], [], "已清空"

        # 绑定事件
        def handle_submit(message, history, model):
            for new_history, status_text in process_message(message, history, model):
                yield new_history, status_text, ""

        msg.submit(
            handle_submit,
            [msg, chatbot, model_dropdown],
            [chatbot, status, msg]
        )

        submit_btn.click(
            handle_submit,
            [msg, chatbot, model_dropdown],
            [chatbot, status, msg]
        )

        clear_btn.click(
            clear_chat,
            None,
            [chatbot, msg, status]
        )

        # 模型切换
        model_dropdown.change(
            lambda m: f"已切换到模型: {m}",
            [model_dropdown],
            [status]
        )

    return demo


if __name__ == "__main__":
    # 首先进行完整的诊断
    print("=" * 60)
    print("Ollama 连接诊断")
    print("=" * 60)

    # 方法1：直接测试 ollama 包
    print("\n1. 直接测试 ollama 包...")
    try:
        # 最简单的测试
        response = ollama.generate(model='deepseek-r1:8b', prompt='hello')
        print(f"✅ generate() 测试成功: {response['response'][:50]}...")
    except Exception as e:
        print(f"❌ generate() 测试失败: {e}")

    print("\n2. 测试 chat() 方法...")
    try:
        # 测试标准格式
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        response = ollama.chat(model='deepseek-r1:8b', messages=messages)
        print(f"✅ chat() 测试成功: {response['message']['content'][:50]}...")
    except Exception as e:
        print(f"❌ chat() 测试失败: {e}")
        print(f"错误详情: {type(e).__name__}: {e}")

    print("\n3. 测试模型列表...")
    try:
        models = ollama.list()
        print(f"✅ 模型列表: {models}")
    except Exception as e:
        print(f"❌ 获取模型列表失败: {e}")

    # 启动界面
    print("\n" + "=" * 60)
    print("启动 Gradio 界面...")
    print("=" * 60)

    demo = main()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True
    )