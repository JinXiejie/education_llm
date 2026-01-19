import gradio as gr
import ollama
from typing import Generator

import requests
import json
from typing import Generator, List, Tuple, Dict, Any

from conductive_edu.config import Config


def stream_ollama_response(message: str, history: list[Tuple[str, str]], model: str = "deepseek-r1:8b") -> Generator[str, None, None]:
    """流式响应函数"""
    # 构建消息列表
    messages = []
    print(f"发送的消息格式: {json.dumps(messages, indent=2, ensure_ascii=False)}")

    # 添加历史对话
    try:
        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})

        # 添加当前消息
        messages.append({"role": "user", "content": message})
        print(f"发送的消息格式: {json.dumps(messages, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"添加历史对话 时出错: {e}")
        yield f"抱歉，出错了: {str(e)}"

    # 流式调用 Ollama
    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )

        # 流式生成响应
        response = ""
        for chunk in stream:
            content = chunk['message']['content']
            response += content
            yield response

    except Exception as e:
        print(f"调用 Ollama 时出错: {e}")
        yield f"抱歉，出错了: {str(e)}"


# 获取可用模型
def get_available_models():
    """获取本地可用的 Ollama 模型"""
    try:
        model_name_list = []
        models = ollama.list()
        # for model in models['models']:
        #     model_name_list = model['name']
        # temp = [model['name'] for model in models['models']]
        return [model['model'] for model in models['models']]
    except:
        return ["llama2"]  # 默认模型


def format_chat_history(history: List[List[str]]) -> List[Tuple[str, str]]:
    """将 Gradio 聊天历史格式转换为我们的格式"""
    formatted_history = []
    for turn in history:
        if len(turn) >= 2:
            formatted_history.append((turn[0], turn[1]))
    return formatted_history

# 创建界面
with gr.Blocks(title="Ollama 聊天助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Ollama 本地大模型聊天
    完全本地运行，保护隐私！
    """)

    # 模型选择
    available_models = get_available_models()
    model_selector = gr.Dropdown(
        choices=available_models,
        value=available_models[3],
        label="选择模型"
    )

    # 聊天界面
    chatbot = gr.Chatbot(
        height=500,
        label="对话记录"
    )

    # 输入区域
    with gr.Row():
        msg = gr.Textbox(
            scale=4,
            placeholder="输入您的问题...",
            show_label=False,
            container=False
        )
        submit_btn = gr.Button("发送", scale=1, variant="primary")

    # 控制按钮
    with gr.Row():
        clear_btn = gr.Button("清空对话", variant="secondary")
        retry_btn = gr.Button("重新生成", variant="secondary")


    # 事件处理
    # def respond(message, history, model):
    #     """响应函数"""
    #     for response in stream_ollama_response(message, history, model):
    #         yield response

    # 主要响应函数
    def respond(message: str, history: List[List[str]], model: str):
        """响应函数 - 修复版本"""
        # 将 Gradio 格式的历史转换为我们的格式
        formatted_history = format_chat_history(history)

        # 更新状态
        yield "", "正在思考..."  # 先返回空的聊天和状态更新

        # 流式生成响应
        full_response = ""
        for response_chunk in stream_ollama_response(message, formatted_history, model):
            full_response = response_chunk
            # 创建新的历史记录（包括当前响应）
            new_history = history + [[message, full_response]]
            yield new_history, "正在生成..."

        yield history + [[message, full_response]], "完成"


    print('绑定事件')
    # 绑定事件
    msg.submit(
        respond,
        [msg, chatbot, model_selector],
        [chatbot]
    ).then(
        lambda: "",
        None,
        [msg]
    )

    print('执行按钮')
    submit_btn.click(
        respond,
        [msg, chatbot, model_selector],
        [chatbot]
    ).then(
        lambda: "",
        None,
        [msg]
    )

    clear_btn.click(
        lambda: [],
        None,
        [chatbot]
    )

    retry_btn.click(
        lambda history: respond(history[-1][0], history[:-1], model_selector.value) if history else "",
        [chatbot],
        [chatbot]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(share=False)
