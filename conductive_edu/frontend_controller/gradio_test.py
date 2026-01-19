import gradio as gr
import ollama
import json


# 最简单的测试版本
def chat(message, history):
    """
    最简单的聊天函数
    history 是 Gradio 格式: [[user_msg, assistant_msg], ...]
    """
    print(f"收到的 message: {message}")
    print(f"收到的 history 类型: {type(history)}")
    print(f"收到的 history 内容: {json.dumps(history, indent=2, ensure_ascii=False)}")

    # 构建 Ollama 消息
    messages = []

    # 转换历史记录
    if history:
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                user_msg = turn[0] if turn[0] is not None else ""
                assistant_msg = turn[1] if turn[1] is not None else ""

                messages.append({"role": "user", "content": str(user_msg)})
                messages.append({"role": "assistant", "content": str(assistant_msg)})
    print(f"收到的 messages 类型: {type(messages)}")
    # 添加当前消息
    messages.append({"role": "user", "content": str(message)})

    print(f"发送给 Ollama 的消息: {json.dumps(messages, indent=2, ensure_ascii=False)}")

    # 调用 Ollama
    try:
        stream = ollama.chat(
            model='deepseek-r1:8b',
            messages=messages,
            stream=True
        )
        print(f"收到的 stream 类型: {type(stream)}")
        print(f"收到的 messages 类型: {type(messages)}")

        response = ""
        for chunk in stream:
            print(f"stream_response 收到的 chunk 类型: {type(chunk)}")
            # print(f"format_history_for_ollama Gradio 历史记录内容: {json.dumps(chunk, indent=2, ensure_ascii=False)}")
            if 'message' in chunk and 'content' in chunk['message']:
                token = chunk['message']['content']
                # print("token:" + str(token))
                response += token
                # print("response:" + str(response))
                # print(f"收到的 response 类型: {type(response)}")
                yield response

    except Exception as e:
        print(f"错误: {e}")
        yield f"调用 Ollama 失败: {str(e)}"


# 创建最简单的界面
demo = gr.ChatInterface(
    fn=chat,
    title="Ollama 测试",
    description="最简单的测试界面"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)


# G:\anaconda3\python.exe G:\PycharmProjects\education_llm\conductive_edu\gradio_test.py
# * Running on local URL:  http://127.0.0.1:7860
# * To create a public link, set `share=True` in `launch()`.
# 收到的 message: 你好
# 收到的 history 类型: <class 'list'>
# 收到的 history 内容: []
# 收到的 messages 类型: <class 'list'>
# 发送给 Ollama 的消息: [
#   {
#     "role": "user",
#     "content": "你好"
#   }
# ]
# 收到的 stream 类型: <class 'generator'>
# 收到的 messages 类型: <class 'list'>
# 收到的 response 类型: <class 'str'>


# 问题根源：错误是因为消息格式不正确。Gradio 的 Chatbot 组件默认的 history 格式是 List[List[str]]（列表的列表），而 Ollama 期望的是 List[Dict[str, str]]（字典列表）。上面的修复代码已经正确处理了这个转换。