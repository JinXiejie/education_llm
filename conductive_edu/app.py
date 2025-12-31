from langchain_core.callbacks import StreamingStdOutCallbackHandler
from openai import responses

from conductive_edu.algorithm.agent import Agent
from conductive_edu.algorithm.rag import Rag
from conductive_edu.config import Config
import gradio as gr
import random
import time


# 调用函数
def chat_completion():
    # 创建agent实例
    agent = Agent()
    # 初始化一个messages列表
    system_prompt = Config.SYSTEM_PROMPT
    messages = [{
        "role": "system",
        "content": system_prompt
    }]
    # 调用函数
    while True:
        question = input("Question: ")
        if question.lower() in ["exit", "quit"]:  #### 输入“exit”或“quit”可以退出对话框！
            print("Ending conversation.")
            break

        # 将用户问题字典对象添加到messages列表中
        messages.append({"role": "user", "content": question})
        # print(messages[-1])
        # 调用API并获取响应
        response = agent.chat_stream(messages=messages)
        # 将大模型的回复信息添加到messages列表中
        messages.append({"role": "assistant", "content": response})
        print("\n")


def rag_completion():
    # print("生成文本嵌入", embedding)
    persist_dir = Config.PERSIST_DIR
    knowledge_path = Config.KNOWLEDGE_PATH
    db_create = False
    rag_client = Rag().create_client(persist_dir, knowledge_path, db_create)
    while True:
        question = input("Question: ")
        # 输入“exit”或“quit”可以退出对话框！
        if question.lower() in ["exit", "quit"]:
            print("Ending conversation.")
            break
        # 调用API并获取响应
        rag_client.run(question, callbacks=[StreamingStdOutCallbackHandler()])
        print("\n")


chat_history = []


def predict(question, chat_history):
    # print("生成文本嵌入", embedding)
    persist_dir = Config.PERSIST_DIR
    knowledge_path = Config.KNOWLEDGE_PATH
    is_create_db = Config.IS_CREATE_DB
    rag_client = Rag().create_client(persist_dir, knowledge_path, is_create_db)
    new_user_input = {"role": "user", "content": question}
    chat_history.append(new_user_input)  # 添加用户输入到历史记录中
    # chat_input = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
    # 调用API并获取响应
    # response = rag_client.invoke(question, callbacks=[StreamingStdOutCallbackHandler()])
    response = rag_client.invoke(question)['result']
    result_text = ""
    for i in range(0, len(response), 20):
        result_text += response[i:i + 20]
        gr_result = {"role": "assistant", "content": result_text}
        # yield result_text
        yield gr_result, gr_result
        time.sleep(1)  # 每秒输出5个字
    new_ai_response = {"role": "assistant", "content": result_text}
    chat_history.append(new_ai_response)  # 添加AI的回复到历史记录中
    yield chat_history, chat_history

    # 使用chat_history来维护对话状态
    #     # new_user_input = {"role": "user", "content": input_text}
    #     # chat_history.append(new_user_input)  # 添加用户输入到历史记录中
    #     # chat_input = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
    #     # inputs = tokenizer(chat_input, return_tensors="pt", padding=True, truncation=True)
    #     # outputs = model.generate(**inputs, max_new_tokens=50)
    #     # response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("###")[1]  # 假设使用特定的分隔符来分隔回复和历史记录
    #     # new_ai_response = {"role": "assistant", "content": response}
    #     # chat_history.append(new_ai_response)  # 添加AI的回复到历史记录中
    #     # return chat_history, chat_history  # 返回更新后的聊天历史记录以供显示和进一步使用


css = """
    .outer {
    width: 100%;
    height: 900px;
    display: inline-block;
    overflow-y: scroll;
    }
    .title {
    height: 100px;
    width: 100%;
    }"""
with gr.Blocks(css=css) as demo:
    gr.Markdown("# AI大学: Multi-Agent 教学智能体")
    image_path = "G:\PycharmProjects\education_llm\image\img.png"
    with gr.Row():
        with gr.Column(scale=3):
            gr.Image(image_path, height=300, width=1200, label="AI Education")
    chatbot = gr.Chatbot(height=600)  # 使用Chatbot组件来显示对话历史和输入输出框更自然地展示对话内容
    with gr.Row():
        textbox = gr.Textbox(show_label=False, placeholder="输入消息...")
        submit_btn = gr.Button(value="发送")
    # submit_btn.click(predict, inputs=[textbox, chatbot], outputs=[textbox, chatbot])
    textbox.submit(predict, [textbox, chatbot], [textbox, chatbot], queue=False).then(
        lambda: None, None, textbox, queue=False
    )  # 使用then来清理文本框以便于连续输入，但不立即清除，以便于查看之前的对话内容。
    submit_btn.click()


if __name__ == "__main__":
    demo.launch(share=False)
#     rag_completion()
# agent = Agent()
# file_path = "G:\PycharmProjects\education_llm\conductive_edu\data\knowledge.pdf"
# chat_completion()
