
from conductive_edu import algorithm
from conductive_edu.algorithm.agent import Agent
from conductive_edu.config import Config


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
        if question.lower() in ["exit", "quit"]:   #### 输入“exit”或“quit”可以退出对话框！
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

def embed_completion():
    agent = Agent()
    embedding = agent.embed_stream("什么是证券营销")
    print("生成文本嵌入", embedding)


def generate(question):
    response = agent_executor.invoke({"question": question})
    return response.get("output")

import gradio as gr
with gr.Blocks() as demo:
    gr.Markdown("# Gradio Demo UI 🖍️")
    input_text = gr.Text(label="Your Input")
    btn = gr.Button("Submit")
    result = gr.Textbox(label="Generated Result")

    btn.click(fn=generate, inputs=[input_text], outputs=[result])

gr.close_all()
demo.launch()

if __name__ == "__main__":
    agent = Agent()
    file_path = "G:\PycharmProjects\education_llm\conductive_edu\data\knowledge.pdf"
    agent.embed_api(file_path)
    # chat_completion()
    # embed_completion()


