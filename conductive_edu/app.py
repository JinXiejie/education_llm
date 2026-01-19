from langchain_core.callbacks import StreamingStdOutCallbackHandler
from openai import responses

from conductive_edu.algorithm.agent import Agent
from conductive_edu.algorithm.rag import Rag
from conductive_edu.backend_serve.document_QA import OllamaDocumentQA, RAGGradioInterface, LocalRAGSystem
from conductive_edu.config import Config
import gradio as gr
import random
import time
import ollama

from conductive_edu.frontend_controller.gradio_streaming_ui import GradioStreamingUI
from conductive_edu.frontend_controller.multi_model_dashboard import OllamaMultiModelDashboard


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


# css = """
#     .outer {
#     width: 100%;
#     height: 900px;
#     display: inline-block;
#     overflow-y: scroll;
#     }
#     .title {
#     height: 100px;
#     width: 100%;
#     }"""
# with gr.Blocks(css=css) as demo:
#     gr.Markdown("# AI大学: Multi-Agent 教学智能体")
#     image_path = "G:\PycharmProjects\education_llm\image\img.png"
#     with gr.Row():
#         with gr.Column(scale=3):
#             gr.Image(image_path, height=300, width=1200, label="AI Education")
#     chatbot = gr.Chatbot(height=600)  # 使用Chatbot组件来显示对话历史和输入输出框更自然地展示对话内容
#     with gr.Row():
#         textbox = gr.Textbox(show_label=False, placeholder="输入消息...")
#         submit_btn = gr.Button(value="发送")
#     # submit_btn.click(predict, inputs=[textbox, chatbot], outputs=[textbox, chatbot])
#     textbox.submit(predict, [textbox, chatbot], [textbox, chatbot], queue=False).then(
#         lambda: None, None, textbox, queue=False
#     )  # 使用then来清理文本框以便于连续输入，但不立即清除，以便于查看之前的对话内容。
#     submit_btn.click()


def main():
    """主函数"""

    # 检查 Ollama 是否可用
    try:
        import ollama

        # 测试连接
        ollama.list()
        print("✅ Ollama 连接成功！")
        print("🚀 正在启动 Gradio 界面...")
        # 创建应用实例
        streaming_ui = GradioStreamingUI()
        # document_qa = OllamaDocumentQA()
        document_qa = RAGGradioInterface()
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
                    # doc_qa_interface = document_qa.create_document_qa_interface()
                    doc_qa_interface = document_qa.create_interface()

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
            share=True,  # 不创建公开链接
            debug=True
        )
    except Exception as e:
        print(f"❌ 无法连接到 Ollama: {e}")
        print("请确保：")
        print("1. Ollama 已安装 (https://ollama.ai)")
        print("2. 终端运行: ollama serve")
        print("3. 已下载模型: ollama pull llama2")
        input("按回车键退出...")



if __name__ == "__main__":
    main()
    # demo.launch(share=True)
#     rag_completion()
# agent = Agent()
# file_path = "G:\PycharmProjects\education_llm\conductive_edu\data\knowledge.pdf"
# chat_completion()
