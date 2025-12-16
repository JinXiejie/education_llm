from llama_index.core import download_loader, GPTVectorStoreIndex, SimpleDirectoryReader,VectorStoreIndex,Settings
from llama_index.llms.ollama import Ollama
import ollama
import json
from conductive_edu.config import Config


class Rag(object):
    def __init__(self, agent_type="user", device=None, logger=None, monitor=None):
        # 定义想要调用的函数（默认DeepSeek）
        self.model = Config.DEEPSEEK_R1_MODEL
        # self.model = Config.DEEPSEEK_R1_1B
        self.api_url = Config.BASE_URL
    def embed_llama_index(self, question, agent_type='generate'):
        llm = Ollama(model=self.model, base_url=self.api_url, request_timeout=60)

        response = llm.complete(question)
        for line in response.iter_lines():
            if line:
                json_data = json.loads(line.decode("utf-8"))
                print(json_data.get("response", ""), end="", flush=True)


if __name__ == "__main__":
    rag = Rag()
    question = '请用Java帮我实现二分查找算法'
    rag.embed_llama_index(question)
    # print(completion)

#
# client = ollama.Client()
# llm_predictor = LLMPredictor(llm=LLMPredictor.LLMNames.OLLAMA)
# model = client.list_models()[0]
# print(client.list_models())
#
# Settings.llm = Ollama(model="deepseek-r1:1.5b", base_url="http://localhost:11434", temperature=0.1)
#
# # 读取本地文档
# # data_dir = '/Users/xiaofan/Downloads/附件1：武汉学院线上线下混合式教学认定审批表.docx'
# # documents = SimpleDirectoryReader(data_dir).load_data()
# # /Users/xiaofan/PycharmProjects/education_llm/conductive_edu/data/实习手册_参考.pdf
# data_dir = '/Users/xiaofan/PycharmProjects/education_llm/conductive_edu/data'
# # documents = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".pdf"]).load_data()
# print("\n步骤1: 加载文档...")
# try:
#     documents = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".pdf"]).load_data()
#     # 或者更精确地指定文件：
#     # documents = SimpleDirectoryReader(input_files=["./data/my_knowledge.txt"]).load_data()
#     if not documents:
#         print("错误：未能加载任何文档。请检查 './data/my_knowledge.txt' 是否存在且包含内容。")
#         # return
#     print(f"成功加载 {len(documents)} 个文档片段（初始加载时可能是一个大文档）。")
# except Exception as e:
#     print(f"加载文档时出错: {e}")
#     # return
#
# print("\n步骤2: 创建索引 (这可能需要一点时间，具体取决于文档大小和网络)...")
# index = 0
# try:
#     index = VectorStoreIndex.from_documents(documents, show_progress=True)
#     print("索引创建成功！")
# except Exception as e:
#     print(f"创建索引时出错: {e}")
#     # 常见错误可能是OpenAI API Key未设置或无效，或者网络问题
#     print("请确保OPENAI_API_KEY环境变量已正确设置，并且网络连接正常。")
#     # return
#
# print("\n步骤3: 创建查询引擎...")
# query_engine = index.as_query_engine(
#     similarity_top_k=3, # 指定检索最相关的3个文本块
#     # streaming=True # 如果LLM支持流式输出，可以开启
# )
# print("查询引擎创建成功！")
#
# # 通过GPTVectorStoreIndex构建向量索引，这一步能让模型快速检索相关信息
# index = GPTVectorStoreIndex.from_documents(documents)
#
# # 构建完成的索引可保存至本地，方便后续使用
# index_dir = '/Users/xiaofan/Downloads/'
# index.VectorStoreIndex(index_dir)
#
# print("\n步骤4: 执行查询...")
# user_query = "RAG技术有什么优势？它能解决什么问题？"
# print(f"用户问题: {user_query}")
#
# try:
#     response = query_engine.query(user_query)
#     print("\n查询完成！")
#
#     # 5. 打印结果
#     print("\n步骤5: LLM生成的回答:")
#     print("========================")
#     print(response)  # response对象包含答案文本和源节点等信息
#     print("========================")
#
#     # (可选) 查看检索到的源文本块
#     if response.source_nodes:
#         print("\n检索到的相关源文本块:")
#         for i, source_node in enumerate(response.source_nodes):
#             print(f"--- 源文本块{i + 1} (相似度: {source_node.score:.4f}) ---")
#             print(source_node.get_content()[:200] + "...")  # 打印部分内容
#             print(f"来源文件: {source_node.metadata.get('file_name', 'N/A')}")
#             print("--------------------------------------")
#
# except Exception as e:
#     print(f"执行查询时出错: {e}")
#
#     print("\nRAG应用执行完毕。")
#
# if __name__ == "__main__":
#     main()
#
# class LocalAgent:
#     def __init__(self, index_path):
#         self.index = GPTVectorStoreIndex.load_from_disk(index_path, service_context=service_context)
#
#         def query(self, question):
#             response = self.index.query(question)
#             return response.response
#
#
# if __name__ == "__main__":
#     agent = LocalAgent('your_index_directory')
#     while True:
#         user_input = input("请输入问题（输入'quit'退出）：")
#         if user_input.lower() == 'quit':
#             break
# answer = agent.query(user_input)
# print("智能体回答：", answer)
