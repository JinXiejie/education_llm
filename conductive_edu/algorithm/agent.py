import requests
import json

from conductive_edu.config import Config
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from hf_xet import force_sigint_shutdown
import numpy as np

class Agent(object):
    def __init__(self, agent_type="user", device=None, logger=None, monitor=None):
        # 定义想要调用的函数（默认DeepSeek）
        self.model_name = Config.DEEPSEEK_R1_MODEL_LIST[1]
        self.base_url = Config.BASE_URL

    """
        API:/generate
        功能: 生成指定模型的文本补全。输入提示词后，模型根据提示生成文本结果
        请求方法: POST
        API参数:
        :param model： 必填,模型名称
        :param prompt：必填 生成文本所使用的提示词
        :param suffix: 可选 生成的补全之后附加的文本
        :param stream: 可选 是否流式传输响应，默认为true
        :param system: 可选 覆盖模型系统信息的字段，影响生成文本的风格
        :param temperature: 可选，控制文本生成的随机性 默认值为1
    """
    def generate_stream(self, question, agent_type='api/generate'):
        api_url = self.base_url + agent_type
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "prompt": question,
            "stream": True  # 开启流式输出
        }
        response = requests.post(api_url, headers=headers, json=data, stream=True)
        for line in response.iter_lines():
            if line:
                json_data = json.loads(line.decode("utf-8"))
                print(json_data.get("response", ""), end="", flush=True)

    """
        API:/chat
        功能: 模拟对话补全，支持多轮交互，适用于聊天机器人等场景
        请求方法: POST
        API参数:
        :param model： 必填,模型名称
        :param messages：必填,对话的消息列表，按顺序包含历史对话，每条消息包含role和content
        :param role: user(用户) assistant（助手） 或system（系统）
        :param content：消息内容
        :param stream: 可选,是否流式传输响应,默认true
    """
    def chat_stream(self, messages, agent_type='api/chat'):
        api_url = self.base_url + agent_type
        think_over = False
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": True  # 开启流式输出
        }
        response = requests.post(api_url, headers=headers, json=data, stream=True)
        answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_object = json.loads(line)
                    if 'message' in json_object and 'content' in json_object['message']:
                        chunk = json_object['message']['content']
                        if '</think>' in chunk:
                            think_over = True
                        if think_over == True and chunk != '</think>':
                            answer += chunk
                            print(chunk, end='', flush=True)
                        # answer += chunk
                        # print(chunk, end='', flush=True)
                except json.decoder.JSONDecodeError:
                    pass
        # return full_response
        return answer

    """
        API:/embed
        功能: 为输入的文本生成嵌入向量，常用于语义搜索或分类等任务。
        请求方法: POST
        API参数:
        :param model: 必填,生成嵌入模型名称
        :param input: 必填,文本或文本列表，用户生成嵌入
        :param truncate: 可选,是否在文本超出上下文长度时进行截断，默认true
        :param stream: 可选,是否流式传输响应，默认为true
    """
    def embed_stream(self, text, agent_type='api/embed'):
        api_url = self.base_url + agent_type

        headers = {"Content-Type": "application/json"}
        data = {"model": self.model_name, "input": text}
        response = requests.post(api_url, json=data, headers=headers)
        response
        return response.json()

    def embed_api(self, file_path, agent_type='api/embed'):
        document = PyPDFLoader(file_path).load()
        print(f'载入后的变量类型为：{type(document)},', f"该PDF一共包含{len(document)}页")
        # 知识库中单段文本长度
        chunk_size = 500
        # 知识库中相似文本重合长度
        overlap_size = 50
        # 声明一个RecursiveCharacterTextSplitter实例
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
        split_docs = text_splitter.split_documents(document)
        # 定义Embeddings
        embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

        persist_directory = "../../data_base/vector_db/chroma"
        persist_directory = "G:\PycharmProjects\education_llm\conductive_edu\data_base"
        vectordb = Chroma.from_documents(documents=split_docs[:100], embedding=embedding, persist_directory=persist_directory)
        vectordb.persist()  # 持久化向量数据库

        question = "什么是机器学习？"
        sim_docs = vectordb.similarity_search(question, k=3)
        print(f"检索到的内容数：{len(sim_docs)}")
        for i, sim_doc in enumerate(sim_docs):
            print(f"检索到第{i}个内容：\n{sim_doc.page_content[:200]}", end="\n----------\n")

        #
        # # print(split_docs)
        # print(f"分割后的块数：{len(split_docs)}")
        # print(f"分割后的字符数（可以用来大致评估token数）：{sum([len(doc.page_content) for doc in split_docs])}")
        # response = requests.post(api_url, json=data, headers=headers)



        query1 = '机器学习'
        query2 = '强化学习'
        emb1 = embedding.embed_query(query1)
        emb2 = embedding.embed_query(query2)
        print(emb1)
        print(emb2)
        # 计算两个词向量的相关性
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        print(f"{query1}和{query2}向量之间的点积为：{np.dot(emb1, emb2)}")

    # https: // blog.csdn.net / coding2008 / article / details / 152780143
    def data_processor(self, file_path, agent_type='api/embed'):
        api_url = self.base_url + agent_type
        document = PyPDFLoader(file_path).load()
        print(f'载入后的变量类型为：{type(document)},', f"该PDF一共包含{len(document)}页")
        # 知识库中单段文本长度
        chunk_size = 500
        # 知识库中相似文本重合长度
        overlap_size = 50
        # 声明一个RecursiveCharacterTextSplitter实例
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
        split_docs = text_splitter.split_documents(document)

        persist_directory = "../../data_base/vector_db/chroma"
        vectordb = Chroma.from_documents(documents=split_docs[:100], embedding=embedding, persist_directory=persist_directory)
        vectordb.persist() # 持久化向量数据库

        # print(split_docs)
        print(f"分割后的块数：{len(split_docs)}")
        print(f"分割后的字符数（可以用来大致评估token数）：{sum([len(doc.page_content) for doc in split_docs])}")
        response = requests.post(api_url, json=data, headers=headers)
        # 定义Embeddings
        embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
        query1 = '机器学习'
        query2 = '强化学习'
        emb1 = embedding.embed_query(query1)
        emb2 = embedding.embed_query(query2)
        print(emb1)
        print(emb2)
        # 计算两个词向量的相关性
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        print(f"{query1}和{query2}向量之间的点积为：{np.dot(emb1, emb2)}")






