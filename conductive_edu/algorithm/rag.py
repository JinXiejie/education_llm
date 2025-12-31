import os
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import VectorDBQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from conductive_edu.config import Config
from langchain_community.llms import Ollama



class Rag(object):
    def __init__(self, knowledge_path="user", device=None, logger=None, monitor=None):
        # 定义想要调用的函数（默认DeepSeek）
        self.llm_model_name = Config.DEEPSEEK_R1_MODEL_LIST[1]
        self.base_url = Config.BASE_URL

        self.embed_model_name = Config.EMBEDDING_MODEL_LIST[0]
        # 使用cpu
        self.model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
        self.encode_kwargs = {'normalize_embeddings': False}

        # 实例化embedding模型
    def get_embeddings(self):
        # 这里需要按照自己选择的模型，补充huggingface上下载的模型包的名称
        # 实例化
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            # cache_dir=cache_dir,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )
        return embeddings

    # 对文档加载+分割
    def load_knowledge(self, knowledge_file, file_type):
        # knowledge_file = "G:\PycharmProjects\education_llm\conductive_edu\data\knowledge.pdf"
        if file_type == 'pdf':
            document = PyPDFLoader(knowledge_file).load()
            print(f'载入后的变量类型为：{type(document)},', f"该PDF一共包含{len(document)}页")
            # 知识库中单段文本长度
            chunk_size = Config.CHUNK_SIZE
            # 知识库中相似文本重合长度
            overlap_size = Config.OVERLAP_SIZE
            # 声明一个RecursiveCharacterTextSplitter实例
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
            split_docs = text_splitter.split_documents(document)
            return split_docs

    def create_prompt(self):
        # 加入提示词
        custom_prompt = """
        Context: {context}
        Question: {question}
        Helpful Answer:
        严格规则
                做一个平易近人且充满活力的老师，通过引导用户学习来帮助他们掌握知识。
                1、了解用户。如果您不知道他们的目标或年纪水平，在深入之前请询问用户，保持简介。如果他们不回答，目标是给出能让一名高中生理解的解释
                2、建立在现有知识的基础上。将新想法与用户已知的内容联系起来。
                3、引导用户，而不仅仅是给出答案。使用提问、提示和小步骤，让用户自己发现答案。
                4、检查并巩固。在难点之后，确认用户能够复述或运用该概念。提供快速总结、助记符或小复习来帮助巩固这些概念。
                5、节奏多样化。将解释、问题和活动（如角色扮演、联系环节或要求用户教你）混合起来，使其感觉像是一场对话，而不是一场演讲。
                最重要的是：不要替用户完成他们的工作。不要回答作业问题——通过与用户协作，从他们已知的知识出发，帮助他们找到答案。
                语气和方法
                保持热情、耐心，并用平实的语言表达；不要使用过多的感叹号或表情符号。保持会话流畅：始终清楚下一步该做什么，一旦活动达到目的就切换或结束。并且要简介——永远不要发送长篇大论的回复。追求良好的互动交流
        """
        # 定义查询方法
        prompt_template = PromptTemplate(
            template=custom_prompt, input_variables=["context", "question"],
        )
        return prompt_template

    # 使用LLM对文档编码
    def get_vector(self, persist_dir, knowledge_file, embeddings, db_create):
        # persist_dir = "G:\PycharmProjects\education_llm\conductive_edu\data_base\\"
        if db_create:
            if os.path.exists(persist_dir):
                os.mkdir(persist_dir)
            else:
                # 检查是否已存在数据库，可能会出现维度不一致报错
                for f in os.listdir(persist_dir):
                    file_path = os.path.join(persist_dir, f)
                    os.remove(file_path)
            split_docs = self.load_knowledge(knowledge_file, file_type='pdf')
            vectordb = Chroma.from_documents(split_docs[:100], embeddings, persist_directory=persist_dir)
        else:
            # 直接加载已经构建好的向量数据库
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        return vectordb

    def create_client(self, persist_dir, knowledge_file, is_create_db = False):
        vectordb = self.get_vector(persist_dir, knowledge_file, self.get_embeddings(), is_create_db)
        custom_qa = VectorDBQA.from_chain_type(
            # llm = Ollama(model="qwen2-7b:latest",temperature=0.3),
            llm=Ollama(base_url=self.base_url, model=self.llm_model_name, temperature=0.3),  # 模型选择
            chain_type="stuff",
            vectorstore=vectordb,
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.create_prompt()}
        )
        return custom_qa

    def run(self, is_create_db):
        # print("生成文本嵌入", embedding)
        persist_dir = Config.PERSIST_DIR
        knowledge_path = Config.KNOWLEDGE_PATH
        rag_client = self.create_client(persist_dir, knowledge_path, is_create_db)
        while True:
            question = input("Question: ")
            # 输入“exit”或“quit”可以退出对话框！
            if question.lower() in ["exit", "quit"]:
                print("Ending conversation.")
                break
            # 调用API并获取响应
            rag_client.invoke(question, callbacks=[StreamingStdOutCallbackHandler()])
            print("\n")



