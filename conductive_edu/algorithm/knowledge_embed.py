import json
import os

from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import VectorDBQA
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import PromptTemplate
# from langchain.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from conductive_edu.config import Config
from langchain_community.llms import Ollama

# 实例化embedding模型
def init_embed_model(model_name):
      # 这里需要按照自己选择的模型，补充huggingface上下载的模型包的名称
    # model_name = "moka-ai/m3e-base"
    model_kwargs = {'device': 'cpu', 'trust_remote_code': True} # 使用cpu
    encode_kwargs = {'normalize_embeddings': False}
    # 实例化
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        # cache_dir=cache_dir,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

# 对文档加载+分割
def load_knowledge_file(file_path, file_type):
    # file_path = "G:\PycharmProjects\education_llm\conductive_edu\data\knowledge.pdf"
    if file_type == 'pdf':
        document = PyPDFLoader(file_path).load()
        print(f'载入后的变量类型为：{type(document)},', f"该PDF一共包含{len(document)}页")
        # 知识库中单段文本长度
        chunk_size = Config.CHUNK_SIZE
        # 知识库中相似文本重合长度
        overlap_size = Config.OVERLAP_SIZE
        # 声明一个RecursiveCharacterTextSplitter实例
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
        split_docs = text_splitter.split_documents(document)
        return split_docs

# 使用LLM对文档编码
def create_data_base(persist_directory, split_docs, embeddings, re_create):
    # persist_directory请填写你希望chroma向量知识库存储的位置
    # persist_directory = "G:\PycharmProjects\education_llm\conductive_edu\data_base\"
    if re_create:
        if os.path.exists(persist_directory):
            os.mkdir(persist_directory)
        else:
            # 检查是否已存在数据库，可能会出现维度不一致报错
            for f in os.listdir(persist_directory):
                file_path = os.path.join(persist_directory, f)
                os.remove(file_path)
        vectordb = Chroma.from_documents(split_docs[:100], embeddings, persist_directory=persist_directory)
    else:
        # 直接加载已经构建好的向量数据库
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb

def create_prompt():
    # 加入提示词
    custom_prompt = """
    Context: {context}
    Question: {question}
    根据知识库回答问题，什么是redis
    """

    # 定义查询方法
    custom_prompt_template = PromptTemplate(
        template=custom_prompt, input_variables=["context", "question"]
    )
    return custom_prompt_template


def rag_app(question, context, file_path, persist_directory):
    embed_model_name = Config.EMBEDDING_MODEL_LIST[0]
    # file_path = "G:\PycharmProjects\education_llm\conductive_edu\data\knowledge.pdf"
    # persist_directory = "G:\PycharmProjects\education_llm\conductive_edu\data_base\\"
    url = Config.BASE_URL
    model_name = Config.DEEPSEEK_R1_MODEL_LIST[1]

    embeddings = init_embed_model(embed_model_name)
    split_docs = load_knowledge_file(file_path, file_type='pdf')
    vectordb = create_data_base(persist_directory, split_docs, embeddings, re_create=False)
    custom_prompt_template = create_prompt()
    custom_qa = VectorDBQA.from_chain_type(
        # llm = Ollama(model="qwen2-7b:latest",temperature=0.3),
        llm=Ollama(base_url=url, model=model_name, temperature=0.3),  # 模型选择
        chain_type="stuff",
        vectorstore=vectordb,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt_template},
    )

    response = custom_qa.run(question)
    answer = ""
    think_over = False
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





