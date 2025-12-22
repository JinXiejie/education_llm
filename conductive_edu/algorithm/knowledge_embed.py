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
model_name = "BAAI/bge-large-zh-v1.5"  # 这里需要按照自己选择的模型，补充huggingface上下载的模型包的名称
# model_name = "moka-ai/m3e-base"
cache_dir = "G:\PycharmProjects\education_llm\conductive_edu\\algorithm\embed_model\\"  # 这里需要按照自己选择的模型，补充huggingface上下载的模型包的名称
model_kwargs = {'device': 'cpu', 'trust_remote_code': True} # 使用cpu
encode_kwargs = {'normalize_embeddings': False}
# 实例化
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    # cache_dir=cache_dir,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# # 对文档加载+分割
# file_path = "G:\PycharmProjects\education_llm\conductive_edu\data\\"
# loader = DirectoryLoader(path=file_path,glob='*.pdf') # 加载文件夹中的所有xlsx类型的文件（可以按需修改）
# docs = loader.load() # 将数据转成 document 对象，每个文件会作为一个 document
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0) # 调节分块最大长度和相邻块之间的重叠token数
# splits = text_splitter.split_documents(docs)


file_path = "G:\PycharmProjects\education_llm\conductive_edu\data\knowledge.pdf"
document = PyPDFLoader(file_path).load()
print(f'载入后的变量类型为：{type(document)},', f"该PDF一共包含{len(document)}页")
# 知识库中单段文本长度
chunk_size = 500
# 知识库中相似文本重合长度
overlap_size = 50
# 声明一个RecursiveCharacterTextSplitter实例
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
split_docs = text_splitter.split_documents(document)

# 使用LLM对文档编码
# persist_directory请填写你希望chroma向量知识库存储的位置
persist_directory = "G:\PycharmProjects\education_llm\conductive_edu\data_base"
# 检查是否已存在数据库，可能会出现维度不一致报错
vector_store = Chroma.from_documents(split_docs[:100], embeddings, persist_directory=persist_directory)

# 直接加载已经构建好的向量数据库
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# 加入提示词
custom_prompt = """
Context: {context}
Question: {question}
根据知识库回答问题，什么是redis
"""

url = Config.BASE_URL
model_name = Config.DEEPSEEK_R1_MODEL_LIST[1]

# 定义查询方法
custom_prompt_template = PromptTemplate(
    template=custom_prompt, input_variables=["context", "question"]
)

custom_qa = VectorDBQA.from_chain_type(
    # llm = Ollama(model="qwen2-7b:latest",temperature=0.3),
    llm = Ollama(base_url= url, model=model_name,temperature=0.3),   # 模型选择
    chain_type="stuff",
    vectorstore = vector_store,
    return_source_documents=False,
    chain_type_kwargs={"prompt": custom_prompt_template},
)
# 提问
Question = "什么是redis"
print("答案:"+custom_qa.run(Question), end="\n")


