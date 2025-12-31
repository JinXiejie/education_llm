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
    # model_name = "'BAAI/bge-large-zh-v1.5'"
    model_kwargs = {'device': 'cpu', 'trust_remote_code': True}  # 使用cpu
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
def create_data_base(persist_directory, file_path, embeddings, re_create):
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
        split_docs = load_knowledge_file(file_path, file_type='pdf')
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



def init(file_path, persist_directory):
    embed_model_name = Config.EMBEDDING_MODEL_LIST[0]
    # file_path = "G:\PycharmProjects\education_llm\conductive_edu\data\knowledge.pdf"
    # persist_directory = "G:\PycharmProjects\education_llm\conductive_edu\data_base\\"
    url = Config.BASE_URL
    model_name = Config.DEEPSEEK_R1_MODEL_LIST[1]

    embeddings = init_embed_model(embed_model_name)

    vectordb = create_data_base(persist_directory, file_path, embeddings, re_create=False)
    custom_prompt_template = create_prompt()
    custom_qa = VectorDBQA.from_chain_type(
        # llm = Ollama(model="qwen2-7b:latest",temperature=0.3),
        llm=Ollama(base_url=url, model=model_name, temperature=0.3),  # 模型选择
        chain_type="stuff",
        vectorstore=vectordb,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt_template},
    )
    return custom_qa
custom_qa = init("", messages, file_path, persist_directory)

def rag_app(question, custom_qa, messages):
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
            except json.decoder.JSONDecodeError:
                pass
    return answer


system_prompt = Config.SYSTEM_PROMPT
messages = [{
    "role": "system",
    "content": system_prompt
}]
file_path = "G:\PycharmProjects\education_llm\conductive_edu\data\knowledge.pdf"
persist_directory = "G:\PycharmProjects\education_llm\conductive_edu\data_base\\"
custom_qa = init("", messages, file_path, persist_directory)
while True:
    question = input("Question: ")
    if question.lower() in ["exit", "quit"]:  #### 输入“exit”或“quit”可以退出对话框！
        print("Ending conversation.")
        break

    # 将用户问题字典对象添加到messages列表中
    messages.append({"role": "user", "content": question})
    # print(messages[-1])
    # 调用API并获取响应
    response = rag_app(question, custom_qa, messages)
    # 将大模型的回复信息添加到messages列表中
    messages.append({"role": "assistant", "content": response})
    print("\n")
