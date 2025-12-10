import requests
import json

from conductive_edu.config import Config


class Agent(object):
    def __init__(self, agent_type="user", device=None, logger=None, monitor=None):
        # 定义想要调用的函数（默认DeepSeek）
        self.model = Config.DEEPSEEK_R1_MODEL
        # self.model = Config.DEEPSEEK_R1_1B
        self.api_url = Config.BASE_URL

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
    def generate_stream(self, question, agent_type='generate'):
        model = self.model
        api_url = self.api_url + agent_type
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
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
    def chat_stream(self, messages, agent_type='chat'):
        model = self.model
        api_url = self.api_url + agent_type
        think_over = False
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
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
    def embed_stream(self, text, agent_type='embed'):
        model = self.model
        api_url = self.api_url + agent_type
        headers = {"Content-Type": "application/json"}
        data = {"model": model, "input": text}
        response = requests.post(api_url, json=data, headers=headers)
        response
        return response.json()



