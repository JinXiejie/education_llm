# import requests
# import json
#
# # 定义函数
# def func(question, model, api_url):
#     headers = {
#     "Content-Type": "application/json"
#     }
#     data = {
#     "model": model,
#     "prompt": question,
#     "stream": True  # 开启流式输出
#     }
#     response = requests.post(api_url, headers=headers, json=data, stream=True)
#     for line in response.iter_lines():
#         if line:
#             json_data = json.loads(line.decode("utf-8"))
#             print(json_data.get("response", ""), end="", flush=True)
#
# # 调用函数
# model='deepseek-r1:1.5b'
# api_url = "http://127.0.0.1:11434/api/generate"
# func('你好', model=model, api_url=api_url)



# 导入所需包
import requests
import json

# 定义函数
def func(messages, model, api_url):
    headers = {
    "Content-Type": "application/json"
    }
    data = {
    "model": model,
    "messages":messages,
    "stream": True  # 开启流式输出
    }
    response = requests.post(API_URL, headers=headers, json=data, stream=True)
    answer = ""
    for line in response.iter_lines():
        if line:
            try:
                json_object = json.loads(line)
                if 'message' in json_object and 'content' in json_object['message']:
                    chunk = json_object['message']['content']
                    answer += chunk
                    print(chunk, end='', flush=True)
            except json.decoder.JSONDecodeError:
                pass
    # return full_response
    return answer

# 初始化一个messages列表
messages = [{
        "role": "system",
        "content": "大语言模型。"
    }]
# 定义想要调用的函数（默认DeepSeek）
model='deepseek-r1:1.5b'
api_url = "http://127.0.0.1:11434/api/generate"

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
    response = func(messages=messages, model = model, api_url = api_url)
    # 将大模型的回复信息添加到messages列表中
    messages.append({"role": "assistant", "content": response})