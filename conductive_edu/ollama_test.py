# test_ollama.py
import ollama
import json

# 测试 1: 检查模型
print("1. 检查可用模型...")
try:
    models = ollama.list()
    print(f"可用模型: {models}")
except Exception as e:
    print(f"错误: {e}")

# 测试 2: 简单对话
print("\n2. 测试简单对话...")
try:
    response = ollama.chat(
        model='deepseek-r1:8b',
        messages=[
            {"role": "user", "content": "你好"}
        ]
    )
    print(f"响应: {response['message']['content']}")
except Exception as e:
    print(f"错误: {e}")

# 测试 3: 流式对话
print("\n3. 测试流式对话...")
try:
    stream = ollama.chat(
        model='deepseek-r1:8b',
        messages=[
            {"role": "user", "content": "写一首短诗"}
        ],
        stream=True
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print()
except Exception as e:
    print(f"错误: {e}")