
import ollama
import json
import time
from typing import Generator, List, Dict, Any

from conductive_edu.config import Config

class OllamaStreamingChat:
    """Ollama 流式聊天类"""

    def __init__(self):
        self.model_name = Config.LLM_MODEL_NAME
        self.system_prompt = Config.SYSTEM_PROMPT
        self.history = []

    def format_history_for_ollama(self, gradio_history: List, messages) -> List[Dict[str, str]]:
        """
        将 Gradio 的历史记录格式转换为 Ollama 格式
        """
        if gradio_history and isinstance(gradio_history, list):
            for turn in gradio_history:
                if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    user_msg = str(turn[0]).strip()
                    assistant_msg = str(turn[1]).strip()
                    # 处理用户消息
                    if user_msg:
                        messages.append({"role": "user", "content": user_msg})

                    # 处理助手消息
                    if assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg})
        return messages

    def stream_response(self, message: str, history: List, model: str = None) -> Generator[str, None, None]:
        """
        流式生成响应
        """
        if model:
            self.model = model

        # 添加系统提示
        # 转换历史记录格式
        system_messages = [{"role": "system", "content": self.system_prompt}]
        ollama_messages = self.format_history_for_ollama(history, system_messages)

        # 添加当前用户消息
        current_msg = str(message).strip()
        if not current_msg:
            yield "错误：消息为空"
            return

        ollama_messages.append({"role": "user", "content": current_msg})

        print(f"发送给 Ollama 的消息: {json.dumps(ollama_messages, indent=2, ensure_ascii=False)}")

        # 流式生成响应
        try:
            stream = ollama.chat(model=self.model, messages=ollama_messages, stream=True)
            full_response = ""
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    token = chunk['message']['content']
                    full_response += token
                    yield full_response

        except Exception as e:
            print(f"Ollama 调用错误: {e}")
            yield f"错误: {str(e)}"

    def clear_history(self):
        """清空历史记录"""
        self.history = []
