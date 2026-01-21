import gradio as gr
import ollama
from typing import Generator, List, Dict, Optional, Any
import os, requests, hashlib, json
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from conductive_edu.config import Config


class DBProcessor:
    def __init__(self):
        self.persist_directory = Config.PERSIST_DIR
        self.system_prompt = Config.SYSTEM_PROMPT
        self.chunk_size = 1000
        self.chunk_overlap = 200

        # 初始化向量数据库
        self.vectorstore = None
        self.collection_name = "rag_documents"

        # 初始化嵌入模型
        self.embedding_model = Config.EMBEDDING_MODEL_NAME
        print(f"加载嵌入模型: {self.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    def load_vectorstore(self):
        """加载向量数据库"""
        try:
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                print("加载现有的向量数据库...")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                print(f"向量数据库加载成功，包含 {self.vectorstore._collection.count()} 个文档")
            else:
                print("创建新的向量数据库...")
                self.vectorstore = Chroma(
                    # documents=
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
        except Exception as e:
            print(f"加载向量数据库失败: {e}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )

    def clear_database(self) -> bool:
        """清空向量数据库"""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                self.vectorstore = None

            # 重新创建
            self.load_vectorstore()
            print("向量数据库已清空")
            return True

        except Exception as e:
            print(f"清空数据库失败: {e}")
            return False