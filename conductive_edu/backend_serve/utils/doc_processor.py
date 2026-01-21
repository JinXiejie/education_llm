
from typing import Generator, List, Dict, Optional, Any
import os, requests, hashlib
from datetime import datetime
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from conductive_edu.backend_serve.utils.db_processor import DBProcessor
from conductive_edu.config import Config


class DocumentProcessor:

    def __init__(self,
                 # embedding_model: str = "all-MiniLM-L6-v2",
                 embedding_model: str = Config.EMBEDDING_MODEL_NAME,
                 # persist_directory: str = "./chroma_db",
                 persist_directory: str = Config.PERSIST_DIR,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        初始化 RAG 系统

        Args:
            embedding_model: 嵌入模型名称
            persist_directory: 向量数据库存储目录
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
        """
        self.persist_directory = persist_directory
        self.system_prompt = Config.SYSTEM_PROMPT
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化嵌入模型
        print(f"加载嵌入模型: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""]
        )

        # 初始化向量数据库
        # self.vectorstore = None
        self.collection_name = "rag_documents"

        # 初始化 Ollama 模型
        # self.llm_model = "deepseek-r1:8b"
        self.llm_model = "deepseek-r1:1.5b"

        # 创建存储目录
        os.makedirs(persist_directory, exist_ok=True)
        # os.makedirs("./uploaded_docs", exist_ok=True)

        # 加载现有的向量数据库
        db_processor = DBProcessor()
        db_processor.load_vectorstore()
        self.vectorstore = db_processor.vectorstore

    def _get_file_loader(self, file_path: str):
        """根据文件类型获取相应的加载器"""
        ext = os.path.splitext(file_path)[1].lower()

        loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader,
            '.csv': CSVLoader
        }

        if ext in loaders:
            return loaders[ext](file_path)
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

    def process_document(self, file_path: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        处理单个文档

        Args:
            file_path: 文档路径
            metadata: 文档元数据

        Returns:
            处理后的文档块列表
        """
        print(f"处理文档: {file_path}")

        try:
            # 加载文档
            loader = self._get_file_loader(file_path)
            documents = loader.load()

            # 添加元数据
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)

            # 文本分割
            texts = self.text_splitter.split_documents(documents)

            print(f"文档分割为 {len(texts)} 个块")
            return texts

        except Exception as e:
            print(f"处理文档失败: {e}")
            raise

    def add_local_documents(self, file_paths: List[str], metadata: Optional[Dict] = None) -> bool:
        """
        添加文档到向量数据库

        Args:
            file_paths: 文档路径列表
            metadata: 文档元数据

        Returns:
            是否成功
        """
        try:
            all_texts = []

            for file_path in file_paths:
                if not os.path.exists(file_path):
                    print(f"文件不存在: {file_path}")
                    continue

                # 生成文档ID
                file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]

                doc_metadata = {
                    "source": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_type": os.path.splitext(file_path)[1],
                    "file_size": os.path.getsize(file_path),
                    "hash": file_hash,
                    "upload_time": datetime.now().isoformat()
                }

                if metadata:
                    doc_metadata.update(metadata)

                # 处理文档
                texts = self.process_document(file_path, doc_metadata)
                all_texts.extend(texts)

            if all_texts:
                # 添加到向量数据库
                self.vectorstore.add_documents(all_texts)
                # self.vectorstore.persist()

                print(f"成功添加 {len(all_texts)} 个文档块到向量数据库")
                return True
            else:
                print("没有有效的文档可以添加")
                return False

        except Exception as e:
            print(f"添加文档失败: {e}")
            return False

    def search_documents(self, query: str, k: int = 4) -> List[Dict]:
        """
        搜索相关文档

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            相关文档列表
        """
        if not self.vectorstore:
            print("向量数据库未初始化")
            return []

        try:
            # 相似度搜索
            docs = self.vectorstore.similarity_search_with_score(query, k=k)

            results = []
            for doc, score in docs:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "source": doc.metadata.get("source", "unknown")
                }
                results.append(result)

            print(f"搜索到 {len(results)} 个相关文档")
            return results

        except Exception as e:
            print(f"搜索文档失败: {e}")
            return []


    def get_document_stats(self) -> Dict:
        """获取文档统计信息"""
        if not self.vectorstore:
            return {"total_documents": 0}

        try:
            count = self.vectorstore._collection.count()
            return {"total_documents": count}
        except:
            return {"total_documents": 0}