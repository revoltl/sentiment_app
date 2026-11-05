import os
import pickle
import numpy as np
import faiss
import asyncio
import ollama
from typing import List, Dict, Any

class OllamaEmbedder:
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)
        self.async_client = ollama.AsyncClient(host=base_url)

    def embed_text(self, text: str) -> np.ndarray:
        """获取单个文本的向量"""
        response = self.client.embeddings(model=self.model_name, prompt=text)
        return np.array(response['embedding'], dtype=np.float32)

    async def _async_embed_text(self, text: str) -> np.ndarray:
        """异步获取单个文本的向量"""
        response = await self.async_client.embeddings(model=self.model_name, prompt=text)
        return np.array(response['embedding'], dtype=np.float32)

    async def embed_documents_async(self, texts: List[str]) -> np.ndarray:
        """并发异步获取多个文本的向量 (利用 RTX 5090 的高性能)"""
        tasks = [self._async_embed_text(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        return np.array(embeddings, dtype=np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """获取多个文本的向量 (保持同步接口，内部可调用异步)"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # 如果已经在运行的循环中，使用 thread 或其他方式，
            # 但在构建脚本中通常不是这种情况
            return asyncio.run_coroutine_threadsafe(self.embed_documents_async(texts), loop).result()
        else:
            return loop.run_until_complete(self.embed_documents_async(texts))

class VectorStore:
    def __init__(self, index_dir: str, index_name: str):
        self.index_dir = index_dir
        self.index_name = index_name
        self.index_path = os.path.join(index_dir, f"{index_name}.index")
        self.metadata_path = os.path.join(index_dir, f"{index_name}.pkl")
        
        self.index = None
        self.metadata = []

    def create_index(self, dimension: int):
        """创建新的 FAISS 索引 (使用内积 IndexFlatIP 以支持余弦相似度)"""
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add_documents(self, embeddings: np.ndarray, docs_metadata: List[Dict[str, Any]]):
        """添加文档向量及元数据 (执行 L2 归一化)"""
        if self.index is None:
            self.create_index(embeddings.shape[1])
        
        # L2 归一化，配合 IndexFlatIP 即可计算余弦相似度
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(docs_metadata)

    def save(self):
        """保存索引和元数据"""
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self):
        """加载索引和元数据"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """执行向量搜索"""
        if self.index is None:
            self.load()
        
        if self.index is None:
            return []

        # FAISS 要求输入是 2D array
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        # 查询向量也需要归一化
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.metadata):
                res = self.metadata[idx].copy()
                # 在归一化后的 IP 索引中，distance 即为余弦相似度得分 (score)
                res['score'] = float(distances[0][i])
                results.append(res)
        
        return results

import json

def format_few_shot_examples(retrieved_docs: List[Dict[str, Any]], rag_sidecar: Dict[str, Any]) -> str:
    """根据任务类型将检索到的文档格式化为 Few-shot 示例字符串"""
    mission = rag_sidecar["mission"]
    fmt = rag_sidecar["data_format"]
    text_col = fmt["text_col"]
    label_col = fmt["label_col"]
    
    examples = []
    for doc in retrieved_docs:
        content = str(doc[text_col])
        
        if mission == "class":
            res_val = json.dumps({"label": doc[label_col]}, ensure_ascii=False)
        elif mission == "topic":
            topic_col = fmt["topic_col"]
            res_val = json.dumps({"topic": doc[topic_col], "label": doc[label_col]}, ensure_ascii=False)
        elif mission == "aspect":
            aspect_cols = fmt["aspect_cols"]
            results = []
            for col in aspect_cols:
                val = doc.get(col)
                # 假设 -1 表示未提及，过滤掉以减小 prompt 长度
                if val is not None and str(val) != "-1":
                    results.append({"aspect": col, "label": val})
            res_val = json.dumps({"results": results}, ensure_ascii=False)
        else:
            res_val = str(doc[label_col])

        # 格式化得分，保留 3 位小数
        score_val = round(doc.get('score', 0.0), 3)
        example = f"相似度得分: {score_val}\n评论: {content}\n结果: {res_val}"
        examples.append(example)
    
    return "\n\n".join(examples)
