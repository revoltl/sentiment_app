import os
import sys

# 把工程根目录加入模块搜索路径，保证从任意目录运行都能导入 src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from tqdm import tqdm
import json

from src.utils.rag_utils import OllamaEmbedder, VectorStore

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_rag_index(csv_path: str, rag_config_path: str):
    # 1. 加载全局配置
    config = load_json(rag_config_path)
    embed_conf = config["embedding"]
    vs_conf = config["vector_store"]

    # 2. 加载 sidecar JSON 配置
    json_path = csv_path.rsplit('.', 1)[0] + '.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到 sidecar JSON 配置文件: {json_path}")
    
    sidecar_conf = load_json(json_path)
    fmt = sidecar_conf["data_format"]
    
    # 自动生成索引名称（基于文件名）
    index_name = os.path.basename(csv_path).rsplit('.', 1)[0]

    # 3. 读取数据
    print(f"正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 4. 初始化嵌入模型和向量库
    embedder = OllamaEmbedder(embed_conf['model_name'], embed_conf['base_url'])
    vector_store = VectorStore(vs_conf['base_dir'], index_name)

    # 5. 执行入库
    print(f"任务类型: {sidecar_conf['mission']}, 正在构建索引: {index_name}")
    
    # 准备元数据
    metadata = df.to_dict('records')
    texts = df[fmt['text_col']].astype(str).tolist()

    # 批量处理以防内存溢出或请求过快
    # 5090 性能极强，我们可以适当增加并发批次
    batch_size = 50
    for i in tqdm(range(0, len(texts), batch_size), desc="构建向量库 (并发 Embedding)"):
        batch_texts = texts[i:i+batch_size]
        batch_meta = metadata[i:i+batch_size]
        
        # 获取向量
        embeddings = embedder.embed_documents(batch_texts)
        
        # 存入向量库
        vector_store.add_documents(embeddings, batch_meta)

    # 6. 保存
    vector_store.save()
    print(f"索引已保存至: {os.path.join(vs_conf['base_dir'], index_name)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 向量库构建工具 (Sidecar JSON 版)")
    parser.add_argument("--csv_path", type=str, required=True, help="RAG 数据源 CSV 路径 (需存在同名 .json)")
    parser.add_argument("--rag_config", type=str, required=True, help="全局 RAG 配置文件路径")
    
    args = parser.parse_args()
    
    build_rag_index(args.csv_path, args.rag_config)
