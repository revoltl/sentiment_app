import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from src.utils.font_utils import find_cjk_font, cjk_font_families
import pyLDAvis
import pyLDAvis.lda_model
import numpy as np
import math

# 设置绘图风格和中文字体
plt.rcParams['font.sans-serif'] = cjk_font_families()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 14  # 增大全局字体

def load_all_stopwords(stopwords_dir: str) -> set[str]:
    """合并读取目录下所有 .txt 文件作为停用词表"""
    stopwords = set()
    if not os.path.exists(stopwords_dir):
        raise FileNotFoundError(f"停用词目录不存在: {stopwords_dir}")
    
    txt_files = [f for f in os.listdir(stopwords_dir) if f.endswith(".txt")]
    if not txt_files:
        raise FileNotFoundError(f"停用词目录下没有 .txt 文件: {stopwords_dir}")

    for filename in txt_files:
        file_path = os.path.join(stopwords_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    return stopwords

def clean_text(text: str) -> str:
    """清理文本，去除特殊字符和多余空格"""
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
    return " ".join(text.split())

def tokenize(text: str, stopwords: set[str]) -> str:
    """对中文文本进行分词并去除停用词"""
    words = jieba.lcut(clean_text(text))
    filtered_words = [w for w in words if len(w) > 1 and w not in stopwords]
    return " ".join(filtered_words)

def save_topic_results(results_str: str, output_path: str):
    """保存主题分析结果字符串到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"topic_info": results_str}, f, ensure_ascii=False, indent=4)

def load_topic_results(input_path: str) -> str:
    """从 JSON 文件加载主题分析结果字符串"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到主题分析结果文件: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["topic_info"]

# --- 可视化函数 ---

def plot_text_length_dist(texts: list[str], output_dir: str):
    """文本长度分布图（竖向柱状，4:3 比例）"""
    lengths = [len(str(t)) for t in texts]
    # 4:3 比例 -> width=12, height=9
    plt.figure(figsize=(12, 9))
    ax = sns.histplot(lengths, bins=30, kde=True, color='skyblue')
    
    # 添加数据标签
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2., height + 0.1, f'{height:}', ha="center", fontsize=10)

    plt.title("文本长度分布", fontsize=20)
    plt.xlabel("长度", fontsize=16)
    plt.ylabel("频数", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_dist.png"), transparent=True, dpi=300)
    plt.close()

def plot_word_frequency(processed_texts: list[str], output_dir: str, top_n: int = 20):
    """词频图（横向条状，倒序，4:3 比例）"""
    all_words = " ".join(processed_texts).split()
    word_counts = pd.Series(all_words).value_counts().head(top_n)
    
    plt.figure(figsize=(12, 9))
    ax = sns.barplot(x=word_counts.values, y=word_counts.index, hue=word_counts.index, palette="viridis", legend=False)
    
    # 添加数据标签
    for i, v in enumerate(word_counts.values):
        ax.text(v + 0.1, i, f'{v:.3f}', va='center', fontsize=12)
        
    plt.title(f"Top {top_n} 高频词", fontsize=20)
    plt.xlabel("出现次数", fontsize=16)
    plt.ylabel("词汇", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "word_frequency.png"), transparent=True, dpi=300)
    plt.close()

def plot_wordcloud(processed_texts: list[str], output_dir: str):
    """词云图（4:3 比例）"""
    text = " ".join(processed_texts)
    font_path = find_cjk_font()
    if not font_path:
        print("警告：未找到中文字体，词云中的中文可能显示为方块")
        
    # 4:3 比例 -> 1200x900
    wc = WordCloud(
        font_path=font_path,
        background_color=None,
        mode="RGBA",
        width=1200,
        height=900,
        max_words=100
    ).generate(text)
    
    plt.figure(figsize=(12, 9))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wordcloud.png"), transparent=True, dpi=300)
    plt.close()

def plot_tfidf_keywords(processed_texts: list[str], output_dir: str, top_n: int = 20):
    """TF-IDF 关键词图（横向条状，倒序，4:3 比例）"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    tfidf_series = pd.Series(mean_tfidf, index=feature_names).sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 9))
    ax = sns.barplot(x=tfidf_series.values, y=tfidf_series.index, hue=tfidf_series.index, palette="magma", legend=False)
    
    # 添加数据标签
    for i, v in enumerate(tfidf_series.values):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=12)
        
    plt.title(f"Top {top_n} TF-IDF 关键词", fontsize=20)
    plt.xlabel("平均 TF-IDF 得分", fontsize=16)
    plt.ylabel("词汇", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tfidf_keywords.png"), transparent=True, dpi=300)
    plt.close()

def plot_topic_keywords(lda_model, feature_names, n_top_words, output_dir: str):
    """Topic 关键词柱状图（平方矩阵布局，动态 DPI，4:3 比例）"""
    n_topics = lda_model.n_components
    
    # 计算平方矩阵布局 (优先保证 cols >= rows 适配 4:3)
    cols = math.ceil(math.sqrt(n_topics))
    rows = math.ceil(n_topics / cols)
    
    # 动态设定 DPI 保证清晰度
    dynamic_dpi = max(300, 100 * max(cols, rows))
    
    # 整体画布维持 4:3 比例 -> 12:9
    fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 9 * rows), squeeze=False)
    
    for i in range(rows):
        for j in range(cols):
            topic_idx = i * cols + j
            ax = axes[i, j]
            if topic_idx < n_topics:
                topic = lda_model.components_[topic_idx]
                top_idx = topic.argsort()[: -n_top_words - 1 : -1]
                top_features = [feature_names[k] for k in top_idx]
                weights = topic[top_idx]
                
                sns.barplot(x=weights, y=top_features, ax=ax, hue=top_features, palette="rocket", legend=False)
                
                # 添加数据标签
                for k, v in enumerate(weights):
                    ax.text(v + 0.01, k, f'{v:.3f}', va='center', fontsize=12)
                    
                ax.set_title(f"主题 {topic_idx + 1} 核心词", fontsize=18)
                ax.tick_params(axis='both', which='major', labelsize=14)
            else:
                ax.axis('off')
                
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topic_keywords.png"), transparent=True, dpi=dynamic_dpi)
    plt.close()

def save_lda_vis(lda_model, tf_matrix, vectorizer, output_path: str):
    """保存 pyLDAvis 交互式 HTML"""
    vis_data = pyLDAvis.lda_model.prepare(lda_model, tf_matrix, vectorizer)
    pyLDAvis.save_html(vis_data, output_path)

# --- 主逻辑 ---

def run_lda_analysis(texts: list[str], lda_config: dict, visualize: bool = False, output_dir: str = None) -> str:
    """
    执行 LDA 主题分析并返回格式化的字符串。
    如果 visualize=True，则生成各种可视化图表到 output_dir。
    """
    if not texts:
        return "数据为空，无法进行主题分析。"

    print(f"LDA 正在处理 {len(texts)} 条文本数据...")

    # 1. 加载停用词
    stopwords_dir = lda_config["stopwords_dir"]
    stopwords = load_all_stopwords(stopwords_dir)

    # 2. 分词与预处理
    processed_texts = [tokenize(t, stopwords) for t in texts]
    processed_texts = [t for t in processed_texts if t.strip()]
    
    if not processed_texts:
        return "数据预处理后为空（可能全是停用词或特殊字符），无法进行主题分析。"

    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_text_length_dist(texts, output_dir)
        plot_word_frequency(processed_texts, output_dir)
        plot_wordcloud(processed_texts, output_dir)
        plot_tfidf_keywords(processed_texts, output_dir)

    # 3. 向量化
    model_params = lda_config["lda_model"]
    vectorizer = CountVectorizer(
        max_df=model_params["max_df"], 
        min_df=model_params["min_df"]
    ) 
    tf = vectorizer.fit_transform(processed_texts)
    
    if tf.shape[1] == 0:
        return "词袋模型为空，无法提取主题。"

    # 4. LDA 模型训练
    lda = LatentDirichletAllocation(
        n_components=model_params["n_topics"],
        max_iter=model_params["max_iter"],
        learning_method=model_params["learning_method"],
        random_state=model_params["random_state"]
    )
    lda.fit(tf)

    # 5. 可视化 LDA
    if visualize and output_dir:
        feature_names = vectorizer.get_feature_names_out()
        plot_topic_keywords(lda, feature_names, model_params["top_n_words"], output_dir)
        save_lda_vis(lda, tf, vectorizer, os.path.join(output_dir, "lda_vis.html"))

    # 6. 提取结果并格式化
    feature_names = vectorizer.get_feature_names_out()
    top_n_words = model_params["top_n_words"]
    results = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[: -top_n_words - 1 : -1]
        top_words = [feature_names[i] for i in top_words_idx]
        results.append(f"主题 {topic_idx + 1}: {', '.join(top_words)}")

    return "\n".join(results)
