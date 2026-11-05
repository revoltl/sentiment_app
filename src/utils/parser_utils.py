import json
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from src.utils.font_utils import find_cjk_font, cjk_font_families
import jieba

# 设置绘图风格
plt.rcParams['font.sans-serif'] = cjk_font_families()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 12

def extract_json(text):
    """从文本中提取 JSON 内容，支持 markdown 代码块"""
    if not text or not isinstance(text, str):
        return None
    
    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass
    
    # 2. 尝试从 ```json ... ``` 中提取
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
            
    # 3. 尝试搜索第一个 { 和最后一个 } 之间的内容
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
            
    return None

def align_labels(parsed_data, mission, label_list, aspect_list=None, topic_list=None):
    """检查提取结果与标签列表的符合性"""
    if parsed_data is None:
        return None
        
    valid_labels = set(label_list)
    
    if mission == "class":
        label = str(parsed_data.get("label", ""))
        return label if label in valid_labels else ""
        
    elif mission == "aspect":
        results = parsed_data.get("results", [])
        if not isinstance(results, list):
            return []
        
        valid_aspects = set(aspect_list) if aspect_list else None
        aligned = []
        for item in results:
            asp = item.get("aspect", "")
            lbl = str(item.get("label", ""))
            if valid_aspects and asp not in valid_aspects:
                continue
            if lbl in valid_labels:
                aligned.append({"aspect": asp, "label": lbl})
        return aligned
        
    elif mission == "topic":
        topic = parsed_data.get("topic", "")
        label = str(parsed_data.get("label", ""))
        valid_topics = set(topic_list) if topic_list else None
        
        if valid_topics and topic not in valid_topics:
            return None
        return {"topic": topic, "label": label} if label in valid_labels else None
        
    return None

def plot_pie_and_bar(data, title, output_path):
    """通用分类分布图（饼图 + 柱状图，比例 4:3）"""
    counts = data.value_counts()
    if counts.empty: return
    
    # 设定 4:3 比例的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 饼图
    counts.plot.pie(autopct='%1.1f%%', ax=ax1, colors=sns.color_palette("pastel"))
    ax1.set_title(f"{title} 占比")
    ax1.set_ylabel("")
    
    # 柱状图
    sns.barplot(x=counts.index, y=counts.values, ax=ax2, palette="viridis", hue=counts.index, legend=False)
    for i, v in enumerate(counts.values):
        ax2.text(i, v + 0.1, f'{v:.2f}', ha='center')
    ax2.set_title(f"{title} 数量")
    ax2.set_xlabel("类别")
    ax2.set_ylabel("频数")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_heatmap(df, row_col, col_col, title, output_path):
    """情感交叉热力图 (4:3 比例)"""
    ct = pd.crosstab(df[row_col], df[col_col])
    if ct.empty: return
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(ct, annot=True, fmt='d', cmap="YlGnBu", cbar=True)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_wordclouds_by_class(df, text_col, label_col, output_dir):
    """按类别生成词云 (4:3 比例)"""
    classes = df[label_col].unique()
    for cls in classes:
        if not cls or pd.isna(cls): continue
        texts = " ".join(df[df[label_col] == cls][text_col].astype(str))
        words = " ".join(jieba.lcut(texts))
        
        if not words.strip(): continue
        
        font_path = find_cjk_font()
        if not font_path:
            print("警告：未找到中文字体，词云中的中文可能显示为方块")
            
        wc = WordCloud(
            font_path=font_path,
            background_color=None, mode="RGBA",
            width=1200, height=900 # 4:3
        ).generate(words)
        
        plt.figure(figsize=(12, 9))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"类别 [{cls}] 词云")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"wordcloud_{cls}.png"), dpi=300)
        plt.close()
