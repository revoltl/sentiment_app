import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_classification_metrics(y_true, y_pred, labels):
    """通用分类指标计算 (Accuracy, Precision, Recall, F1)"""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    }
    return metrics

def calculate_topic_metrics(df, true_topic_col, true_label_col, pred_topic_col, pred_label_col, labels):
    """
    Topic 评估逻辑：(Topic, Label) 二元组必须完全匹配。
    我们将 (Topic, Label) 拼接为单个字符串进行评估。
    """
    y_true = df[true_topic_col].fillna("未知").astype(str) + "_" + df[true_label_col].fillna("未知").astype(str)
    y_pred = df[pred_topic_col].fillna("未知").astype(str) + "_" + df[pred_label_col].fillna("未知").astype(str)
    
    # 获取所有可能的 (Topic, Label) 组合用于指标计算
    unique_labels = sorted(list(set(y_true.unique().astype(str)) | set(y_pred.unique().astype(str))))
    
    return calculate_classification_metrics(y_true, y_pred, unique_labels)

def calculate_aspect_metrics(df, true_cols, pred_cols, labels):
    """
    Aspect 评估逻辑：全量方面比对。
    将多列数据展开为一维列表，对比每一对 (Row, Aspect) 的预测准确性。
    """
    y_true_all = []
    y_pred_all = []
    
    for t_col, p_col in zip(true_cols, pred_cols):
        y_true_all.extend(df[t_col].fillna("未提及").astype(str).tolist())
        y_pred_all.extend(df[p_col].fillna("未提及").astype(str).tolist())
    
    # labels 应该包含 "未提及" + 情感标签
    return calculate_classification_metrics(y_true_all, y_pred_all, labels)

def plot_confusion_matrix(y_true, y_pred, labels, output_path, title="混淆矩阵"):
    """
    绘制混淆矩阵并保存。
    风格：4:3 比例，透明背景，大字体，带标签。
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 4:3 比例 (例如 8:6 或 12:9)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                ax=ax, annot_kws={"size": 14})
    
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel('预测标签 (Predicted)', fontsize=14)
    ax.set_ylabel('真实标签 (Actual)', fontsize=14)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 紧凑布局并设置透明背景
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, transparent=True)
    plt.close(fig)

def save_metrics_to_json(metrics, output_path):
    """保存指标到 JSON 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
