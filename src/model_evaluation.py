import argparse
import os
import pandas as pd
import sys
import json

# 设置 PYTHONPATH
sys.path.append(os.getcwd())

from src.utils.eval_utils import (
    calculate_classification_metrics, 
    calculate_topic_metrics, 
    calculate_aspect_metrics,
    plot_confusion_matrix, 
    save_metrics_to_json
)

def main():
    parser = argparse.ArgumentParser(description="LLM NLP Pipeline - 模型评估分支流程")
    
    # 必需参数
    parser.add_argument("--input_csv", required=True, help="包含推理结果和真实标签的 CSV 路径")
    parser.add_argument("--mission", choices=["class", "aspect", "topic"], required=True, help="任务类型")
    parser.add_argument("--output_dir", required=True, help="评估结果输出目录")
    parser.add_argument("--task_name", required=True, help="任务名称，用于图表标题")
    
    # 标签与映射参数
    parser.add_argument("--label_list", required=True, help="评估时采用的标签顺序列表（逗号分隔）")
    parser.add_argument("--mapping", help="真实标签到评估标签的映射字典，JSON 格式")
    parser.add_argument("--topic_mapping", help="真实主题到评估主题的映射字典 (针对 topic 任务)")

    # 任务特定列参数 (class)
    parser.add_argument("--true_col", help="[class] 真实标签所在的列名")
    parser.add_argument("--pred_col", help="[class] 预测标签所在的列名")

    # 任务特定列参数 (topic)
    parser.add_argument("--true_topic_col", help="[topic] 真实主题所在的列名")
    parser.add_argument("--true_label_col", help="[topic] 真实标签所在的列名")
    parser.add_argument("--pred_topic_col", help="[topic] 预测主题所在的列名")
    parser.add_argument("--pred_label_col", help="[topic] 预测标签所在的列名")

    # 任务特定列参数 (aspect)
    parser.add_argument("--true_cols", help="[aspect] 真实方面列名列表（逗号分隔）")
    parser.add_argument("--pred_cols", help="[aspect] 预测方面列名列表（逗号分隔）")

    args = parser.parse_args()

    # 1. 准备环境
    os.makedirs(args.output_dir, exist_ok=True)
    labels = [l.strip() for l in args.label_list.split(",")]
    mapping = json.loads(args.mapping) if args.mapping else None
    topic_mapping = json.loads(args.topic_mapping) if args.topic_mapping else None
    
    # 2. 读取数据
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"找不到输入文件: {args.input_csv}")
    
    df = pd.read_csv(args.input_csv)
    
    # 3. 根据任务类型执行评估
    print(f">>> 正在执行 {args.mission} 任务评估: {args.task_name}")
    
    if args.mission == "class":
        if not args.true_col or not args.pred_col:
            raise ValueError("class 任务必须提供 --true_col 和 --pred_col")
        
        y_true_raw = df[args.true_col].fillna("空值").astype(str).str.strip()
        y_true = y_true_raw.map(lambda x: mapping.get(str(x), "未知")) if mapping else y_true_raw
        y_pred = df[args.pred_col].fillna("空值").astype(str).str.strip()
        
        metrics = calculate_classification_metrics(y_true, y_pred, labels)
        plot_confusion_matrix(y_true, y_pred, labels, os.path.join(args.output_dir, "confusion_matrix.png"), f"{args.task_name} - 混淆矩阵")

    elif args.mission == "topic":
        if not all([args.true_topic_col, args.true_label_col, args.pred_topic_col, args.pred_label_col]):
            raise ValueError("topic 任务必须提供主题和标签的真实/预测列名")
        
        # 应用映射到真实主题列
        if topic_mapping:
            df[args.true_topic_col] = df[args.true_topic_col].fillna("空值").astype(str).map(lambda x: topic_mapping.get(str(x), "未知"))
        
        # 应用映射到真实标签列
        if mapping:
            df[args.true_label_col] = df[args.true_label_col].fillna("空值").astype(str).map(lambda x: mapping.get(str(x), "未知"))
        
        metrics = calculate_topic_metrics(df, args.true_topic_col, args.true_label_col, args.pred_topic_col, args.pred_label_col, labels)
        # Topic 任务暂不生成混淆矩阵，因为类别组合可能过多
        
    elif args.mission == "aspect":
        if not args.true_cols or not args.pred_cols:
            raise ValueError("aspect 任务必须提供 --true_cols 和 --pred_cols")
        
        true_cols = [c.strip() for c in args.true_cols.split(",")]
        pred_cols = [c.strip() for c in args.pred_cols.split(",")]
        
        if len(true_cols) != len(pred_cols):
            raise ValueError("真实列数与预测列数不一致")
            
        # 应用映射到所有真实列
        if mapping:
            for col in true_cols:
                df[col] = df[col].fillna("空值").astype(str).map(lambda x: mapping.get(str(x), "未知"))
        
        metrics = calculate_aspect_metrics(df, true_cols, pred_cols, labels)
        
        # 混淆矩阵展现整体方面预测情况
        y_true_all = []
        y_pred_all = []
        for t_col, p_col in zip(true_cols, pred_cols):
            y_true_all.extend(df[t_col].astype(str).tolist())
            y_pred_all.extend(df[p_col].astype(str).tolist())
        plot_confusion_matrix(y_true_all, y_pred_all, labels, os.path.join(args.output_dir, "confusion_matrix.png"), f"{args.task_name} - 整体方面混淆矩阵")

    # 4. 保存与输出
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")

    json_path = os.path.join(args.output_dir, "metrics.json")
    save_metrics_to_json(metrics, json_path)
    print(f">>> 指标已保存至: {json_path}")

if __name__ == "__main__":
    main()
