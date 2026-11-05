import argparse
import os
import pandas as pd
import json
import sys
import io

# 强制设置输出编码为 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 设置 PYTHONPATH 确保可以导入 src 模块
sys.path.append(os.getcwd())

from src.utils.parser_utils import (
    extract_json, align_labels, plot_pie_and_bar, 
    plot_heatmap, plot_wordclouds_by_class
)

def main():
    parser = argparse.ArgumentParser(description="成果解析器：解析 LLM 输出并可视化")
    parser.add_argument("--input_csv", required=True, help="主流程输出的 CSV 路径")
    parser.add_argument("--mission", choices=["class", "aspect", "topic"], required=True, help="任务类型")
    parser.add_argument("--label_list", required=True, help="逗号分隔的合法标签列表")
    parser.add_argument("--aspect_list", help="逗号分隔的合法方面列表 (仅 aspect 任务)")
    parser.add_argument("--topic_list", help="逗号分隔的合法主题列表 (仅 topic 任务)")
    parser.add_argument("--output_dir", required=True, help="分析结果保存目录")
    parser.add_argument("--text_col", default="comment", help="评论文本列名，默认 comment")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    labels = [x.strip() for x in args.label_list.split(",") if x.strip()]
    aspects = [x.strip() for x in args.aspect_list.split(",") if x.strip()] if args.aspect_list else None
    topics = [x.strip() for x in args.topic_list.split(",") if x.strip()] if args.topic_list else None
    
    print(f"正在读取结果: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    if "llm_raw_output" not in df.columns:
        print("错误：CSV 中缺少 llm_raw_output 列")
        return

    parsed_results = []
    extract_failed_count = 0
    empty_count = 0
    
    for idx, row in df.iterrows():
        raw_out = row["llm_raw_output"]
        if pd.isna(raw_out) or not str(raw_out).strip():
            empty_count += 1
            parsed_results.append(None)
            continue
            
        json_obj = extract_json(raw_out)
        if json_obj is None:
            extract_failed_count += 1
            parsed_results.append(None)
            continue
            
        aligned = align_labels(json_obj, args.mission, labels, aspects, topics)
        parsed_results.append(aligned)
        
    if empty_count > 0:
        print(f"报告：检测到 {empty_count} 条空输出。")
    if extract_failed_count > 0:
        print(f"报告：检测到 {extract_failed_count} 条 JSON 提取失败。")

    # 处理解析后的数据进行可视化
    if args.mission == "class":
        df["parsed_label"] = [p if isinstance(p, str) else "" for p in parsed_results]
        plot_pie_and_bar(df["parsed_label"], "整体情感分布", os.path.join(args.output_dir, "class_distribution.png"))
        plot_wordclouds_by_class(df, args.text_col, "parsed_label", args.output_dir)
        
    elif args.mission == "aspect":
        # 1. 展开为长表用于绘图
        flattened = []
        for i, res in enumerate(parsed_results):
            if isinstance(res, list):
                for item in res:
                    flattened.append({
                        "id": df.iloc[i].get("id", i),
                        "text": df.iloc[i][args.text_col],
                        "aspect": item["aspect"],
                        "label": item["label"]
                    })
        fdf = pd.DataFrame(flattened)
        
        # 2. 稀疏列解析保存 (重要：为每个 aspect 创建独立列)
        if aspects:
            for asp in aspects:
                col_name = f"pred_{asp}"
                pred_vals = []
                for res in parsed_results:
                    val = "未提及" # 默认值
                    if isinstance(res, list):
                        for item in res:
                            if item["aspect"] == asp:
                                val = item["label"]
                                break
                    pred_vals.append(val)
                df[col_name] = pred_vals

        if not fdf.empty:
            plot_heatmap(fdf, "aspect", "label", "方面*情感交叉分析", os.path.join(args.output_dir, "aspect_emotion_heatmap.png"))
            plot_pie_and_bar(fdf["aspect"], "方面分布", os.path.join(args.output_dir, "aspect_distribution.png"))
            top_aspects = fdf["aspect"].value_counts().head(5).index
            for asp in top_aspects:
                asp_df = fdf[fdf["aspect"] == asp]
                plot_pie_and_bar(asp_df["label"], f"方面[{asp}] 情感分布", os.path.join(args.output_dir, f"aspect_detail_{asp.replace('#','_')}.png"))
        
    elif args.mission == "topic":
        # 1. 展开为长表用于绘图
        flattened = []
        pred_topics = []
        pred_labels = []
        for i, res in enumerate(parsed_results):
            if isinstance(res, dict):
                flattened.append({
                    "id": df.iloc[i].get("id", i),
                    "text": df.iloc[i][args.text_col],
                    "topic": res["topic"],
                    "label": res["label"]
                })
                pred_topics.append(res["topic"])
                pred_labels.append(res["label"])
            else:
                pred_topics.append("")
                pred_labels.append("")
        
        # 2. 双列解析保存
        df["pred_topic"] = pred_topics
        df["pred_label"] = pred_labels

        fdf = pd.DataFrame(flattened)
        if not fdf.empty:
            plot_heatmap(fdf, "topic", "label", "主题*安全性交叉分析", os.path.join(args.output_dir, "topic_security_heatmap.png"))
            plot_pie_and_bar(fdf["topic"], "主题分布", os.path.join(args.output_dir, "topic_distribution.png"))
            for tpc in fdf["topic"].unique():
                tpc_df = fdf[fdf["topic"] == tpc]
                plot_pie_and_bar(tpc_df["label"], f"主题[{tpc}] 安全分布", os.path.join(args.output_dir, f"topic_detail_{tpc}.png"))

    # 保存解析后的 CSV (保存为原文件名，但在 output_dir 下)
    output_csv = os.path.join(args.output_dir, os.path.basename(args.input_csv))
    df["parsed_result"] = [json.dumps(p, ensure_ascii=False) if p else "" for p in parsed_results]
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"解析后的结果已保存至: {output_csv}")
    print(f"分析图表已保存至目录: {args.output_dir}")

if __name__ == "__main__":
    main()
