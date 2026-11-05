import argparse
import os
import pandas as pd
import json
import sys

# 设置 PYTHONPATH 确保可以导入 src 模块
sys.path.append(os.getcwd())

from src.utils.topic_analysis import run_lda_analysis, save_topic_results

def load_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="分支流程：数据理解与主题可视化")
    parser.add_argument("--comment_csv", required=True, help="评论CSV路径")
    parser.add_argument("--comment_col", required=True, help="评论文本列列名")
    parser.add_argument("--lda_config", required=True, help="LDA主题分析配置路径")
    parser.add_argument("--output_dir", required=True, help="可视化结果输出目录")
    parser.add_argument("--n_topics", type=int, help="覆盖配置中的主题数量")
    
    args = parser.parse_args()

    # 1. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"开始数据理解流程，输出目录: {args.output_dir}")

    # 2. 读取数据
    df = pd.read_csv(args.comment_csv)
    texts = df[args.comment_col].astype(str).tolist()

    # 3. 加载配置
    lda_config = load_json(args.lda_config)
    if args.n_topics:
        lda_config["lda_model"]["n_topics"] = args.n_topics
        print(f"命令行覆盖 n_topics: {args.n_topics}")

    # 4. 执行分析与可视化
    # 我们直接使用全量数据进行分析
    topic_results_str = run_lda_analysis(
        texts=texts,
        lda_config=lda_config,
        visualize=True,
        output_dir=args.output_dir
    )

    # 5. 保存主题分析结果字符串，以便 main 流程加载
    result_json_path = os.path.join(args.output_dir, "topic_analysis_result.json")
    save_topic_results(topic_results_str, result_json_path)

    print(f"数据理解流程执行完毕。")
    print(f"可视化图表已保存至: {args.output_dir}")
    print(f"主题分析结果已保存至: {result_json_path}")

if __name__ == "__main__":
    main()
