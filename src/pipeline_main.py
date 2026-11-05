import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any

# 设置 PYTHONPATH
sys.path.append(os.getcwd())

import ollama
import pandas as pd
from tqdm import tqdm
from src.utils.topic_analysis import run_lda_analysis, load_topic_results
from src.utils.rag_utils import OllamaEmbedder, VectorStore, format_few_shot_examples
from src.utils.web_utils import WebSearcher


# 这个是日志初始化函数，功能是初始化日志记录器并同时输出到文件和控制台。
def setup_logger(log_dir: str, model_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_model = model_name.replace(":", "-")
    log_file = os.path.join(log_dir, f"pipeline_main_{sanitized_model}_{timestamp}.log")

    logger = logging.getLogger("PIPELINE_MAIN")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    return logger


# 这个是JSON读取函数，功能是读取JSON文件并返回字典对象。
def load_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_web_config(path: str) -> dict:
    """读取联网配置，并用本地覆盖文件与环境变量补充密钥。

    随仓库分发的配置不含任何密钥。本地密钥可以放在同名的 .local.json，
    也可以放在环境变量 TAVILY_API_KEY 与 DEEPSEEK_API_KEY 中，后者优先。
    """
    cfg = load_json(path)
    base, ext = os.path.splitext(path)
    local_path = base + ".local" + ext
    if os.path.exists(local_path):
        cfg.update(load_json(local_path))
    for key in ("TAVILY_API_KEY", "DEEPSEEK_API_KEY"):
        value = os.environ.get(key)
        if value:
            cfg[key] = value
    return cfg


# 这个是模板键值读取函数，功能是读取模板JSON中的指定键并返回字符串值。
def read_template_value(file_path: str, key: str) -> str:
    data = load_json(file_path)
    if key not in data:
        raise KeyError(f"模板文件缺少键: {key}, 文件: {file_path}")
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"模板键值必须是字符串: {key}, 文件: {file_path}")
    return value


# 这个是背景信息读取函数，功能是读取背景信息文本并返回去除首尾空白后的内容。
def read_bg_info(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# 这个是推理参数合并函数，功能是合并Ollama配置与生成参数为最终推理参数。
def merge_ollama_options(ollama_config: dict, generation_config: dict) -> dict:
    # 参数合并逻辑严格参考 参考-2_运行LLM.py
    # 严格移除默认值，缺少键则报错
    final_options = ollama_config["options"].copy()
    if "temperature" in generation_config:
        final_options["temperature"] = generation_config["temperature"]
    if "max_new_tokens" in generation_config:
        final_options["num_predict"] = generation_config["max_new_tokens"]
    if "top_p" in generation_config:
        final_options["top_p"] = generation_config["top_p"]
    if "top_k" in generation_config:
        final_options["top_k"] = generation_config["top_k"]
    if "repetition_penalty" in generation_config:
        final_options["repeat_penalty"] = generation_config["repetition_penalty"]
    return final_options


# 这个是思维链提取函数，功能是从模型响应中提取思维链并分离最终回答内容。
def extract_reasoning(content: str, response_message) -> tuple[str, str]:
    reasoning_content = ""
    if hasattr(response_message, "thinking") and response_message.thinking:
        reasoning_content = response_message.thinking
    elif "<think>" in content:
        parts = content.split("<think>")
        if len(parts) > 1:
            inner = parts[1]
            if "</think>" in inner:
                reasoning_part, content_part = inner.split("</think>", 1)
                reasoning_content = reasoning_part.strip()
                content = content_part.strip()
    return content, reasoning_content


# 这个是大模型调用函数，功能是调用Ollama进行一次对话推理并返回结果与思维链。
def call_llm(messages: list[dict], model_name: str, ollama_config: dict, generation_config: dict, disable_think: bool) -> tuple[str, str]:
    final_options = merge_ollama_options(ollama_config, generation_config)
    # 严格移除默认值，必须在配置中指定
    current_think = ollama_config["think"]
    if disable_think:
        current_think = False

    response = ollama.chat(
        model=model_name,
        messages=messages,
        stream=False,
        options=final_options,
        keep_alive=ollama_config["keep_alive"],
        think=current_think,
        format=ollama_config["format"],
    )
    content = response.message.content
    return extract_reasoning(content, response.message)


# ===========================
# 待实现占位函数（按主思路保留）
# ===========================

# 这个是主题分析函数，功能是对评论数据做主题分析。
def analyze_topic(comment_df: pd.DataFrame, comment_col: str, lda_config: dict) -> str:
    """使用 LDA 算法提取文本中的潜在主题"""
    texts = comment_df[comment_col].astype(str).tolist()
    return run_lda_analysis(texts, lda_config)


# 这个是方面列表生成函数，功能是根据主题自动生成方面列表。
def llm_analyze_aspect(comment_topic: str, model_name: str, lda_config: dict, ollama_config: dict, generation_config: dict) -> list[str]:
    """调用 LLM 根据主题提取结果生成 方面列表 (aspect_list)"""
    template = lda_config["prompts"]["llm_analyze_aspect"]
    n_topics = lda_config["lda_model"]["n_topics"]
    prompt = template.format(comment_topic=comment_topic, n_topics=n_topics)
    messages = [{"role": "user", "content": prompt}]
    # 临时禁用思维链以获取干净的列表
    content, _ = call_llm(messages, model_name, ollama_config, generation_config, disable_think=True)
    return [x.strip() for x in content.split(",") if x.strip()]


# 这个是主题候选生成函数，功能是根据主题自动生成主题候选列表。
def llm_analyze_topic(comment_topic: str, model_name: str, lda_config: dict, ollama_config: dict, generation_config: dict) -> list[str]:
    """调用 LLM 根据主题提取结果生成 主题候选列表 (topic_list)"""
    template = lda_config["prompts"]["llm_analyze_topic"]
    n_topics = lda_config["lda_model"]["n_topics"]
    prompt = template.format(comment_topic=comment_topic, n_topics=n_topics)
    messages = [{"role": "user", "content": prompt}]
    # 临时禁用思维链以获取干净的列表
    content, _ = call_llm(messages, model_name, ollama_config, generation_config, disable_think=True)
    return [x.strip() for x in content.split(",") if x.strip()]


# 这个是RAG检索函数，功能是执行RAG检索并返回few-shot示例文本。
def rag(comment: str, top_k: int, embedder: OllamaEmbedder, vector_store: VectorStore, rag_sidecar: Dict[str, Any]) -> str:
    """执行 RAG 检索并返回格式化的 few-shot 示例"""
    query_vector = embedder.embed_text(comment)
    retrieved_docs = vector_store.search(query_vector, top_k=top_k)
    return format_few_shot_examples(retrieved_docs, rag_sidecar)


# 这个是词库匹配函数，功能是进行结构化词库匹配并返回词解释文本（当前为占位实现）。
def sql_match(comment: str) -> str:
    # TODO: 接入结构化词库匹配模块
    _ = comment
    # 词库模块未接入时返回空串，拼接时会自动省略该段，避免占位文字干扰判断
    return ""


# 这个是联网搜索函数，功能是执行联网搜索并返回背景摘要。
def web_search(comment: str, web_searcher: WebSearcher) -> str:
    """执行联网搜索并返回背景摘要"""
    return web_searcher.run_web_search_flow(comment)


# 这个是任务提示组装函数，功能是按任务类型组装任务标签要求提示词。
def build_label_req(comment_mission: str, mission_tpl: dict, label_list: list[str], aspect_list: list[str], topic_list: list[str]) -> str:
    if comment_mission not in mission_tpl:
        raise KeyError(f"prompt_mission.json 缺少任务键: {comment_mission}")
    template = mission_tpl[comment_mission]
    if comment_mission == "class":
        return template.format(label_list=label_list)
    if comment_mission == "aspect":
        return template.format(aspect_list=aspect_list, label_list=label_list)
    if comment_mission == "topic":
        return template.format(topic_list=topic_list, label_list=label_list)
    raise ValueError(f"不支持的任务类型: {comment_mission}")


# 这个是输出示例组装函数，功能是读取模板后按任务类型和当前词列表进行格式化。
def build_output_example(comment_mission: str, output_eg_tpl: dict, label_list: list[str], aspect_list: list[str], topic_list: list[str]) -> str:
    if comment_mission not in output_eg_tpl:
        raise KeyError(f"prompt_outputEg.json 缺少任务键: {comment_mission}")
    template = output_eg_tpl[comment_mission]
    if not isinstance(template, str):
        raise TypeError(f"prompt_outputEg.json 的 {comment_mission} 模板必须是字符串")

    first_label = label_list[0]
    if comment_mission == "class":
        class_example = json.dumps({"label": first_label}, ensure_ascii=False)
        return template.format(class_example=class_example)
    if comment_mission == "aspect":
        selected_aspects = aspect_list[:3]
        results = [{"aspect": a, "label": first_label} for a in selected_aspects]
        aspect_example = json.dumps({"results": results}, ensure_ascii=False)
        return template.format(aspect_example=aspect_example)
    if comment_mission == "topic":
        first_topic = topic_list[0]
        topic_example = json.dumps({"topic": first_topic, "label": first_label}, ensure_ascii=False)
        return template.format(topic_example=topic_example)
    raise ValueError(f"不支持的任务类型: {comment_mission}")


# 这个是列表解析函数，功能是将逗号分隔字符串解析为列表。
def parse_optional_list(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    return [x.strip() for x in raw_value.split(",") if x.strip()]


# 这个是主流程入口函数，功能是串联主干流程并完成提示词组装、模型推理与结果保存。
def main() -> None:
    # 流程第1步：定义并解析命令行参数，统一主流程运行入口配置。
    parser = argparse.ArgumentParser(description="主干流程：按10段Prompt组装并调用LLM")
    parser.add_argument("--comment_csv", required=True, help="评论CSV路径，至少包含id/comment列")
    parser.add_argument("--bg_info_txt", required=True, help="评论背景信息txt路径")
    parser.add_argument("--comment_mission", choices=["class", "aspect", "topic"], required=True, help="任务类型")
    parser.add_argument("--id_col", required=True, help="ID列列名")
    parser.add_argument("--comment_col", required=True, help="评论文本列列名")
    parser.add_argument("--model_name", required=True, help="Ollama模型名称")
    parser.add_argument("--prompts_dir", required=True, help="提示词模板目录")
    parser.add_argument("--output_csv", required=True, help="结果输出路径")
    parser.add_argument("--ollama_config", required=True, help="Ollama配置路径")
    parser.add_argument("--gen_config", required=True, help="生成参数配置路径")
    parser.add_argument("--lda_config", required=True, help="LDA主题分析配置路径")
    parser.add_argument("--rag_config", help="RAG全局配置路径")
    parser.add_argument("--rag_csv", help="RAG 数据源 CSV 路径 (需存在同名 .json)")
    parser.add_argument("--web_config", help="WebSearch 配置文件路径")
    parser.add_argument("--disable_think", action="store_true", help="禁用思维链")
    parser.add_argument("--sample", type=int, help="仅处理前N条")
    parser.add_argument("--top_k", type=int, required=True, help="RAG召回条数")
    parser.add_argument("--is_analyze_topic", action="store_true", help="是否开启主题预分析")
    parser.add_argument("--is_llm_analyze_aspect", action="store_true", help="方面级任务是否由LLM生成aspect_list")
    parser.add_argument("--is_llm_analyze_topic", action="store_true", help="主题任务是否由LLM生成topic_list")
    parser.add_argument("--is_rag", action="store_true", help="是否开启RAG检索")
    parser.add_argument("--is_web_search", action="store_true", help="是否开启 WebSearch 联网搜索")
    parser.add_argument("--label_list", help="逗号分隔标签列表")
    parser.add_argument("--aspect_list", help="逗号分隔方面列表")
    parser.add_argument("--topic_list", help="逗号分隔主题列表")
    parser.add_argument("--topic_result_path", help="预计算的主题分析结果 JSON 路径")
    parser.add_argument("--n_topics", type=int, help="覆盖配置中的主题数量")
    args = parser.parse_args()

    # 流程第2步：初始化日志，记录本次运行过程与错误信息。
    logger = setup_logger("logs", args.model_name)
    logger.info("开始执行主干流程 pipeline_main.py")

    # 流程第3步：加载LLM相关配置（Ollama配置与生成参数配置）及LDA配置。
    ollama_config = load_json(args.ollama_config)
    generation_config = load_json(args.gen_config)
    lda_config = load_json(args.lda_config)
    if args.n_topics:
        lda_config["lda_model"]["n_topics"] = args.n_topics
        logger.info(f"命令行覆盖 n_topics: {args.n_topics}")
    
    # 流程第3.5步：加载 RAG 配置（如果启用）
    rag_embedder = None
    rag_vector_store = None
    rag_sidecar = None
    if args.is_rag:
        if not args.rag_config or not args.rag_csv:
            raise ValueError("开启 RAG 必须提供 --rag_config 和 --rag_csv")
        
        # 1. 加载全局配置
        rag_global_conf = load_json(args.rag_config)
        
        # 2. 加载 sidecar 配置
        json_path = args.rag_csv.rsplit('.', 1)[0] + '.json'
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到 RAG sidecar JSON 配置文件: {json_path}")
        rag_sidecar = load_json(json_path)
        
        # 3. 初始化索引名称和路径
        index_name = os.path.basename(args.rag_csv).rsplit('.', 1)[0]
        
        # 4. 初始化组件
        rag_embedder = OllamaEmbedder(rag_global_conf["embedding"]["model_name"], rag_global_conf["embedding"]["base_url"])
        rag_vector_store = VectorStore(rag_global_conf["vector_store"]["base_dir"], index_name)
        rag_vector_store.load()
        logger.info(f"RAG 模块已初始化，数据源: {args.rag_csv}")

    # 流程第3.6步：初始化 WebSearch 模块（如果启用）
    web_searcher = None
    if args.is_web_search:
        if not args.web_config:
            raise ValueError("开启 WebSearch 必须提供 --web_config")
        web_config = load_web_config(args.web_config)
        web_searcher = WebSearcher(web_config, ollama_config, generation_config)
        logger.info("WebSearch 模块已初始化")

    # 流程第4步：读取评论数据。
    comment_df = pd.read_csv(args.comment_csv)

    # 流程第5步：读取列名参数并做必需字段校验（评论列与ID列）。
    id_col = args.id_col
    comment_col = args.comment_col
    if id_col not in comment_df.columns or comment_col not in comment_df.columns:
        raise ValueError(f"CSV缺少指定列: id_col={id_col}, comment_col={comment_col}")

    # 流程第6步：读取背景信息并确定任务类型。
    comment_bg_info = read_bg_info(args.bg_info_txt)
    comment_mission = args.comment_mission

    # 流程第7步：初始化标签、方面、主题列表，并进行任务相关的参数校验。
    # 严格校验：根据任务类型（class/aspect/topic）检查必需的列表参数，避免参数冗余。
    label_list = parse_optional_list(args.label_list)
    aspect_list = parse_optional_list(args.aspect_list)
    topic_list = parse_optional_list(args.topic_list)

    if not label_list:
        raise ValueError(f"任务 {comment_mission} 必须提供 --label_list")

    if comment_mission == "class":
        # 分类任务仅需 label_list
        if aspect_list:
            logger.warning("分类任务不需要 --aspect_list，将被忽略")
        if topic_list:
            logger.warning("分类任务不需要 --topic_list，将被忽略")
    
    elif comment_mission == "aspect":
        # 方面级任务需要 label_list 和 aspect_list
        if not aspect_list and not args.is_llm_analyze_aspect:
            raise ValueError("方面级任务必须提供 --aspect_list 或开启 --is_llm_analyze_aspect")
        if aspect_list and len(aspect_list) < 3:
            raise ValueError("方面级任务手动提供的 aspect_list 至少包含3个方面")
        if topic_list:
            logger.warning("方面级任务不需要 --topic_list，将被忽略")

    elif comment_mission == "topic":
        # 主题任务需要 label_list 和 topic_list
        if not topic_list and not args.is_llm_analyze_topic:
            raise ValueError("主题任务必须提供 --topic_list 或开启 --is_llm_analyze_topic")
        if aspect_list:
            logger.warning("主题任务不需要 --aspect_list，将被忽略")

    # 流程第8步：按开关执行主题预分析，并在需要时动态生成aspect/topic候选列表。
    comment_topic = None  # 移除空字符串默认值
    
    # 优先加载预计算的主题分析结果
    if args.topic_result_path:
        logger.info(f"正在加载预计算的主题分析结果: {args.topic_result_path}")
        comment_topic = load_topic_results(args.topic_result_path)
        # 如果提供了预计算路径，强制开启 is_analyze_topic 的效果（除非用户显式关闭逻辑，但这里我们认为加载了就要用）
        args.is_analyze_topic = True 
    
    if args.is_analyze_topic and not comment_topic:
        logger.info("执行 LDA 主题预分析...")
        comment_topic = analyze_topic(comment_df, comment_col, lda_config)
        
    if comment_topic:
        logger.info(f"当前使用的主题信息：\n{comment_topic}")
        
        if comment_mission == "aspect" and args.is_llm_analyze_aspect:
            logger.info("调用 LLM 动态生成方面列表...")
            aspect_list = llm_analyze_aspect(comment_topic, args.model_name, lda_config, ollama_config, generation_config)
            logger.info(f"动态生成方面：{aspect_list}")
            
        if comment_mission == "topic" and args.is_llm_analyze_topic:
            logger.info("调用 LLM 动态生成主题列表...")
            topic_list = llm_analyze_topic(comment_topic, args.model_name, lda_config, ollama_config, generation_config)
            logger.info(f"动态生成主题：{topic_list}")

    # 流程第9步：在执行 LLM 逐条推理前，按 sample 参数进行采样（确保 LDA 使用的是全量数据）。
    if args.sample is not None:
        if args.sample <= 0:
            raise ValueError(f"sample 参数必须大于0，当前为: {args.sample}")
        comment_df = comment_df.head(args.sample)
        logger.info(f"采样模式：LDA 已完成全量分析，LLM 仅处理前 {args.sample} 条数据")

    # 流程第10步：加载10个Prompt模板组件，为后续逐条评论组装做准备。
    prompt_role_tpl = read_template_value(os.path.join(args.prompts_dir, "prompt_role.json"), "prompt_role")
    prompt_topic_tpl = read_template_value(os.path.join(args.prompts_dir, "prompt_topic.json"), "prompt_topic")
    prompt_bg_info_tpl = read_template_value(os.path.join(args.prompts_dir, "prompt_bgInfo.json"), "prompt_bgInfo")
    prompt_comment_tpl = read_template_value(os.path.join(args.prompts_dir, "prompt_comment.json"), "prompt_comment")
    prompt_cot_tpl = read_template_value(os.path.join(args.prompts_dir, "prompt_COT.json"), "prompt_COT")
    prompt_mission_tpl = load_json(os.path.join(args.prompts_dir, "prompt_mission.json"))
    prompt_output_eg_tpl = load_json(os.path.join(args.prompts_dir, "prompt_outputEg.json"))
    prompt_rag_tpl = read_template_value(os.path.join(args.prompts_dir, "prompt_ragFewShot.json"), "prompt_ragFewShot")
    prompt_sql_tpl = read_template_value(os.path.join(args.prompts_dir, "prompt_sqlWordMatch.json"), "prompt_sqlWordMatch")
    prompt_web_tpl = read_template_value(os.path.join(args.prompts_dir, "prompt_webSearch.json"), "prompt_webSearch")

    # 流程第10步：初始化结果容器，收集模型输出与思维链。
    llm_predictions: list[str] = []
    reasoning_contents: list[str] = []
    composed_prompts: list[str] = []

    # 流程第11步：逐条评论执行主干流程（组装Prompt -> 调用LLM -> 收集结果）。
    for _, row in tqdm(comment_df.iterrows(), total=len(comment_df), desc="主干流程推理"):
        row_id = row[id_col]
        comment = str(row[comment_col])

        prompt_role = prompt_role_tpl
        prompt_topic = prompt_topic_tpl.format(commentTopic=comment_topic) if args.is_analyze_topic else ""
        prompt_bg_info = prompt_bg_info_tpl.format(commentBgInfo=comment_bg_info)
        prompt_comment = prompt_comment_tpl.format(comment=comment)

        # 组装 RAG few-shot
        if args.is_rag:
            rag_few_shot_val = rag(comment, args.top_k, rag_embedder, rag_vector_store, rag_sidecar)
            prompt_rag_few_shot = prompt_rag_tpl.format(rag_few_shot=rag_few_shot_val)
        else:
            prompt_rag_few_shot = ""

        prompt_cot = prompt_cot_tpl
        prompt_label_req = build_label_req(comment_mission, prompt_mission_tpl, label_list, aspect_list, topic_list)
        prompt_output_eg = build_output_example(comment_mission, prompt_output_eg_tpl, label_list, aspect_list, topic_list)
        sql_hit = sql_match(comment)
        prompt_sql_word_match = prompt_sql_tpl.format(sql_word_match=sql_hit) if sql_hit else ""
        
        # 组装 WebSearch
        if args.is_web_search:
            web_context = web_search(comment, web_searcher)
            # 没有检索到内容时不拼接该段，避免留下空标题
            prompt_web_search = prompt_web_tpl.format(web_search=web_context) if web_context else ""
        else:
            prompt_web_search = ""

        # 流程第11.1步：严格按主思路顺序组装10个组件。
        user_prompt_parts = [
            prompt_role,
            prompt_topic,
            prompt_bg_info,
            prompt_comment,
            prompt_cot,
            prompt_label_req,
            prompt_output_eg,
            prompt_rag_few_shot,
            prompt_sql_word_match,
            prompt_web_search,
        ]
        user_prompt = "\n\n".join([p for p in user_prompt_parts if p])
        composed_prompts.append(user_prompt)

        # 流程第11.2步：构建messages并发起一次模型推理。
        messages = [{"role": "user", "content": user_prompt}]

        try:
            final_content, reasoning_content = call_llm(
                messages=messages,
                model_name=args.model_name,
                ollama_config=ollama_config,
                generation_config=generation_config,
                disable_think=args.disable_think,
            )
            llm_predictions.append(final_content)
            reasoning_contents.append(reasoning_content)
        except Exception as e:
            logger.error(f"推理错误 (id={row_id}): {e}")
            llm_predictions.append("")
            reasoning_contents.append("")

    # 流程第12步：整理输出DataFrame并写入结果CSV。
    output_df = comment_df.copy()
    output_df["composed_user_prompt"] = composed_prompts
    output_df["llm_raw_output"] = llm_predictions
    if any(reasoning_contents):
        output_df["reasoning_content"] = reasoning_contents

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    output_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"结果已保存: {args.output_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"主流程执行失败: {e}")
        raise
