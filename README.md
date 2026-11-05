# 基于大语言模型的消费者评论情感分析系统

面向经济管理研究的通用评论情感分析工具，以本地大语言模型完成情感判断。系统把数据准备、情感判断、结果统计与评估拆成相互独立的模块，通过标准化的 CSV 与 JSON 文件通信，支持单独运行，也支持串成一条完整流程。

本工程是大学生创新创业训练计划项目的软件成果，目标是给经济管理类研究提供一个可复用的情感分析工具。

## 一、能做什么

1. 读取评论 CSV，按要求拼装提示词，调用本地大语言模型给出情感标签。
2. 支持三类任务：整句情感分类、方面级情感判定、主题与攻击性联合判定。
3. 支持三种增强方式：语义检索示例（RAG）、联网词条释义、主题预分析（LDA）。
4. 解析模型输出为结构化标签，计算准确率、宏平均 F1 与混淆矩阵。
5. 在网页上发起任务、查看实时日志、浏览生成的图表。

## 二、环境要求

- Windows 10 或 11，或 Linux 与 macOS
- Python 3.10 及以上
- Ollama 0.5 及以上，且至少下载一个中文能力较好的模型
- 内存 16 GB 以上，有独立显卡时推理更快

本工程在 Windows 11、Python 3.11 与 Ollama 0.35 上完成测试。

## 三、安装

第一步，安装并启动 Ollama，下载模型。

```
ollama pull qwen3:8b
ollama pull qwen3-embedding:8b
```

第二个模型只在开启检索示例时需要，不开启可以跳过。

第二步，安装 Python 依赖。

```
pip install -r requirements.txt
```

第三步，检查依赖是否齐全。

```
python check_env.py
```

脚本会逐项报告 Python 版本、依赖包、Ollama 服务与模型是否就绪。

## 四、快速开始

### 方式一　网页界面

```
python start.py
```

浏览器打开 `http://127.0.0.1:5000`。左侧有三个页面，仪表盘看结果，配置编辑器改参数，任务发射台发起任务。

在任务发射台里，任务类型选「主推理任务」，数据路径保持默认的示例文件，点击发射即可。日志会实时滚动，运行结束后到仪表盘查看结果。

### 方式二　命令行

```
python src/pipeline_main.py ^
  --comment_csv data/samples/class/comments.csv ^
  --bg_info_txt data/samples/class/bg_info.txt ^
  --comment_mission class ^
  --id_col id --comment_col comment ^
  --model_name qwen3:8b ^
  --prompts_dir prompts ^
  --output_csv output/结果分析/sample_class/result.csv ^
  --ollama_config configs/llm_ollama_config.json ^
  --gen_config configs/llm_generation_config.json ^
  --lda_config configs/lda_config.json ^
  --top_k 3 ^
  --label_list "消极,中立,积极" ^
  --sample 10 ^
  --disable_think
```

`--sample 10` 表示只处理前 10 条，去掉该参数则处理全部。`--disable_think` 关闭思维链，速度更快。

## 五、三类任务的参数

### 整句情感分类

```
--comment_mission class --label_list "消极,中立,积极"
```

### 方面级情感判定

```
--comment_mission aspect --label_list "未提及,消极,中立,积极" ^
--aspect_list "Food#Taste,Food#Appearance,Service#Hospitality"
```

方面列表可以按自己的语料改写，逐项用逗号分隔。

### 主题与攻击性判定

```
--comment_mission topic --label_list "安全,攻击性" ^
--topic_list "race,region,gender"
```

示例数据在 `data/samples/` 下，三个任务各有一份 CSV 与一份背景说明。

## 六、三种增强方式

### 语义检索示例

先用 `src/rag_builder.py` 把训练集建成向量索引，再在主流程加 `--is_rag` 与 `--rag_csv`。

```
python src/rag_builder.py --csv_path data/samples/class/rag_source.csv --rag_config configs/rag_config.json
```

数据源 CSV 需要同名的 JSON 描述文件，写明任务类型与列名，格式见示例文件。

### 联网词条释义

在主流程加 `--is_web_search` 与 `--web_config`。搜索服务支持两种，在 `configs/websearch_config.json` 的 `PROVIDER` 字段选择，`deepseek` 或 `tavily`。

密钥不写进配置文件。把密钥填在 `configs/websearch_config.local.json`，或者在运行前设置环境变量 `DEEPSEEK_API_KEY` 与 `TAVILY_API_KEY`，环境变量优先。该本地文件已在忽略清单中，不会进入版本库。

没有配置密钥时该功能自动跳过，控制台给出提示，其余部分不受影响。

### 主题预分析

在主流程加 `--is_analyze_topic`，系统会先用 LDA 抽取主题并生成可视化，再把主题背景写进提示词。

## 七、结果与评估

主流程只做推理，解析与评估是两步后续操作。

第一步，主流程输出 CSV，包含原始输入、拼装后的提示词与模型原始输出。

第二步，解析标签并生成分布图与词云。

```
python src/result_parser.py --input_csv output/结果分析/sample_class/result.csv ^
  --mission class --label_list "消极,中立,积极" ^
  --output_dir output/结果分析/sample_class --text_col comment
```

解析后同一文件会多出 parsed_label 与 parsed_result 两列，同目录生成类别分布图与词云。

第三步，计算指标。

```
python src/model_evaluation.py --input_csv output/结果分析/sample_class/result.csv ^
  --mission class --output_dir output/模型评估/sample_class ^
  --task_name "影评三分类示例" --label_list "消极,中立,积极" ^
  --true_col label --pred_col parsed_label ^
  --mapping "{\"0\":\"消极\",\"1\":\"中立\",\"2\":\"积极\"}"
```

mapping 把数据里的数字标签映射到文字标签，取值按自己的数据调整。

评估结果包括准确率、宏平均 F1、逐类指标与混淆矩阵图。

## 八、目录结构

```
sentiment_app/
├── start.py                 启动网页界面
├── check_env.py             环境自检
├── requirements.txt
├── configs/                 配置文件
├── prompts/                 提示词模板
├── src/
│   ├── pipeline_main.py     主流程：拼装提示词并推理
│   ├── data_understanding.py 数据理解与主题分析
│   ├── rag_builder.py       构建向量索引
│   ├── model_evaluation.py  评估与绘图
│   ├── result_parser.py     方面级结果展开
│   └── utils/               检索、联网、解析、主题等工具
├── web/                     网页界面
├── data/
│   ├── samples/             示例数据
│   ├── stopwords/           停用词表
│   └── raw/                 用户自己的数据
├── output/                  运行结果
└── logs/                    运行日志
```

## 九、常见问题

**提示找不到模型。** 先执行 `ollama list` 确认模型名，再检查 `--model_name` 是否与列表一致。

**推理很慢。** 关闭思维链（`--disable_think`），或改用参数量更小的模型。方面级任务的输出较长，速度明显慢于整句分类。

**联网功能没有生效。** 密钥写在 `configs/websearch_config.local.json` 或环境变量里，`configs/websearch_config.json` 只有空占位。再确认所选服务 `PROVIDER` 与网络是否可用。

**网页打不开。** 确认 5000 端口未被占用，或修改 `start.py` 里的端口。

**中文显示为方框。** 系统缺少中文字体。`src/utils/font_utils.py` 会依次查找 Windows、macOS 与常见 Linux 的字体位置，安装任一中文字体后重新生成图表即可。

## 十、许可

MIT License。学术使用时请引用本项目对应的论文。
