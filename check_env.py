# -*- coding: utf-8 -*-
"""环境自检：逐项检查运行本系统所需的条件。

用法：
    python check_env.py
"""
import importlib
import json
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
OK = "通过"
NO = "缺失"

PACKAGES = [
    ("pandas", "数据处理"),
    ("numpy", "数值计算"),
    ("tqdm", "进度显示"),
    ("ollama", "本地模型调用"),
    ("jieba", "中文分词"),
    ("sklearn", "评估指标"),
    ("matplotlib", "绘图"),
    ("seaborn", "绘图"),
    ("wordcloud", "词云"),
    ("pyLDAvis", "主题可视化"),
    ("faiss", "向量检索"),
    ("flask", "网页服务"),
]


def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    print("[%s] Python %d.%d.%d，要求 3.10 及以上" % (OK if ok else NO, v.major, v.minor, v.micro))
    return ok


def check_packages():
    ok = True
    for name, use in PACKAGES:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "")
            print("[%s] %-12s %-10s %s" % (OK, name, use, ver))
        except Exception as e:
            ok = False
            print("[%s] %-12s %-10s %s" % (NO, name, use, e))
    return ok


def check_ollama():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print("[%s] Ollama 服务未响应：%s" % (NO, e))
        print("       请先启动 Ollama，再重新运行本脚本")
        return False
    names = [m.get("name", "") for m in data.get("models", [])]
    print("[%s] Ollama 服务在线，已安装 %d 个模型" % (OK, len(names)))
    for want in ["qwen3:8b", "qwen3-embedding:8b"]:
        hit = any(n == want or n.startswith(want.split(":")[0]) for n in names)
        print("       %s %s" % ("有" if hit else "无", want))
    return True


def check_dirs():
    need = ["configs", "prompts", "src", "web", "data/samples", "output", "logs"]
    ok = True
    for d in need:
        p = os.path.join(BASE, d)
        exists = os.path.isdir(p)
        ok = ok and exists
        print("[%s] 目录 %s" % (OK if exists else NO, d))
    return ok


def main():
    print("检查目录 %s\n" % BASE)
    results = [check_python(), check_dirs(), check_packages(), check_ollama()]
    print()
    if all(results):
        print("全部通过，可以运行 python start.py 启动网页界面。")
    else:
        print("存在未通过项，请按上面的提示处理。")


if __name__ == "__main__":
    main()
