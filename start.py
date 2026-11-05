# -*- coding: utf-8 -*-
"""启动网页界面。

用法：
    python start.py
    python start.py --port 8080
"""
import argparse
import os
import sys
import threading
import webbrowser

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from web.app import app


def main():
    ap = argparse.ArgumentParser(description="启动基于大语言模型的消费者评论情感分析系统的网页界面")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，默认只允许本机访问")
    ap.add_argument("--port", type=int, default=5000, help="监听端口")
    ap.add_argument("--debug", action="store_true", help="开启调试模式")
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()
    print("服务地址 http://%s:%d" % (args.host, args.port))
    print("按 Ctrl+C 停止")
    if args.open:
        url = "http://%s:%d" % ("127.0.0.1" if args.host == "0.0.0.0" else args.host, args.port)
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
