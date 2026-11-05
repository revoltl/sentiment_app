"""跨平台中文字体查找，供词云与图表使用。"""

import os
import platform

_WINDOWS = ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]
_MACOS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]
_LINUX = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

_FAMILIES = [
    "SimHei", "Microsoft YaHei", "PingFang SC",
    "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
]


def find_cjk_font():
    """这个是字体查找函数，功能是返回当前系统可用的中文字体路径，找不到时返回 None。"""
    local = "simhei.ttf"
    if os.path.exists(local):
        return local
    system = platform.system()
    if system == "Windows":
        candidates = _WINDOWS
    elif system == "Darwin":
        candidates = _MACOS
    else:
        candidates = _LINUX
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def cjk_font_families():
    """这个是字体名函数，功能是返回 matplotlib 可尝试的中文字体名列表。"""
    return list(_FAMILIES)
