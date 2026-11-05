from flask import Flask, render_template, jsonify, request, send_from_directory, Response
import os
import json
import subprocess
import signal
import sys
import time

app = Flask(__name__)

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")

# 全局进程对象，用于命令发射器
current_process = None

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/config_editor')
def config_editor():
    return render_template('config_editor.html')

@app.route('/command_launcher')
def command_launcher():
    # 把当前解释器路径传给页面，避免依赖用户自建的虚拟环境目录
    return render_template('command_launcher.html',
                           python_exe=sys.executable.replace('\\', '/'))

# --- 接口：数据可视化 (Gallery) ---
@app.route('/api/gallery')
def get_gallery():
    """获取 output 目录下的文件结构"""
    structure = {}
    for root, dirs, files in os.walk(OUTPUT_DIR):
        rel_path = os.path.relpath(root, OUTPUT_DIR)
        if rel_path == ".": continue
        
        # 统一路径分隔符为正斜杠，防止前端路径拼接错误
        rel_path_web = rel_path.replace("\\", "/")
        
        # 过滤出图片、html、json
        items = []
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.html', '.json')):
                items.append(f)
        
        if items:
            structure[rel_path_web] = items
    return jsonify(structure)

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# --- 接口：配置管理 ---
@app.route('/api/configs')
def list_configs():
    # 本地密钥文件不出现在列表里，避免在页面上暴露
    files = [f for f in os.listdir(CONFIGS_DIR)
             if f.endswith('.json') and not f.endswith('.local.json')]
    return jsonify(files)

@app.route('/api/config/<name>', methods=['GET', 'POST'])
def handle_config(name):
    if name.endswith('.local.json'):
        return jsonify({"status": "error", "message": "本地密钥文件不支持在页面编辑"}), 403
    path = os.path.join(CONFIGS_DIR, name)
    if request.method == 'GET':
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    else:
        data = request.json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return jsonify({"status": "success"})

# --- 接口：命令发射器 ---
@app.route('/api/launch', methods=['POST'])
def launch_command():
    global current_process
    if current_process and current_process.poll() is None:
        return jsonify({"status": "error", "message": "已经有一个任务在运行中"})
    
    data = request.json
    cmd = data.get("command") # 接收前端拼装好的命令列表
    
    # 模拟发射，实际使用时需要将 venv 路径处理好
    try:
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            cwd=BASE_DIR,
            shell=True
        )
        return jsonify({"status": "success", "pid": current_process.pid})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/logs')
def stream_logs():
    def generate():
        global current_process
        if not current_process:
            yield "data: 等待任务启动...\n\n"
            return
        
        while current_process.poll() is None:
            line = current_process.stdout.readline()
            if line:
                yield f"data: {line}\n\n"
            else:
                time.sleep(0.1)
        
        # 读取剩余输出
        for line in current_process.stdout:
            yield f"data: {line}\n\n"
        yield "data: [任务结束]\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # 仅允许本机访问，确保安全
    app.run(host='127.0.0.1', port=5000, debug=True)
