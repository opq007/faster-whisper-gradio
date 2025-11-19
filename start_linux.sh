#!/bin/bash

echo "Whisper 语音转文字服务启动脚本 (Linux)"
echo "========================================"

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查并安装依赖
echo "正在检查依赖包..."
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "正在安装依赖包..."
    python3 -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "错误: 依赖包安装失败"
        exit 1
    fi
fi

# 创建上传目录
mkdir -p uploads

# 启动服务
echo ""
echo "启动 Whisper API 服务..."
echo "默认账号: admin/admin123 或 user/user123"
echo "服务地址: http://localhost:5000"
echo "API 文档: http://localhost:5000/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

python3 api_server.py