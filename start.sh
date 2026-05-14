#!/bin/bash
cd "$(dirname "$0")"

echo "🚀 正在安装依赖..."
pip install -r requirements.txt

echo "✅ 依赖安装完成"
echo "🌐 启动服务器..."
echo "📍 访问地址: http://localhost:8000"

uvicorn main:app --reload --host 0.0.0.0 --port 8000
