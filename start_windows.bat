@echo off
echo Whisper 语音转文字服务启动脚本 (Windows)
echo ========================================

:: 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

:: 检查并安装依赖
echo 正在检查依赖包...
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装依赖包...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo 错误: 依赖包安装失败
        pause
        exit /b 1
    )
)

:: 创建上传目录
if not exist "uploads" mkdir uploads

:: 启动服务
echo.
echo 启动 Whisper API 服务...
echo 默认账号: admin/admin123 或 user/user123
echo 服务地址: http://localhost:5000
echo API 文档: http://localhost:5000/docs
echo.
echo 按 Ctrl+C 停止服务
echo.

python api_server.py

pause