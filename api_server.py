from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import jwt
import datetime
import os
import hashlib
import aiofiles
import time
import logging
from whisper_service import WhisperService

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper API", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置
SECRET_KEY = 'your-secret-key-change-in-production'
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# 固定API Token配置（用于自动化调用）
API_TOKENS = {
    'whisper-api-key-2024': 'automation',
    'test-token': 'test'
}

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 简单的用户认证（生产环境应使用数据库）
USERS = {
    'admin': hashlib.sha256('admin123'.encode()).hexdigest(),
    'user': hashlib.sha256('user123'.encode()).hexdigest()
}

whisper_service = WhisperService()
security = HTTPBearer()

def create_access_token(data: dict):
    """创建 JWT token"""
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证 JWT token 或固定 API token"""
    token = credentials.credentials
    
    # 首先检查是否为固定API token
    if token in API_TOKENS:
        return f"api_user_{API_TOKENS[token]}"
    
    # 然后验证JWT token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("user")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
async def login(request: LoginRequest):
    """用户登录接口"""
    password_hash = hashlib.sha256(request.password.encode()).hexdigest()
    
    if request.username in USERS and USERS[request.username] == password_hash:
        access_token = create_access_token(data={"user": request.username})
        return {
            "token": access_token,
            "user": request.username
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )

@app.get("/api/model/info")
async def get_model_info(current_user: str = Depends(verify_token)):
    """获取模型信息"""
    try:
        info = whisper_service.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    beam_size: int = Form(5),
    current_user: str = Depends(verify_token)
):
    """语音转文字接口"""
    start_time = time.time()
    
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    # 检查文件大小
    if audio.size and audio.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # 保存上传的文件
    file_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    
    try:
        logger.info(f"用户 {current_user} 开始转录音频文件: {audio.filename} (大小: {audio.size} bytes)")
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await audio.read()
            await f.write(content)
        
        # 进行转录
        transcribe_start = time.time()
        result = whisper_service.transcribe(file_path, beam_size)
        transcribe_time = time.time() - transcribe_start
        
        total_time = time.time() - start_time
        logger.info(f"转录完成 - 文件: {audio.filename}, 转录耗时: {transcribe_time:.2f}s, 总耗时: {total_time:.2f}s, 片段数: {len(result.get('segments', []))}")
        
        return result
    
    except Exception as e:
        logger.error(f"转录失败 - 文件: {audio.filename}, 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"已清理临时文件: {file_path}")

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "whisper-api"}

# 挂载静态文件
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == '__main__':
    import uvicorn
    print("Starting Whisper API Server with FastAPI...")
    print("Default users:")
    print("  Username: admin, Password: admin123")
    print("  Username: user, Password: user123")
    uvicorn.run(app, host="0.0.0.0", port=5000)