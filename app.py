#!/usr/bin/env python3
"""
整合版 Whisper 服务 - 统一 FastAPI + Gradio 界面
- FastAPI REST API with Bearer token auth
- Gradio UI with modern interface
- faster-whisper for ASR
- 统一认证和服务层
- 支持基础转录和高级字幕生成
"""

import os
import io
import sys
import uuid
import shutil
import tempfile
import asyncio
import traceback
import hashlib
import datetime
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import timedelta, datetime

import requests
import subprocess
import jwt
import aiofiles
import json
from faster_whisper import WhisperModel

from fastapi import FastAPI, File, UploadFile, Header, Body, HTTPException, Query, Depends, status, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import gradio as gr

# ----------------------------
# 配置和常量
# ----------------------------
class AppConfig:
    """应用配置类 - 统一管理所有配置参数"""
    
    # 服务配置
    API_TOKEN = os.environ.get("API_TOKEN", "whisper-api-key-2024")  # MUST set in production
    SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 7860))
    
    # URL 配置
    BASE_HOST = os.environ.get("BASE_HOST", "127.0.0.1")  # 用于生成 URL 的主机地址
    BASE_URL = f"http://{BASE_HOST}:{PORT}"
    API_BASE_URL = f"{BASE_URL}/api"
    GRADIO_URL = f"{BASE_URL}/ui"
    DOCS_URL = f"{BASE_URL}/docs"
    
    # Whisper模型配置
    DEFAULT_MODEL = os.environ.get("FW_MODEL", "small")
    DEFAULT_DEVICE = os.environ.get("FW_DEVICE", "cpu")
    DEFAULT_COMPUTE = os.environ.get("FW_COMPUTE", "int8")  # e.g. "int8", "float16", or None
    CPU_THREADS = 8
    BEAM_SIZE = 5
    
    # 文件和目录配置
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'output'
    DEBUG_FOLDER = 'debug'
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    # 任务配置
    JOB_TIMEOUT = 3600  # 1小时超时
    POLLING_INTERVAL = 2.0  # 轮询间隔（秒）
    
    # FFmpeg配置
    FFMPEG_PATHS = [
        "ffmpeg",
        r"D:\programs\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
    ]
    
    # 认证配置
    API_TOKENS = {
        'whisper-api-key-2024': 'automation',
        'test-token': 'test'
    }
    
    USERS = {
        'admin': hashlib.sha256('admin123'.encode()).hexdigest(),
        'user': hashlib.sha256('user123'.encode()).hexdigest()
    }
    
    # 支持的文件格式
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    AUDIO_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']
    
    @classmethod
    def init_directories(cls):
        """初始化必要的目录"""
        for folder in [cls.UPLOAD_FOLDER, cls.OUTPUT_FOLDER, cls.DEBUG_FOLDER]:
            os.makedirs(folder, exist_ok=True)
    
    @classmethod
    def get_supported_extensions(cls):
        """获取支持的文件扩展名"""
        return cls.VIDEO_EXTENSIONS + cls.AUDIO_EXTENSIONS
    
    @classmethod
    def get_api_urls(cls, endpoint: str) -> List[str]:
        """获取 API 端点的多个 URL 地址，用于兼容不同的网络环境"""
        return [
            f"http://127.0.0.1:{cls.PORT}{endpoint}",
            f"http://localhost:{cls.PORT}{endpoint}",
            f"http://0.0.0.0:{cls.PORT}{endpoint}",
            f"http://[::1]:{cls.PORT}{endpoint}"  # IPv6 localhost
        ]

# 初始化配置
config = AppConfig()
config.init_directories()

# 全局变量
JOBS: Dict[str, Dict[str, Any]] = {}

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# FastAPI 应用初始化
# ----------------------------
app = FastAPI(title="整合版 Whisper 语音转文字服务", version="2.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ----------------------------
# 工具类
# ----------------------------
class FileUtils:
    """文件操作工具类"""
    
    @staticmethod
    def get_output_dir() -> Path:
        """获取输出目录"""
        current_dir = Path(__file__).parent
        output_dir = current_dir / config.OUTPUT_FOLDER
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @staticmethod
    def create_job_dir() -> Path:
        """创建任务目录，使用yyyyMMdd-HHMMSS格式命名"""
        output_dir = FileUtils.get_output_dir()
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        job_dir_name = f"job_{date_str}-{time_str}"
        job_dir = output_dir / job_dir_name
        job_dir.mkdir(exist_ok=True)
        return job_dir
    
    @staticmethod
    def generate_job_id() -> str:
        """生成任务ID，使用yyyyMMdd-HHMMSS格式"""
        now = datetime.now()
        return now.strftime("%Y%m%d-%H%M%S")
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """获取文件扩展名"""
        return Path(filename).suffix.lower()
    
    @staticmethod
    def is_video_file(filename: str) -> bool:
        """判断是否为视频文件"""
        return FileUtils.get_file_extension(filename) in config.VIDEO_EXTENSIONS
    
    @staticmethod
    def is_audio_file(filename: str) -> bool:
        """判断是否为音频文件"""
        return FileUtils.get_file_extension(filename) in config.AUDIO_EXTENSIONS
    
    @staticmethod
    def is_supported_file(filename: str) -> bool:
        """判断是否为支持的文件格式"""
        return FileUtils.is_video_file(filename) or FileUtils.is_audio_file(filename)

class SystemUtils:
    """系统工具类"""
    
    _ffmpeg_path = None
    
    @classmethod
    def run_cmd(cls, cmd: List[str]) -> str:
        """执行命令并返回输出"""
        if os.name == 'nt':  # Windows
            cmd = [str(c) for c in cmd]
        
        proc = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            shell=os.name == 'nt'
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return proc.stdout
    
    @classmethod
    def check_ffmpeg_available(cls) -> bool:
        """检查ffmpeg是否可用"""
        if cls._ffmpeg_path:
            return True
        
        for ffmpeg_path in config.FFMPEG_PATHS:
            try:
                subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True)
                cls._ffmpeg_path = ffmpeg_path
                logger.info(f"Found ffmpeg at: {ffmpeg_path}")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        logger.warning("FFmpeg not found in system PATH or configured paths")
        return False
    
    @classmethod
    def get_ffmpeg_path(cls) -> str:
        """获取ffmpeg路径"""
        if not cls.check_ffmpeg_available():
            raise RuntimeError(
                "FFmpeg 未安装或不在 PATH 中。请安装 FFmpeg: https://ffmpeg.org/download.html"
            )
        return cls._ffmpeg_path

class MediaProcessor:
    """媒体处理工具类"""
    
    @staticmethod
    def extract_audio(input_path: Path, output_path: Path, sample_rate: int = 16000):
        """从视频或音频文件中提取音频"""
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()
        
        ffmpeg_path = SystemUtils.get_ffmpeg_path()
        
        cmd = [
            ffmpeg_path, "-y", "-i", str(input_path),
            "-ac", "1", "-ar", str(sample_rate),
            "-vn", "-f", "wav", str(output_path)
        ]
        SystemUtils.run_cmd(cmd)
    
    @staticmethod
    def download_from_url(url: str, output_path: Path, timeout: int = 60):
        """从URL下载文件"""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading URL: {url} to {output_path}")
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            logger.info(f"Downloaded successfully: {output_path}")
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise
    
    @staticmethod
    def mux_softsub(video_path: Path, srt_path: Path, output_path: Path):
        """生成软字幕视频（将SRT字幕嵌入到视频文件中）"""
        video_path = Path(video_path).resolve()
        srt_path = Path(srt_path).resolve()
        output_path = Path(output_path).resolve()
        
        ffmpeg_path = SystemUtils.get_ffmpeg_path()
        
        # 使用相对路径避免Windows路径解析问题
        import os
        original_cwd = os.getcwd()
        
        try:
            # 切换到输出文件所在目录
            output_dir = output_path.parent
            os.chdir(output_dir)
            
            # 使用相对路径
            video_rel = os.path.relpath(video_path, output_dir)
            srt_rel = os.path.relpath(srt_path, output_dir)
            output_rel = output_path.name
            
            # 确保使用正斜杠
            video_rel = video_rel.replace('\\', '/')
            srt_rel = srt_rel.replace('\\', '/')
            
            # 修复软字幕参数 - 使用正确的字幕流映射
            cmd = [
                ffmpeg_path, "-y", "-i", video_rel, "-i", srt_rel,
                "-c", "copy", "-c:s", "mov_text",
                "-disposition:s:0", "0",  # 设置字幕流为默认流
                "-metadata:s:s:0", "language=chi",
                "-map", "0", "-map", "1",  # 映射视频流和字幕流
                output_rel
            ]
            
            SystemUtils.run_cmd(cmd)
            Logger.info(f"软字幕视频生成成功: {output_path}")
            
        except Exception as e:
            Logger.error(f"软字幕视频生成失败: {e}")
            raise
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)
    
    @staticmethod
    def burn_hardsub(video_path: Path, srt_path: Path, output_path: Path):
        """生成硬字幕视频（将字幕直接烧录到视频画面中）"""
        video_path = Path(video_path).resolve()
        srt_path = Path(srt_path).resolve()
        output_path = Path(output_path).resolve()
        
        ffmpeg_path = SystemUtils.get_ffmpeg_path()
        
        # 使用相对路径避免Windows路径解析问题
        import os
        original_cwd = os.getcwd()
        
        try:
            # 切换到输出文件所在目录
            output_dir = output_path.parent
            os.chdir(output_dir)
            
            # 使用相对路径
            video_rel = os.path.relpath(video_path, output_dir)
            srt_rel = os.path.relpath(srt_path, output_dir)
            output_rel = output_path.name
            
            # 确保使用正斜杠
            video_rel = video_rel.replace('\\', '/')
            srt_rel = srt_rel.replace('\\', '/')
            
            cmd = [
                ffmpeg_path, "-y", "-i", video_rel, "-vf", 
                f"subtitles='{srt_rel}'",
                "-c:a", "copy", output_rel
            ]
            
            SystemUtils.run_cmd(cmd)
            Logger.info(f"硬字幕视频生成成功: {output_path}")
            
        except Exception as e:
            Logger.error(f"硬字幕视频生成失败: {e}")
            raise
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)
    
    @staticmethod
    def get_media_duration(file_path: Path) -> float:
        """获取媒体文件的时长（秒）"""
        file_path = Path(file_path).resolve()
        ffmpeg_path = SystemUtils.get_ffmpeg_path()
        
        cmd = [
            ffmpeg_path, "-i", str(file_path), "-f", "null", "-"
        ]
        
        try:
            result = SystemUtils.run_cmd(cmd)
            # 从FFmpeg输出中解析时长信息
            import re
            duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', result)
            if duration_match:
                hours, minutes, seconds = map(float, duration_match.groups())
                return hours * 3600 + minutes * 60 + seconds
            else:
                Logger.warning(f"无法获取媒体时长: {file_path}")
                return 0.0
        except Exception as e:
            Logger.error(f"获取媒体时长失败: {e}")
            return 0.0
    
    @staticmethod
    def merge_audio_video(video_path: Path, audio_path: Path, output_path: Path):
        """将音频合并到视频中（替换原音频）"""
        video_path = Path(video_path).resolve()
        audio_path = Path(audio_path).resolve()
        output_path = Path(output_path).resolve()
        
        ffmpeg_path = SystemUtils.get_ffmpeg_path()
        
        # 使用相对路径避免Windows路径解析问题
        import os
        original_cwd = os.getcwd()
        
        try:
            # 切换到输出文件所在目录
            output_dir = output_path.parent
            os.chdir(output_dir)
            
            # 使用相对路径
            video_rel = os.path.relpath(video_path, output_dir)
            audio_rel = os.path.relpath(audio_path, output_dir)
            output_rel = output_path.name
            
            # 确保使用正斜杠
            video_rel = video_rel.replace('\\', '/')
            audio_rel = audio_rel.replace('\\', '/')
            
            cmd = [
                ffmpeg_path, "-y", "-i", video_rel, "-i", audio_rel,
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", output_rel  # 以较短的流为准
            ]
            
            SystemUtils.run_cmd(cmd)
            Logger.info(f"音视频合并成功: {output_path}")
            
        except Exception as e:
            Logger.error(f"音视频合并失败: {e}")
            raise
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)

class SubtitleGenerator:
    """字幕生成工具类"""
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """格式化时间戳为SRT格式"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((td.total_seconds() - int(td.total_seconds())) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    @staticmethod
    def write_srt(segments, output_path: Path, bilingual: bool = False, translated_segments=None):
        """写入SRT字幕文件"""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                start = SubtitleGenerator.format_timestamp(seg.start)
                end = SubtitleGenerator.format_timestamp(seg.end)
                orig = seg.text.strip()
                
                if bilingual and translated_segments:
                    trans = translated_segments[i-1].text.strip() if i-1 < len(translated_segments) else ""
                    text_block = (orig + "\n" + trans).strip()
                else:
                    text_block = orig
                
                f.write(f"{i}\n{start} --> {end}\n{text_block}\n\n")

class Logger:
    """日志工具类"""
    
    @staticmethod
    def debug(message: str, job_id: str = None):
        """写入调试日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file = Path(__file__).parent / config.DEBUG_FOLDER / "asr_debug.log"
        
        with open(log_file, "a", encoding="utf-8") as f:
            if job_id:
                f.write(f"[{timestamp}] [JOB:{job_id}] {message}\n")
            else:
                f.write(f"[{timestamp}] {message}\n")
    
    @staticmethod
    def info(message: str, job_id: str = None):
        """写入信息日志"""
        if job_id:
            logger.info(f"[JOB:{job_id}] {message}")
        else:
            logger.info(message)
    
    @staticmethod
    def error(message: str, job_id: str = None):
        """写入错误日志"""
        if job_id:
            logger.error(f"[JOB:{job_id}] {message}")
        else:
            logger.error(message)

# ----------------------------
# Whisper 服务类（重构版）
# ----------------------------
class WhisperService:
    """Whisper 语音转文字服务类，支持模型复用和多种转录模式"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config = config
            self.model_cache = {}
            self.initialized = True
    
    async def _load_model(self, model_name: str = None, device: str = None, compute_type: str = None):
        """加载 Whisper 模型（带缓存）"""
        model_name = model_name or self.config.DEFAULT_MODEL
        device = device or self.config.DEFAULT_DEVICE
        compute_type = compute_type or self.config.DEFAULT_COMPUTE
        
        cache_key = (model_name, device, compute_type)
        
        if cache_key in self.model_cache:
            Logger.info(f"Using cached model: {model_name}")
            return self.model_cache[cache_key]
        
        try:
            Logger.info(f"Loading Whisper model: {model_name}")
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None, 
                lambda: WhisperModel(
                    model_name, 
                    device=device, 
                    compute_type=compute_type,
                    cpu_threads=self.config.CPU_THREADS
                )
            )
            self.model_cache[cache_key] = model
            Logger.info(f"Whisper model loaded successfully: {model_name}")
            return model
        except Exception as e:
            Logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    async def transcribe_basic(self, audio_path: str, beam_size: int = None, model_name: str = None) -> Dict[str, Any]:
        """
        基础语音转文字
        
        Args:
            audio_path: 音频文件路径
            beam_size: beam search 大小
            model_name: 模型名称
            
        Returns:
            包含转录结果的字典
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        beam_size = beam_size or self.config.BEAM_SIZE
        
        try:
            model = await self._load_model(model_name)
            
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None, 
                lambda: model.transcribe(audio_path, beam_size=beam_size)
            )
            
            result = {
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    }
                    for segment in segments
                ]
            }
            
            Logger.info(f"Basic transcription completed: {len(result['segments'])} segments")
            return result
            
        except Exception as e:
            Logger.error(f"Basic transcription error: {e}")
            raise
    
    async def transcribe_advanced(
        self, 
        audio_path: Path, 
        model_name: str = None, 
        device: str = None, 
        compute_type: Optional[str] = None, 
        beam_size: int = None, 
        task: Optional[str] = None, 
        word_timestamps: bool = False
    ) -> List:
        """
        高级语音转文字
        
        Args:
            audio_path: 音频文件路径
            model_name: 模型名称
            device: 设备
            compute_type: 计算类型
            beam_size: beam search 大小
            task: 任务类型
            word_timestamps: 是否包含词级时间戳
            
        Returns:
            转录片段列表
        """
        beam_size = beam_size or self.config.BEAM_SIZE
        task = task or "transcribe"
        
        model = await self._load_model(model_name, device, compute_type)
        
        try:
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None, 
                lambda: model.transcribe(
                    str(audio_path), 
                    beam_size=beam_size, 
                    task=task, 
                    word_timestamps=word_timestamps
                )
            )
            
            # 立即将segments转换为列表，避免生成器被消耗
            segments_list = list(segments)
            Logger.info(f"Advanced transcription completed: {len(segments_list)} segments")
            return segments_list
            
        except Exception as e:
            Logger.error(f"Advanced transcription error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "default_model": self.config.DEFAULT_MODEL,
            "default_device": self.config.DEFAULT_DEVICE,
            "default_compute_type": self.config.DEFAULT_COMPUTE,
            "cpu_threads": self.config.CPU_THREADS,
            "cached_models": list(self.model_cache.keys()),
            "available_models": list(self.model_cache.keys()) if self.model_cache else []
        }
    
    def clear_cache(self):
        """清除模型缓存"""
        self.model_cache.clear()
        Logger.info("Model cache cleared")

# 创建全局服务实例
whisper_service = WhisperService()

# ----------------------------
# 任务管理类
# ----------------------------
class JobManager:
    """任务管理器 - 处理异步转录任务"""
    
    def __init__(self):
        self.jobs = JOBS
    
    def create_job(self, input_type: str, **kwargs) -> str:
        """创建新任务"""
        job_id = FileUtils.generate_job_id()
        job_dir = FileUtils.create_job_dir()
        
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "input_type": input_type,
            "created_at": datetime.now(),
            "job_dir": str(job_dir),
            "files": {},
            "result": None,
            "error": None,
            **kwargs
        }
        
        Logger.info(f"Created job {job_id} with input_type: {input_type}")
        return job_id
    
    def update_job_status(self, job_id: str, status: str, **kwargs):
        """更新任务状态"""
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            self.jobs[job_id].update(kwargs)
            Logger.info(f"Updated job {job_id} status to: {status}")
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        return self.jobs.get(job_id)
    
    def add_file(self, job_id: str, file_type: str, file_path: str):
        """添加文件到任务"""
        if job_id in self.jobs:
            self.jobs[job_id]["files"][file_type] = file_path
    
    def set_result(self, job_id: str, result: Dict[str, Any]):
        """设置任务结果"""
        if job_id in self.jobs:
            self.jobs[job_id]["result"] = result
    
    def set_error(self, job_id: str, error: str):
        """设置任务错误"""
        if job_id in self.jobs:
            self.jobs[job_id]["error"] = error
            self.update_job_status(job_id, "failed")

# 创建全局任务管理器
job_manager = JobManager()

# ----------------------------
# 认证服务类
# ----------------------------
class AuthService:
    """认证服务类"""
    
    @staticmethod
    def create_access_token(data: dict) -> str:
        """创建 JWT token"""
        to_encode = data.copy()
        expire = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, config.SECRET_KEY, algorithm="HS256")
    
    @staticmethod
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
        """验证 JWT token 或固定 API token"""
        token = credentials.credentials
        
        # 检查固定API token
        if token in config.API_TOKENS:
            return f"api_user_{config.API_TOKENS[token]}"
        
        # 验证JWT token
        try:
            payload = jwt.decode(token, config.SECRET_KEY, algorithms=["HS256"])
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
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> bool:
        """验证用户名密码"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return config.USERS.get(username) == password_hash

# ----------------------------
# Pydantic 模型
# ----------------------------
class LoginRequest(BaseModel):
    username: str
    password: str

class TranscribeRequest(BaseModel):
    input_type: str  # upload | local | url | separate_audio
    local_path: Optional[str] = None
    url: Optional[str] = None
    model_name: Optional[str] = None
    device: Optional[str] = None
    compute_type: Optional[str] = None
    bilingual: Optional[bool] = True
    beam_size: Optional[int] = None
    word_timestamps: Optional[bool] = False
    burn: Optional[str] = "none"  # none|hard|soft
    out_basename: Optional[str] = None
    # For separate_audio mode
    video_local_path: Optional[str] = None
    video_url: Optional[str] = None
    audio_local_path: Optional[str] = None
    audio_url: Optional[str] = None
    
    def __init__(self, **data):
        # 设置默认值
        if 'model_name' not in data or data['model_name'] is None:
            data['model_name'] = config.DEFAULT_MODEL
        if 'device' not in data or data['device'] is None:
            data['device'] = config.DEFAULT_DEVICE
        if 'beam_size' not in data or data['beam_size'] is None:
            data['beam_size'] = config.BEAM_SIZE
        super().__init__(**data)

class BasicTranscribeRequest(BaseModel):
    """基础转录请求模型 - 支持文件上传、URL和本地路径"""
    input_type: str = "upload"  # upload | url | local
    url: Optional[str] = None
    local_path: Optional[str] = None
    model_name: Optional[str] = None
    device: Optional[str] = None
    compute_type: Optional[str] = None
    beam_size: Optional[int] = None
    word_timestamps: Optional[bool] = False
    
    def __init__(self, **data):
        # 设置默认值
        if 'model_name' not in data or data['model_name'] is None:
            data['model_name'] = config.DEFAULT_MODEL
        if 'device' not in data or data['device'] is None:
            data['device'] = config.DEFAULT_DEVICE
        if 'beam_size' not in data or data['beam_size'] is None:
            data['beam_size'] = config.BEAM_SIZE
        super().__init__(**data)

# ----------------------------
# FastAPI 路由 - 认证
# ----------------------------
@app.post("/api/login")
async def login(request: LoginRequest):
    """用户登录接口"""
    if AuthService.authenticate_user(request.username, request.password):
        access_token = AuthService.create_access_token(data={"user": request.username})
        return {
            "token": access_token,
            "user": request.username,
            "token_type": "bearer"
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )

# ----------------------------
# FastAPI 路由 - 基础API
# ----------------------------
@app.get("/api/model/info")
async def get_model_info(current_user: str = Depends(AuthService.verify_token)):
    """获取模型信息"""
    try:
        info = whisper_service.get_model_info()
        return info
    except Exception as e:
        Logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    beam_size: int = Form(5),
    model_name: str = Form("small"),
    current_user: str = Depends(AuthService.verify_token)
):
    """基础语音转文字接口 - 文件上传方式"""
    start_time = time.time()
    
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    # 检查文件大小
    if audio.size and audio.size > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # 检查文件格式
    if not FileUtils.is_supported_file(audio.filename):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # 保存上传的文件
    file_path = os.path.join(config.UPLOAD_FOLDER, audio.filename)
    
    try:
        Logger.info(f"User {current_user} started transcribing: {audio.filename} ({audio.size} bytes)")
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await audio.read()
            await f.write(content)
        
        # 进行转录
        transcribe_start = time.time()
        # 使用配置默认值
        actual_model_name = model_name or config.DEFAULT_MODEL
        actual_beam_size = beam_size or config.BEAM_SIZE
        result = await whisper_service.transcribe_basic(file_path, actual_beam_size, actual_model_name)
        transcribe_time = time.time() - transcribe_start
        
        total_time = time.time() - start_time
        Logger.info(
            f"Transcription completed - File: {audio.filename}, "
            f"Transcription time: {transcribe_time:.2f}s, "
            f"Total time: {total_time:.2f}s, "
            f"Segments: {len(result.get('segments', []))}"
        )
        
        return result
    
    except Exception as e:
        Logger.error(f"Transcription failed - File: {audio.filename}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)
            Logger.debug(f"Cleaned up temporary file: {file_path}")

@app.post("/api/transcribe/basic")
async def transcribe_basic(
    req: BasicTranscribeRequest = Body(...),
    authorization: Optional[str] = Header(None),
    upload_file: Optional[UploadFile] = File(None)
):
    """
    基础语音转文字接口 - 支持文件上传、URL和本地路径
    - input_type=upload: 需要提供 upload_file
    - input_type=url: 需要提供 url
    - input_type=local: 需要提供 local_path
    """
    # 验证token
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = parts[1]
    if token != config.API_TOKEN and token not in config.API_TOKENS:
        raise HTTPException(status_code=403, detail="Invalid token")

    start_time = time.time()
    file_path = None
    
    try:
        # 根据输入类型处理文件
        if req.input_type == "upload":
            if upload_file is None:
                raise HTTPException(status_code=400, detail="upload_file required for input_type 'upload'")
            
            if not upload_file.filename:
                raise HTTPException(status_code=400, detail="No file selected")
            
            # 检查文件大小
            if upload_file.size and upload_file.size > config.MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
            
            # 检查文件格式
            if not FileUtils.is_supported_file(upload_file.filename):
                raise HTTPException(status_code=400, detail="Unsupported file format")
            
            # 保存上传的文件
            file_path = os.path.join(config.UPLOAD_FOLDER, upload_file.filename)
            async with aiofiles.open(file_path, 'wb') as f:
                content = await upload_file.read()
                await f.write(content)
            
            Logger.info(f"Started transcribing uploaded file: {upload_file.filename}")
            
        elif req.input_type == "url":
            if not req.url:
                raise HTTPException(status_code=400, detail="url required for input_type 'url'")
            
            # 创建临时文件
            job_dir = FileUtils.create_job_dir()
            filename = Path(req.url.split("?")[0].split("/")[-1] or "downloaded_audio")
            file_path = job_dir / filename.name
            
            # 下载文件
            MediaProcessor.download_from_url(req.url, file_path)
            Logger.info(f"Started transcribing URL: {req.url}")
            
        elif req.input_type == "local":
            if not req.local_path:
                raise HTTPException(status_code=400, detail="local_path required for input_type 'local'")
            
            # 检查本地文件是否存在
            local_path = Path(req.local_path).resolve()
            if not local_path.exists():
                raise HTTPException(status_code=404, detail=f"Local file not found: {req.local_path}")
            
            # 检查文件格式
            if not FileUtils.is_supported_file(str(local_path)):
                raise HTTPException(status_code=400, detail="Unsupported file format")
            
            file_path = local_path
            Logger.info(f"Started transcribing local file: {req.local_path}")
            
        else:
            raise HTTPException(status_code=400, detail="input_type must be upload|url|local")
        
        # 进行转录
        transcribe_start = time.time()
        result = await whisper_service.transcribe_basic(
            str(file_path), 
            req.beam_size, 
            req.model_name
        )
        transcribe_time = time.time() - transcribe_start
        
        total_time = time.time() - start_time
        Logger.info(
            f"Basic transcription completed - Input: {req.input_type}, "
            f"Transcription time: {transcribe_time:.2f}s, "
            f"Total time: {total_time:.2f}s, "
            f"Segments: {len(result.get('segments', []))}"
        )
        
        return result
    
    except Exception as e:
        Logger.error(f"Basic transcription failed - Input: {req.input_type}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 清理临时文件（仅清理上传和下载的文件，不删除本地文件）
        if file_path and req.input_type in ["upload", "url"]:
            if os.path.exists(file_path):
                if req.input_type == "upload":
                    os.remove(file_path)
                    Logger.debug(f"Cleaned up temporary file: {file_path}")
                # URL下载的文件保留在job_dir中，供调试使用

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    ffmpeg_available = SystemUtils.check_ffmpeg_available()
    ffmpeg_path = SystemUtils._ffmpeg_path if ffmpeg_available else None
    
    return {
        "status": "healthy", 
        "service": "integrated-whisper-api", 
        "version": "2.0.0",
        "ffmpeg_available": ffmpeg_available,
        "ffmpeg_path": ffmpeg_path,
        "dependencies": {
            "faster_whisper": True,
            "fastapi": True,
            "ffmpeg": ffmpeg_available
        },
        "config": {
            "default_model": config.DEFAULT_MODEL,
            "default_device": config.DEFAULT_DEVICE,
            "max_file_size": config.MAX_FILE_SIZE,
            "supported_formats": config.get_supported_extensions()
        }
    }

# ----------------------------
# FastAPI 路由 - 高级API（兼容原app.py）
# ----------------------------
@app.post("/api/transcribe/advanced-json")
async def api_transcribe_advanced_json(
    req: TranscribeRequest = Body(...), 
    authorization: Optional[str] = Header(None)
):
    """
    高级转录接口，仅支持JSON请求（用于local和url类型）
    """
    # 验证token
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = parts[1]
    if token != config.API_TOKEN and token not in config.API_TOKENS:
        raise HTTPException(status_code=403, detail="Invalid token")

    # 只支持 local 和 url 类型
    if req.input_type not in ["local", "url"]:
        raise HTTPException(status_code=400, detail="This endpoint only supports 'local' and 'url' input types")

    # 创建任务
    job_id = job_manager.create_job(req.input_type, **req.dict())
    job_dir = Path(job_manager.get_job(job_id)["job_dir"])

    # 后台任务处理
    async def _do_job():
        try:
            job_manager.update_job_status(job_id, "running")
            local_input = None
            
            # 准备输入文件
            if req.input_type == "local":
                if not req.local_path:
                    raise RuntimeError("local_path required for input_type 'local'")
                p = Path(req.local_path).resolve()
                if not p.exists():
                    raise RuntimeError(f"local_path not found: {req.local_path}")
                local_input = job_dir / p.name
                shutil.copy2(str(p), str(local_input))
                job_manager.add_file(job_id, "input", str(local_input))
                
            elif req.input_type == "url":
                if not req.url:
                    raise RuntimeError("url required for input_type 'url'")
                filename = Path(req.url.split("?")[0].split("/")[-1] or "downloaded_input")
                local_input = job_dir / filename.name
                MediaProcessor.download_from_url(req.url, local_input)
                job_manager.add_file(job_id, "input", str(local_input))

            # 准备音频文件
            audio_path = job_dir / "audio.wav"
            MediaProcessor.extract_audio(local_input, audio_path)

            # 转录
            segments = await whisper_service.transcribe_advanced(
                audio_path, req.model_name, req.device, req.compute_type, 
                req.beam_size, "transcribe", req.word_timestamps
            )

            # 生成输出文件
            out_basename = req.out_basename or f"output_{job_id}"
            
            # 生成SRT字幕
            srt_path = job_dir / f"{out_basename}.srt"
            SubtitleGenerator.write_srt(segments, srt_path, bilingual=False)
            job_manager.add_file(job_id, "srt", str(srt_path))

            # 生成双语字幕
            if req.bilingual:
                translated_segments = await whisper_service.transcribe_advanced(
                    audio_path, req.model_name, req.device, req.compute_type, 
                    req.beam_size, "translate", req.word_timestamps
                )
                bilingual_srt_path = job_dir / f"{out_basename}_bilingual.srt"
                SubtitleGenerator.write_srt(segments, bilingual_srt_path, bilingual=True, translated_segments=translated_segments)
                job_manager.add_file(job_id, "srt_bilingual", str(bilingual_srt_path))

            # 处理视频输出
            if local_input and FileUtils.is_video_file(str(local_input)):
                if req.burn == "soft":
                    out_video = job_dir / f"{out_basename}_softsub{local_input.suffix}"
                    # TODO: 实现软字幕功能
                    # mux_softsub(local_input, srt_path, out_video)
                    # job_manager.add_file(job_id, "video_softsub", str(out_video))
                    pass
                elif req.burn == "hard":
                    out_video = job_dir / f"{out_basename}_hardsub{local_input.suffix}"
                    # TODO: 实现硬字幕功能
                    # burn_hardsub(local_input, srt_path, out_video)
                    # job_manager.add_file(job_id, "video_hardsub", str(out_video))
                    pass

            # 保存转录结果
            result = {
                "segments": [
                    {"start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in segments
                ],
                "transcript_text": "\n".join([seg.text for seg in segments])
            }
            job_manager.set_result(job_id, result)
            job_manager.update_job_status(job_id, "completed")

        except Exception as e:
            Logger.error(f"Job {job_id} failed: {e}", job_id)
            job_manager.set_error(job_id, str(e))

    # 启动后台任务
    asyncio.create_task(_do_job())

    return {"job_id": job_id, "status": "queued"}

@app.post("/api/transcribe/advanced")
async def api_transcribe_advanced(
    req: TranscribeRequest = Body(...), 
    authorization: Optional[str] = Header(None), 
    upload_file: Optional[UploadFile] = File(None), 
    video_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None)
):
    """
    高级转录接口，支持视频处理和字幕生成（兼容原app.py功能）
    - For input_type == "upload": include multipart file upload_file
    - For input_type == "url": provide url in req.url
    - For input_type == "local": provide server local_path (absolute) -- will be copied
    - For input_type == "separate_audio": provide video_file/audio_file or video_url/audio_url or video_local_path/audio_local_path
    """
    # 验证token
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = parts[1]
    if token != config.API_TOKEN and token not in config.API_TOKENS:
        raise HTTPException(status_code=403, detail="Invalid token")

    job_id = str(uuid.uuid4())
    job_tmp = FileUtils.create_job_dir()
    JOBS[job_id] = {"tmp": str(job_tmp), "status": "queued", "files": {}}

    # Run transcription in background
    async def _do_job():
        try:
            JOBS[job_id]["status"] = "running"
            local_input = None
            audio_input = None
            
            # prepare input
            if req.input_type == "upload":
                if upload_file is None:
                    raise RuntimeError("upload_file required for input_type 'upload'")
                local_input = await ensure_local_file_async(upload_file, tmpdir=job_tmp)
            elif req.input_type == "url":
                if not req.url:
                    raise RuntimeError("url required for input_type 'url'")
                filename = Path(req.url.split("?")[0].split("/")[-1] or "downloaded_input")
                local_input = job_tmp / filename.name
                MediaProcessor.download_from_url(req.url, local_input)
            elif req.input_type == "local":
                if not req.local_path:
                    raise RuntimeError("local_path required for input_type 'local'")
                p = Path(req.local_path).resolve()
                if not p.exists():
                    raise RuntimeError(f"local_path not found: {req.local_path}")
                local_input = job_tmp / p.name
                shutil.copy2(str(p), str(local_input))
            elif req.input_type == "separate_audio":
                # Handle video file
                if video_file is not None:
                    local_input = await ensure_local_file_async(video_file, tmpdir=job_tmp)
                elif req.video_url:
                    filename = Path(req.video_url.split("?")[0].split("/")[-1] or "downloaded_video")
                    local_input = job_tmp / filename.name
                    MediaProcessor.download_from_url(req.video_url, local_input)
                elif req.video_local_path:
                    p = Path(req.video_local_path).resolve()
                    if not p.exists():
                        raise RuntimeError(f"video_local_path not found: {req.video_local_path}")
                    local_input = job_tmp / p.name
                    shutil.copy2(str(p), str(local_input))
                else:
                    raise RuntimeError("video file required for input_type 'separate_audio'")
                
                # Handle audio file
                if audio_file is not None:
                    audio_input = await ensure_local_file_async(audio_file, tmpdir=job_tmp)
                elif req.audio_url:
                    filename = Path(req.audio_url.split("?")[0].split("/")[-1] or "downloaded_audio")
                    audio_input = job_tmp / filename.name
                    MediaProcessor.download_from_url(req.audio_url, audio_input)
                elif req.audio_local_path:
                    p = Path(req.audio_local_path).resolve()
                    if not p.exists():
                        raise RuntimeError(f"audio_local_path not found: {req.audio_local_path}")
                    audio_input = job_tmp / p.name
                    shutil.copy2(str(p), str(audio_input))
                else:
                    raise RuntimeError("audio file required for input_type 'separate_audio'")
                
                JOBS[job_id]["files"]["video_input"] = str(local_input)
                JOBS[job_id]["files"]["audio_input"] = str(audio_input)
            else:
                raise RuntimeError("input_type must be upload|url|local|separate_audio")

            if local_input:
                JOBS[job_id]["files"]["input"] = str(local_input)

            # Prepare audio for transcription
            audio_path = job_tmp / "audio.wav"
            if req.input_type == "separate_audio":
                # Use provided audio file directly
                shutil.copy2(str(audio_input), str(audio_path))
            else:
                # Extract audio from video or use audio file directly
                MediaProcessor.extract_audio(local_input, audio_path)

            # Transcribe
            segments = await whisper_service.transcribe_advanced(
                audio_path, req.model_name, req.device, req.compute_type, 
                req.beam_size, "transcribe", req.word_timestamps
            )

            # Generate outputs
            out_basename = req.out_basename or f"output_{job_id}"
            
            # Always generate SRT
            srt_path = job_tmp / f"{out_basename}.srt"
            SubtitleGenerator.write_srt(segments, srt_path, bilingual=False)
            JOBS[job_id]["files"]["srt"] = str(srt_path)

            # Generate bilingual SRT if requested
            if req.bilingual:
                # Translate segments
                translated_segments = await whisper_service.transcribe_advanced(
                    audio_path, req.model_name, req.device, req.compute_type, 
                    req.beam_size, "translate", req.word_timestamps
                )
                bilingual_srt_path = job_tmp / f"{out_basename}_bilingual.srt"
                SubtitleGenerator.write_srt(segments, bilingual_srt_path, bilingual=True, translated_segments=translated_segments)
                JOBS[job_id]["files"]["srt_bilingual"] = str(bilingual_srt_path)

            # Process video output if requested
            if req.input_type != "separate_audio" and local_input and local_input.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                if req.burn == "soft":
                    # Soft subtitles
                    out_video = job_tmp / f"{out_basename}_softsub{local_input.suffix}"
                    mux_softsub(local_input, srt_path, out_video)
                    JOBS[job_id]["files"]["video_softsub"] = str(out_video)
                elif req.burn == "hard":
                    # Hard subtitles (simplified version)
                    out_video = job_tmp / f"{out_basename}_hardsub{local_input.suffix}"
                    # Note: burn_hardsub function would need to be implemented
                    # burn_hardsub(local_input, srt_path, out_video)
                    JOBS[job_id]["files"]["video_hardsub"] = str(out_video)

            # Store transcription result
            JOBS[job_id]["result"] = {
                "segments": [
                    {"start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in segments
                ]
            }

            JOBS[job_id]["status"] = "completed"

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            logger.error(f"Job {job_id} failed: {e}")

    # Start background job
    asyncio.create_task(_do_job())

    return {"job_id": job_id, "status": "queued"}

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str, authorization: Optional[str] = Header(None)):
    """获取任务状态"""
    # 验证token
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = parts[1]
    if token != config.API_TOKEN and token not in config.API_TOKENS:
        raise HTTPException(status_code=403, detail="Invalid token")

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": job_id,
        "status": job["status"],
        "files": job.get("files", {}),
        "created_at": job["created_at"].isoformat() if job.get("created_at") else None
    }

    if job["status"] == "completed":
        response["result"] = job.get("result")
    elif job["status"] == "failed":
        response["error"] = job.get("error")

    return response

@app.get("/api/download/{job_id}")
async def download_file(job_id: str, file: str = Query(...), authorization: Optional[str] = Header(None)):
    """下载生成的文件"""
    # 验证token
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = parts[1]
    if token != config.API_TOKEN and token not in config.API_TOKENS:
        raise HTTPException(status_code=403, detail="Invalid token")

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    file_path_str = job.get("files", {}).get(file)
    if not file_path_str:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = Path(file_path_str)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(str(file_path), filename=file_path.name)

# ----------------------------
# 辅助函数
# ----------------------------
async def ensure_local_file_async(obj, tmpdir: Optional[Path]=None) -> Path:
    """
    异步版本的 ensure_local_file
    Convert various input types (Gradio NamedString, path string, UploadFile, bytes, io.BytesIO)
    into a local Path (copied into a temp directory).
    """
    Logger.debug(f"ensure_local_file called with object type: {type(obj)}")
    
    if tmpdir is None:
        tmpdir = FileUtils.create_job_dir()
    else:
        tmpdir = Path(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
    
    Logger.debug(f"Using temp directory: {tmpdir}")

    # UploadFile from FastAPI (has .file and .filename)
    if hasattr(obj, "file") and hasattr(obj, "filename"):
        filename = getattr(obj, "filename")
        if not filename:
            filename = "upload.bin"
        dst = tmpdir / Path(filename).name
        async with aiofiles.open(dst, "wb") as f:
            # obj.file might be async object; attempt to read bytes
            try:
                content = await obj.file.read()
            except Exception:
                # Try .readable -> iterate
                await obj.file.seek(0)
                content = await obj.file.read()
            if isinstance(content, str):
                content = content.encode()
            await f.write(content)
        return dst

    # If it's a plain str that points to a file path:
    if isinstance(obj, str):
        p = Path(obj).resolve()  # Get absolute path
        if p.exists():
            dst = tmpdir / p.name
            shutil.copy2(str(p), str(dst))
            return dst
        # If it's a base64 data URL or raw base64, try to handle? (not implemented)
        raise RuntimeError(f"String provided but file not found: {obj}")

    # If it's a Path object
    if isinstance(obj, Path):
        p = obj.resolve()  # Get absolute path
        if p.exists():
            dst = tmpdir / p.name
            shutil.copy2(str(p), str(dst))
            return dst
        raise RuntimeError(f"Path object points to non-existent file: {obj}")

    # If nothing matched
    raise RuntimeError(f"Unsupported uploaded object type: {type(obj)}")

def mux_softsub(input_video: Path, srt_file: Path, out_video: Path):
    """混合软字幕到视频文件"""
    # Ensure paths are absolute
    input_video = Path(input_video).resolve()
    srt_file = Path(srt_file).resolve()
    out_video = Path(out_video).resolve()
    
    # Ensure output directory exists
    out_video.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用找到的 ffmpeg 路径
    ffmpeg_path = globals().get('FFMPEG_PATH', 'ffmpeg')
    
    ext = out_video.suffix.lower()
    if ext in (".mp4", ".mov", ".m4v"):
        cmd = [ffmpeg_path, "-y", "-i", str(input_video), "-i", str(srt_file),
               "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text", str(out_video)]
    else:
        cmd = [ffmpeg_path, "-y", "-i", str(input_video), "-i", str(srt_file),
               "-map", "0", "-map", "1", "-c", "copy", "-c:s", "srt", str(out_video)]
    SystemUtils.run_cmd(cmd)

# ----------------------------
# Gradio 界面 - 整合版
# ----------------------------
def create_gradio_interface():
    """创建整合版 Gradio 界面"""
    
    # 自定义CSS
    custom_css = """
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
        transition: border-color 0.3s;
    }
    .upload-area:hover {
        border-color: #007bff;
    }
    .result-area {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .segment {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #007bff;
    }
    
    /* 字体回退处理 - 避免字体文件404错误影响界面 */
    @font-face {
        font-family: 'ui-monospace';
        src: local('Consolas'), local('Monaco'), local('Courier New'), monospace;
        font-display: swap;
    }
    
    body, pre, code {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 
                     'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', 
                     sans-serif, 'ui-monospace', 'Consolas', 'Monaco', 'Courier New', monospace !important;
    }
    
    /* 隐藏字体加载错误 */
    @font-face {
        font-family: 'ui-monospace';
        src: url('about:blank');
        unicode-range: U+0-10FFFF;
    }
    """
    
    with gr.Blocks(
        css=custom_css, 
        title="整合版 Whisper 语音转文字服务",
        theme=gr.themes.Soft(),
        analytics_enabled=False,
        delete_cache=(1800, 1800)  # 30分钟清理缓存
    ) as demo:
        gr.Markdown("# 🎙️ 整合版 Whisper 语音转文字服务")
        
        # 定义状态变量
        job_completed = gr.State(value=False)
        
        with gr.Tabs():
            # 基础转录标签页
            with gr.TabItem("基础转录"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📤 音频输入")
                        
                        # 输入类型选择
                        basic_input_type = gr.Radio(
                            choices=["upload", "url", "local"],
                            value="upload",
                            label="输入类型",
                            info="选择音频输入方式"
                        )
                        
                        # 上传文件区域
                        with gr.Group(visible=True) as basic_upload_group:
                            audio_input = gr.Audio(
                                label="上传音频文件",
                                type="filepath"
                            )
                        
                        # URL输入区域
                        with gr.Group(visible=False) as basic_url_group:
                            basic_url_input = gr.Textbox(
                                label="音频文件URL",
                                placeholder="输入音频文件的URL地址 (支持http/https)",
                                info="支持常见的音频格式：mp3, wav, m4a, aac, ogg, flac"
                            )
                        
                        # 本地路径区域
                        with gr.Group(visible=False) as basic_local_group:
                            basic_local_input = gr.Textbox(
                                label="本地文件路径",
                                placeholder="输入本地音频文件的完整路径",
                                info="例如: C:/Users/user/music/audio.mp3 或 /home/user/audio.wav"
                            )
                        
                        with gr.Row():
                            basic_model_choice = gr.Dropdown(
                                choices=["tiny", "base", "small", "medium", "large"],
                                value="small",
                                label="模型选择",
                                info="模型越大准确度越高，但速度越慢"
                            )
                            basic_beam_size = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Beam Size",
                                info="提高搜索质量，但会增加处理时间"
                            )
                        
                        transcribe_btn = gr.Button("开始转录", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### 📝 转录结果")
                        basic_result_text = gr.Textbox(
                            label="转录文本",
                            lines=10,
                            max_lines=20,
                            placeholder="转录结果将显示在这里..."
                        )
                        
                        basic_result_info = gr.JSON(label="详细信息", visible=False)
                        basic_download_srt = gr.File(label="下载SRT文件", visible=False)
                        
                        with gr.Accordion("高级信息", open=False):
                            basic_language_info = gr.Textbox(
                                label="检测语言",
                                interactive=False
                            )
                            basic_segments_count = gr.Number(
                                label="片段数量",
                                interactive=False
                            )
                            basic_processing_time = gr.Textbox(
                                label="处理时间",
                                interactive=False
                            )
            
            # 高级字幕生成标签页
            with gr.TabItem("高级字幕生成"):
                # 轮询状态控制
                polling_active = gr.State(value=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📤 上传文件")
                        input_type = gr.Radio(
                            choices=["upload", "url", "local", "separate_audio"],
                            value="upload",
                            label="输入类型"
                        )
                        
                        with gr.Group(visible=True) as upload_group:
                            video_input = gr.Video(label="上传视频文件")
                            audio_input_adv = gr.Audio(label="上传音频文件")
                        
                        with gr.Group(visible=False) as url_group:
                            gr.Markdown("#### 📹 视频文件")
                            video_url_input = gr.Textbox(label="视频文件URL", placeholder="输入视频文件的URL地址")
                            gr.Markdown("#### 🎵 音频文件")
                            audio_url_input = gr.Textbox(label="音频文件URL", placeholder="输入音频文件的URL地址")
                            gr.Markdown("*提示：可以同时提供视频和音频文件，或只提供其中一个*")
                        
                        with gr.Group(visible=False) as local_group:
                            gr.Markdown("#### 📹 视频文件")
                            video_local_path_input = gr.Textbox(label="视频文件本地路径", placeholder="输入视频文件的完整路径")
                            gr.Markdown("#### 🎵 音频文件")
                            audio_local_path_input = gr.Textbox(label="音频文件本地路径", placeholder="输入音频文件的完整路径")
                            gr.Markdown("*提示：可以同时提供视频和音频文件，或只提供其中一个*")
                        
                        with gr.Group(visible=False) as separate_audio_group:
                            gr.Markdown("#### 📹 视频文件")
                            video_separate_input = gr.Video(label="上传视频文件")
                            video_separate_url_input = gr.Textbox(label="视频文件URL", placeholder="输入视频文件的URL地址")
                            video_separate_local_input = gr.Textbox(label="视频文件本地路径", placeholder="输入视频文件的完整路径")
                            
                            gr.Markdown("#### 🎵 音频文件")
                            audio_separate_input = gr.Audio(label="上传音频文件")
                            audio_separate_url_input = gr.Textbox(label="音频文件URL", placeholder="输入音频文件的URL地址")
                            audio_separate_local_input = gr.Textbox(label="音频文件本地路径", placeholder="输入音频文件的完整路径")
                            
                            gr.Markdown("*提示：separate_audio模式需要同时提供视频和音频文件，支持三种输入方式*")
                        
                        gr.Markdown("### ⚙️ 高级选项")
                        
                        with gr.Row():
                            model_choice_adv = gr.Dropdown(
                                choices=["tiny", "base", "small", "medium", "large"],
                                value="small",
                                label="模型选择"
                            )
                            device_choice = gr.Dropdown(
                                choices=["cpu", "cuda"],
                                value="cpu",
                                label="设备选择"
                            )
                        
                        with gr.Row():
                            bilingual = gr.Checkbox(label="双语字幕", value=True)
                            word_timestamps = gr.Checkbox(label="词级时间戳", value=False)
                        
                        with gr.Row():
                            burn_type = gr.Radio(
                                choices=["none", "hard", "soft"],
                                value="none",
                                label="字幕类型"
                            )
                            beam_size_adv = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Beam Size"
                            )
                        
                        transcribe_adv_btn = gr.Button("开始高级转录", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### 📝 转录结果")
                        job_id_display = gr.Textbox(label="任务ID", interactive=False)
                        
                        # 简化状态显示
                        status_info = gr.HTML("<div>等待提交任务...</div>")
                        
                        result_status = gr.JSON(label="详细状态", visible=False)
                        
                        # 转录文本显示
                        transcript_display = gr.Textbox(
                            label="转录文本", 
                            lines=10, 
                            interactive=False,
                            visible=False
                        )
                        
                        # 下载文件组件
                        srt_download = gr.File(label="下载SRT字幕文件", visible=False)
                        bilingual_srt_download = gr.File(label="下载双语SRT字幕文件", visible=False)
                        video_download = gr.File(label="下载字幕视频文件", visible=False)
            
            # API文档标签页
            with gr.TabItem("API文档"):
                gr.Markdown("""
                ## 📚 API 接口文档
                
                ### 认证
                所有API请求都需要在Header中包含Authorization token：
                ```
                Authorization: Bearer <token>
                ```
                
                可用tokens：
                - `whisper-api-key-2024` - 自动化调用
                - `test-token` - 测试用途
                
                ### 基础接口
                
                #### 1. 用户登录
                ```
                POST /api/login
                Content-Type: application/json
                
                {
                    "username": "admin",
                    "password": "admin123"
                }
                ```
                
                #### 2. 获取模型信息
                ```
                GET /api/model/info
                Authorization: Bearer <token>
                ```
                
                #### 3. 基础转录
                ```
                POST /api/transcribe
                Authorization: Bearer <token>
                Content-Type: multipart/form-data
                
                audio: <音频文件>
                beam_size: 5 (可选)
                model_name: small (可选)
                ```
                
                #### 4. 高级转录
                ```
                POST /api/transcribe/advanced
                Authorization: Bearer <token>
                Content-Type: application/json
                
                {
                    "input_type": "upload",
                    "model_name": "small",
                    "bilingual": true,
                    "burn": "none"
                }
                ```
                
                #### 5. 查询任务状态
                ```
                GET /api/job/{job_id}
                Authorization: Bearer <token>
                ```
                
                #### 6. 下载文件
                ```
                GET /api/download/{job_id}?file=srt
                Authorization: Bearer <token>
                ```
                
                #### 7. 健康检查
                ```
                GET /api/health
                ```
                """)
        
        # 事件处理
        def update_input_visibility(input_type):
            return (
                gr.update(visible=(input_type == "upload")),
                gr.update(visible=(input_type == "url")),
                gr.update(visible=(input_type == "local")),
                gr.update(visible=(input_type == "separate_audio"))
            )
        
        input_type.change(
            update_input_visibility,
            inputs=[input_type],
            outputs=[upload_group, url_group, local_group, separate_audio_group]
        )
        
        # 输入类型切换处理
        def update_basic_input_visibility(input_type):
            """根据输入类型显示/隐藏相应的输入组件"""
            if input_type == "upload":
                return (
                    gr.update(visible=True),   # basic_upload_group
                    gr.update(visible=False),  # basic_url_group
                    gr.update(visible=False)    # basic_local_group
                )
            elif input_type == "url":
                return (
                    gr.update(visible=False),  # basic_upload_group
                    gr.update(visible=True),   # basic_url_group
                    gr.update(visible=False)    # basic_local_group
                )
            elif input_type == "local":
                return (
                    gr.update(visible=False),  # basic_upload_group
                    gr.update(visible=False),  # basic_url_group
                    gr.update(visible=True)    # basic_local_group
                )
        
        basic_input_type.change(
            update_basic_input_visibility,
            inputs=[basic_input_type],
            outputs=[basic_upload_group, basic_url_group, basic_local_group]
        )
        
        # 基础转录处理
        async def transcribe_basic_handler(input_type, audio_file, url_input, local_input, model_name, beam_size):
            """扩展的基础转录处理函数 - 直接调用本地方法"""
            start_time = time.time()
            audio_path = None
            
            try:
                # 根据输入类型处理文件
                if input_type == "upload":
                    if not audio_file:
                        return "请先选择音频文件", {}, None, "", "", ""
                    
                    # 检查文件格式
                    if not FileUtils.is_supported_file(audio_file):
                        return "不支持的文件格式", {}, None, "", "", ""
                    
                    # 使用上传的文件路径
                    audio_path = audio_file
                    Logger.info(f"Processing uploaded file: {audio_file}")
                    
                elif input_type == "url":
                    if not url_input:
                        return "请输入音频文件URL", {}, None, "", "", ""
                    
                    # 创建临时文件并下载
                    job_dir = FileUtils.create_job_dir()
                    filename = Path(url_input.split("?")[0].split("/")[-1] or "downloaded_audio")
                    audio_path = job_dir / filename.name
                    
                    try:
                        MediaProcessor.download_from_file(url_input, audio_path)
                        Logger.info(f"Downloaded URL to: {audio_path}")
                    except Exception as e:
                        return f"下载失败: {str(e)}", {}, None, "", "", ""
                    
                elif input_type == "local":
                    if not local_input:
                        return "请输入本地文件路径", {}, None, "", "", ""
                    
                    # 检查本地文件
                    local_path = Path(local_input).resolve()
                    if not local_path.exists():
                        return f"本地文件不存在: {local_input}", {}, None, "", "", ""
                    
                    if not FileUtils.is_supported_file(str(local_path)):
                        return "不支持的文件格式", {}, None, "", "", ""
                    
                    audio_path = str(local_path)
                    Logger.info(f"Processing local file: {local_input}")
                
                else:
                    return "不支持的输入类型", {}, None, "", "", ""
                
                # 直接调用本地 Whisper 服务
                Logger.info(f"Starting transcription for: {audio_path}")
                result = await whisper_service.transcribe_basic(audio_path, beam_size, model_name)
                
                # 格式化结果文本
                full_text = "\n".join([seg["text"] for seg in result.get("segments", [])])
                
                # 提取详细信息
                language = result.get("language", "unknown")
                language_prob = result.get("language_probability", 0)
                segments_count = len(result.get("segments", []))
                processing_time = time.time() - start_time
                
                # 生成SRT文件（可选）
                srt_file_path = None
                if result.get("segments"):
                    # 创建模拟的段对象
                    class Segment:
                        def __init__(self, start, end, text):
                            self.start = start
                            self.end = end
                            self.text = text
                    
                    segments = []
                    for seg in result["segments"]:
                        segments.append(Segment(seg['start'], seg['end'], seg['text']))
                    
                    output_dir = FileUtils.get_output_dir()
                    srt_path = output_dir / f"basic_transcript_{FileUtils.generate_job_id()}.srt"
                    SubtitleGenerator.write_srt(segments, srt_path)
                    srt_file_path = str(srt_path)
                
                # 清理临时文件（仅清理下载的文件）
                if input_type == "url" and os.path.exists(audio_path):
                    # 保留下载的文件用于调试，可以选择删除
                    # os.remove(audio_path)
                    pass
                
                return (
                    full_text,
                    result,
                    srt_file_path,
                    f"检测语言: {language} (置信度: {language_prob:.2f})",
                    segments_count,
                    f"{processing_time:.2f}秒"
                )
                
            except Exception as e:
                Logger.error(f"Basic transcription error: {e}")
                return f"转录失败: {str(e)}", {"error": str(e)}, None, "", "", ""
        
        transcribe_btn.click(
            transcribe_basic_handler,
            inputs=[
                basic_input_type, audio_input, basic_url_input, basic_local_input,
                basic_model_choice, basic_beam_size
            ],
            outputs=[
                basic_result_text, basic_result_info, basic_download_srt,
                basic_language_info, basic_segments_count, basic_processing_time
            ]
        )
        
        # 高级转录处理
        async def transcribe_advanced_handler(*args):
            """
            高级转录处理函数 - 重构版，直接调用本地方法，无需HTTP请求
            """
            try:
                # 解包参数
                (input_type, video_file, audio_file, video_url, audio_url, 
                 video_local_path, audio_local_path, model_name, device, 
                 bilingual, word_timestamps, burn_type, beam_size) = args
                
                # 智能检测：如果用户同时提供了视频和音频文件，自动切换到separate_audio模式
                has_video = bool(video_file or video_url or video_local_path)
                has_audio = bool(audio_file or audio_url or audio_local_path)
                
                # 处理upload模式的智能检测
                if input_type == "upload" and has_video and has_audio:
                    Logger.info("检测到upload模式下同时提供视频和音频文件，切换到separate_audio模式")
                    input_type = "separate_audio"
                elif input_type != "separate_audio" and has_video and has_audio:
                    Logger.info("检测到同时提供视频和音频文件，自动切换到separate_audio模式")
                    input_type = "separate_audio"
                
                # 创建任务
                job_id = job_manager.create_job(input_type, **{
                    "model_name": model_name,
                    "device": device,
                    "bilingual": bilingual,
                    "word_timestamps": word_timestamps,
                    "burn": burn_type,
                    "beam_size": beam_size
                })
                
                job_dir = Path(job_manager.get_job(job_id)["job_dir"])
                
                # 直接处理文件，无需HTTP请求
                local_input = None
                audio_input = None
                
                try:
                    if input_type == "upload":
                        # 处理上传文件
                        if video_file:
                            local_input = job_dir / Path(video_file).name
                            shutil.copy2(video_file, local_input)
                            job_manager.add_file(job_id, "video_input", str(local_input))
                        elif audio_file:
                            local_input = job_dir / Path(audio_file).name
                            shutil.copy2(audio_file, local_input)
                            job_manager.add_file(job_id, "audio_input", str(local_input))
                            
                    elif input_type == "url":
                        # 处理URL下载
                        if video_url and not audio_url:
                            filename = Path(video_url.split("?")[0].split("/")[-1] or "downloaded_video")
                            local_input = job_dir / filename.name
                            MediaProcessor.download_from_url(video_url, local_input)
                            job_manager.add_file(job_id, "video_input", str(local_input))
                        elif audio_url and not video_url:
                            filename = Path(audio_url.split("?")[0].split("/")[-1] or "downloaded_audio")
                            local_input = job_dir / filename.name
                            MediaProcessor.download_from_url(audio_url, local_input)
                            job_manager.add_file(job_id, "audio_input", str(local_input))
                            
                    elif input_type == "local":
                        # 处理本地文件
                        if video_local_path and not audio_local_path:
                            p = Path(video_local_path).resolve()
                            if p.exists():
                                local_input = job_dir / p.name
                                shutil.copy2(str(p), str(local_input))
                                job_manager.add_file(job_id, "video_input", str(local_input))
                        elif audio_local_path and not video_local_path:
                            p = Path(audio_local_path).resolve()
                            if p.exists():
                                local_input = job_dir / p.name
                                shutil.copy2(str(p), str(local_input))
                                job_manager.add_file(job_id, "audio_input", str(local_input))
                                
                    elif input_type == "separate_audio":
                        # 处理分离音视频模式
                        video_file_path = None
                        audio_file_path = None
                        
                        # 处理视频文件
                        if video_file:
                            video_file_path = job_dir / Path(video_file).name
                            shutil.copy2(video_file, video_file_path)
                            job_manager.add_file(job_id, "video_input", str(video_file_path))
                        elif video_url:
                            filename = Path(video_url.split("?")[0].split("/")[-1] or "downloaded_video")
                            video_file_path = job_dir / filename.name
                            MediaProcessor.download_from_url(video_url, video_file_path)
                            job_manager.add_file(job_id, "video_input", str(video_file_path))
                        elif video_local_path:
                            p = Path(video_local_path).resolve()
                            if p.exists():
                                video_file_path = job_dir / p.name
                                shutil.copy2(str(p), str(video_file_path))
                                job_manager.add_file(job_id, "video_input", str(video_file_path))
                        
                        # 处理音频文件
                        if audio_file:
                            audio_file_path = job_dir / Path(audio_file).name
                            shutil.copy2(audio_file, audio_file_path)
                            job_manager.add_file(job_id, "audio_input", str(audio_file_path))
                        elif audio_url:
                            filename = Path(audio_url.split("?")[0].split("/")[-1] or "downloaded_audio")
                            audio_file_path = job_dir / filename.name
                            MediaProcessor.download_from_url(audio_url, audio_file_path)
                            job_manager.add_file(job_id, "audio_input", str(audio_file_path))
                        elif audio_local_path:
                            p = Path(audio_local_path).resolve()
                            if p.exists():
                                audio_file_path = job_dir / p.name
                                shutil.copy2(str(p), str(audio_file_path))
                                job_manager.add_file(job_id, "audio_input", str(audio_file_path))
                        
                        # 设置主输入文件为视频文件（用于输出）
                        if video_file_path:
                            local_input = video_file_path
                        if audio_file_path:
                            audio_input = audio_file_path
                    
                    # 准备音频文件用于转录 - 使用与基础转录相同的处理方式
                    if audio_input:
                        # 直接使用上传的音频文件，让faster-whisper处理格式
                        audio_path = audio_input
                        Logger.info(f"使用上传的音频文件: {audio_path}")
                    elif local_input:
                        # 直接使用本地文件，让faster-whisper处理格式
                        audio_path = local_input
                        Logger.info(f"使用本地文件: {audio_path}")
                    else:
                        raise Exception("无法找到有效的输入文件")
                    
                    # 更新任务状态为运行中
                    job_manager.update_job_status(job_id, "running")
                    
                    # 转录音频
                    segments = await whisper_service.transcribe_advanced(
                        str(audio_path), model_name, device, None, beam_size, "transcribe", word_timestamps
                    )
                    
                    # 调试：检查转录结果
                    Logger.info(f"转录完成，获得 {len(segments)} 个片段")
                    if segments:
                        Logger.info(f"第一个片段内容: {segments[0].text}")
                    else:
                        Logger.warning("转录结果为空！")
                    
                    # 生成输出文件
                    out_basename = f"output_{job_id}"
                    
                    # 生成SRT字幕
                    srt_path = job_dir / f"{out_basename}.srt"
                    SubtitleGenerator.write_srt(segments, srt_path, bilingual=False)
                    job_manager.add_file(job_id, "srt", str(srt_path))
                    
                    # 生成双语字幕
                    if bilingual:
                        translated_segments = await whisper_service.transcribe_advanced(
                            audio_path, model_name, device, None, beam_size, "translate", word_timestamps
                        )
                        bilingual_srt_path = job_dir / f"{out_basename}_bilingual.srt"
                        SubtitleGenerator.write_srt(segments, bilingual_srt_path, bilingual=True, translated_segments=translated_segments)
                        job_manager.add_file(job_id, "srt_bilingual", str(bilingual_srt_path))
                    
                    # 处理视频输出（如果有视频输入）
                    video_output = None
                    base_video = local_input  # 基础视频文件
                    
                    # 处理分离音视频的场景：如果有独立的音频文件，需要将其合并到视频中
                    if audio_input and local_input:
                        # 合并音视频
                        merged_video_path = job_dir / f"{out_basename}_merged{local_input.suffix}"
                        try:
                            MediaProcessor.merge_audio_video(local_input, audio_input, merged_video_path)
                            base_video = merged_video_path
                            job_manager.add_file(job_id, "video_merged", str(merged_video_path))
                            Logger.info(f"音视频合并成功: {merged_video_path}")
                        except Exception as e:
                            Logger.error(f"音视频合并失败: {e}")
                            base_video = local_input  # 合并失败，使用原视频
                    
                    # 如果有视频文件，生成字幕视频
                    if base_video and FileUtils.is_video_file(str(base_video)):
                        # 获取视频和音频时长信息（仅用于日志）
                        video_duration = MediaProcessor.get_media_duration(base_video)
                        audio_duration = max(seg.end for seg in segments) if segments else 0.0
                        Logger.info(f"视频时长: {video_duration:.2f}秒, 音频时长: {audio_duration:.2f}秒")
                        
                        # 选择要使用的SRT文件
                        if bilingual:
                            srt_to_use = bilingual_srt_path
                        else:
                            srt_to_use = srt_path
                        
                        # 根据用户选择生成字幕视频
                        if burn_type == "soft":
                            video_output = job_dir / f"{out_basename}_softsub{base_video.suffix}"
                            try:
                                MediaProcessor.mux_softsub(base_video, srt_to_use, video_output)
                                job_manager.add_file(job_id, "video_softsub", str(video_output))
                                Logger.info(f"软字幕视频生成成功: {video_output}")
                            except Exception as e:
                                Logger.error(f"软字幕视频生成失败: {e}")
                                video_output = None
                                
                        elif burn_type == "hard":
                            video_output = job_dir / f"{out_basename}_hardsub{base_video.suffix}"
                            try:
                                MediaProcessor.burn_hardsub(base_video, srt_to_use, video_output)
                                job_manager.add_file(job_id, "video_hardsub", str(video_output))
                                Logger.info(f"硬字幕视频生成成功: {video_output}")
                            except Exception as e:
                                Logger.error(f"硬字幕视频生成失败: {e}")
                                video_output = None
                    
                    # 保存转录结果
                    result = {
                        "segments": [
                            {"start": seg.start, "end": seg.end, "text": seg.text}
                            for seg in segments
                        ],
                        "transcript_text": "\n".join([seg.text for seg in segments])
                    }
                    job_manager.set_result(job_id, result)
                    job_manager.update_job_status(job_id, "completed")
                    
                    # 准备最终结果
                    final_job_data = job_manager.get_job(job_id)
                    transcript_text = result.get("transcript_text", "")
                    
                    # 准备文件路径而不是URL
                    srt_path = str(srt_path) if 'srt_path' in locals() else None
                    bilingual_srt_path = str(bilingual_srt_path) if 'bilingual_srt_path' in locals() and bilingual else None
                    video_path = str(video_output) if video_output and video_output.exists() else None
                    
                    # 返回最终完成的结果
                    status_html = '<div style="color: #28a745;">✅ 转录完成！</div>'
                    Logger.info(f"高级转录任务完成: {job_id}")
                    
                    return job_id, status_html, final_job_data, transcript_text, True, srt_path, bilingual_srt_path, video_path
                    
                except Exception as processing_error:
                    Logger.error(f"高级转录处理错误: {processing_error}")
                    job_manager.set_error(job_id, str(processing_error))
                    status_html = f'<div style="color: #dc3545;">❌ 处理失败: {str(processing_error)}</div>'
                    return None, status_html, {"error": str(processing_error)}, "", True, None, None, None
                    
            except Exception as e:
                Logger.error(f"高级转录处理异常: {e}")
                status_html = f'<div style="color: #dc3545;">❌ 提交失败: {str(e)}</div>'
                return None, status_html, {"error": str(e)}, "", True, None, None, None
        
        def check_job_status(job_id):
            if not job_id:
                return {}, None, None, None
            
            try:
                # 尝试多种连接方式，确保至少一种能工作
                urls_to_try = config.get_api_urls(f"/api/job/{job_id}")
                
                response = None
                successful_url = None
                
                for url in urls_to_try:
                    try:
                        print(f"尝试连接URL: {url}")  # 调试信息
                        response = requests.get(
                            url,
                            headers={"Authorization": f"Bearer {AppConfig.API_TOKEN}"},
                            timeout=3,  # 更短的超时
                            verify=False
                        )
                        if response.status_code == 200:
                            successful_url = url
                            print(f"成功连接到: {successful_url}")  # 调试信息
                            break
                    except Exception as e:
                        print(f"连接 {url} 失败: {str(e)}")  # 调试信息
                        continue
                
                if not response or response.status_code != 200:
                    raise Exception(f"所有连接尝试都失败了，最后状态码: {response.status_code if response else 'None'}")
                
                print(f"API响应状态码: {response.status_code}")  # 调试信息
                
                if response.status_code == 200:
                    job_data = response.json()
                    status = job_data.get("status", "unknown")
                    
                    # 准备下载链接
                    srt_url = None
                    bilingual_srt_url = None
                    video_url = None
                    
                    if job_data.get("status") == "completed":
                        files = job_data.get("files", {})
                        # 生成多个下载链接，让浏览器选择可用的
                        download_hosts = ["127.0.0.1", "localhost"]
                        
                        if "srt" in files:
                            srt_url = f"http://{download_hosts[0]}:{AppConfig.PORT}/api/download/{job_id}?file=srt"
                        if "srt_bilingual" in files:
                            bilingual_srt_url = f"http://{download_hosts[0]}:{AppConfig.PORT}/api/download/{job_id}?file=srt_bilingual"
                        if "video_hardsub" in files:
                            video_url = f"http://{download_hosts[0]}:{AppConfig.PORT}/api/download/{job_id}?file=video_hardsub"
                        elif "video_softsub" in files:
                            video_url = f"http://{download_hosts[0]}:{AppConfig.PORT}/api/download/{job_id}?file=video_softsub"
                        
                        # 添加转录结果文本
                        if "result" in job_data:
                            segments = job_data["result"].get("segments", [])
                            transcript_text = "\n".join([seg["text"] for seg in segments])
                            job_data["transcript_text"] = transcript_text
                    
                    elif status == "failed":
                        # 添加错误信息
                        error_msg = job_data.get("error", "未知错误")
                        job_data["error_message"] = f"任务失败: {error_msg}"
                    
                    return job_data, srt_url, bilingual_srt_url, video_url
                else:
                    return {"error": f"API调用失败 (状态码: {response.status_code})", "details": response.text}, None, None, None
                    
            except Exception as e:
                return {"error": "连接失败", "details": str(e)}, None, None, None
        
        # 高级转录事件
        def submit_and_start_polling(*args):
            """直接执行高级转录并返回最终结果"""
            import asyncio
            import threading
            import queue
            
            try:
                # 使用线程和队列来处理异步调用
                result_queue = queue.Queue()
                
                def run_async_in_thread():
                    """在新线程中运行异步函数"""
                    try:
                        # 创建新的事件循环
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        
                        # 运行异步函数
                        result = new_loop.run_until_complete(transcribe_advanced_handler(*args))
                        result_queue.put(("success", result))
                    except Exception as e:
                        result_queue.put(("error", e))
                    finally:
                        new_loop.close()
                
                # 启动线程
                thread = threading.Thread(target=run_async_in_thread)
                thread.start()
                thread.join()  # 等待线程完成
                
                # 获取结果
                status, result = result_queue.get()
                
                if status == "error":
                    raise result
                
                # 解包结果
                job_id, status_html, job_data, transcript_text, completed, srt_path, bilingual_srt_path, video_path = result
                
                # 直接返回最终结果，无需轮询
                return (job_id, status_html, job_data, 
                       transcript_text if transcript_text else "", completed, False,
                       srt_path, bilingual_srt_path, video_path)
                
            except Exception as e:
                Logger.error(f"高级转录执行失败: {e}")
                error_html = f'<div style="color: #dc3545;">❌ 转录失败: {str(e)}</div>'
                return (None, error_html, {"error": str(e)}, "", True, False, None, None, None)
        
        transcribe_adv_btn.click(
            submit_and_start_polling,
            inputs=[input_type, video_input, audio_input_adv, video_url_input, audio_url_input,
                   video_local_path_input, audio_local_path_input, model_choice_adv, device_choice, 
                   bilingual, word_timestamps, burn_type, beam_size_adv],
            outputs=[job_id_display, status_info, result_status, 
                    transcript_display, job_completed, polling_active,
                    srt_download, bilingual_srt_download, video_download]  # 添加下载文件输出
        )
        
        # 轮询机制已移除，现在直接返回最终结果
        
        def update_job_display(job_id):
            """更新任务显示的包装函数"""
            print(f"=== update_job_display 被调用，job_id: {job_id} ===")  # 调试信息
            
            if not job_id:
                print("job_id为空，返回默认值")
                return "<div>等待提交任务...</div>", {}, "", None, None, None, gr.update(visible=False)
            
            print(f"开始轮询任务状态: {job_id}")  # 调试信息
            job_data, srt_url, bilingual_srt_url, video_url = check_job_status(job_id)
            print(f"任务状态响应: {job_data.get('status', 'unknown')}")  # 调试信息
            print(f"完整响应数据: {job_data}")  # 调试信息
            
            # 提取状态信息
            status = job_data.get("status", "unknown")
            
            # 状态显示HTML
            status_html = ""
            download_links_html = ""
            
            if status == "queued":
                status_html = '<div style="color: #6c757d;">⏳ 任务排队中，请稍候...</div>'
            elif status == "running":
                status_html = '<div style="color: #007bff;">🔄 正在处理音频，请耐心等待...</div>'
            elif status == "completed":
                status_html = '<div style="color: #28a745;">✅ 任务完成！</div>'
                # 生成下载链接HTML而不是返回URL对象
                links = []
                if srt_url:
                    links.append(f'<a href="{srt_url}" download>📄 SRT字幕</a>')
                if bilingual_srt_url:
                    links.append(f'<a href="{bilingual_srt_url}" download>📄 双语SRT</a>')
                if video_url:
                    links.append(f'<a href="{video_url}" download>🎬 处理后的视频</a>')
                if links:
                    download_links_html = '<div style="margin-top: 10px;">📥 下载文件: ' + ' | '.join(links) + '</div>'
            elif status == "failed":
                error_msg = job_data.get("error_message", "处理失败")
                status_html = f'<div style="color: #dc3545;">❌ {error_msg}</div>'
            else:
                status_html = f'<div style="color: #6c757d;">📊 状态: {status}</div>'
            
            # 转录文本
            transcript_text = job_data.get("transcript_text", "")
            show_transcript = bool(transcript_text)
            
            # 将下载链接添加到状态HTML中
            if download_links_html:
                status_html += download_links_html
            
            print(f"返回状态HTML: {status_html[:50]}..., 显示转录: {show_transcript}")  # 调试信息
            
            # 返回状态HTML、job_data、转录文本，但不返回URL对象（避免Gradio尝试下载）
            return (status_html, job_data, transcript_text, 
                   None, None, None,  # 不返回URL对象
                   gr.update(visible=show_transcript))
        
        # 当job_id改变时立即更新一次
        def job_id_change_handler(job_id):
            print(f"=== job_id_display.change 被触发，job_id: {job_id} ===")
            if not job_id:
                print("job_id为空，停止轮询")
                return "<div>等待提交任务...</div>", {}, "", gr.update(visible=False), False
            
            try:
                # 立即检查一次状态，但不启动轮询
                job_data, srt_url, bilingual_srt_url, video_url = check_job_status(job_id)
                status = job_data.get("status", "unknown")
                
                # 状态显示HTML
                status_html = ""
                download_links_html = ""
                
                if status == "queued":
                    status_html = '<div style="color: #6c757d;">⏳ 任务排队中，请稍候...</div>'
                elif status == "running":
                    status_html = '<div style="color: #007bff;">🔄 正在处理音频，请耐心等待...</div>'
                elif status == "completed":
                    status_html = '<div style="color: #28a745;">✅ 任务完成！</div>'
                    links = []
                    if srt_url:
                        links.append(f'<a href="{srt_url}" download>📄 SRT字幕</a>')
                    if bilingual_srt_url:
                        links.append(f'<a href="{bilingual_srt_url}" download>📄 双语SRT</a>')
                    if video_url:
                        links.append(f'<a href="{video_url}" download>🎬 处理后的视频</a>')
                    if links:
                        download_links_html = '<div style="margin-top: 10px;">📥 下载文件: ' + ' | '.join(links) + '</div>'
                elif status == "failed":
                    error_msg = job_data.get("error_message", job_data.get("error", "处理失败"))
                    status_html = f'<div style="color: #dc3545;">❌ {error_msg}</div>'
                else:
                    status_html = f'<div style="color: #6c757d;">📊 状态: {status}</div>'
                
                transcript_text = job_data.get("transcript_text", "")
                show_transcript = bool(transcript_text)
                
                if download_links_html:
                    status_html += download_links_html
                
                return (status_html, job_data, transcript_text, 
                       gr.update(visible=show_transcript), False)
                
            except Exception as e:
                print(f"job_id_change_handler 错误: {e}")
                error_status_html = f'<div style="color: #dc3545;">❌ 状态检查失败: {str(e)}</div>'
                return (error_status_html, {}, "", gr.update(visible=False), False)
        
        job_id_display.change(
            job_id_change_handler,
            inputs=[job_id_display],
            outputs=[status_info, result_status, transcript_display, transcript_display, polling_active]
        )
        
        # 自动轮询任务状态 - 智能轮询，完成后停止
        
        def timer_tick_handler(job_id, is_polling_active):
            """定时器处理函数 - 直接处理轮询逻辑"""
            print(f"=== job_timer.tick 被触发，job_id: {job_id}, is_polling_active: {is_polling_active} ===")
            
            # 如果轮询不活跃或没有job_id，返回空值
            if not is_polling_active or not job_id:
                print("轮询不活跃或无job_id，跳过处理")
                return gr.skip()
            
            print(f"开始轮询任务状态: {job_id}")
            try:
                job_data, srt_url, bilingual_srt_url, video_url = check_job_status(job_id)
                status = job_data.get("status", "unknown")
                
                # 检查任务是否完成
                is_completed = status in ["completed", "failed"]
                print(f"任务状态: {status}, 是否完成: {is_completed}")
                
                # 状态显示HTML
                status_html = ""
                download_links_html = ""
                
                if status == "queued":
                    status_html = '<div style="color: #6c757d;">⏳ 任务排队中，请稍候...</div>'
                elif status == "running":
                    status_html = '<div style="color: #007bff;">🔄 正在处理音频，请耐心等待...</div>'
                elif status == "completed":
                    status_html = '<div style="color: #28a745;">✅ 任务完成！</div>'
                    # 生成下载链接HTML
                    links = []
                    if srt_url:
                        links.append(f'<a href="{srt_url}" download>📄 SRT字幕</a>')
                    if bilingual_srt_url:
                        links.append(f'<a href="{bilingual_srt_url}" download>📄 双语SRT</a>')
                    if video_url:
                        links.append(f'<a href="{video_url}" download>🎬 处理后的视频</a>')
                    if links:
                        download_links_html = '<div style="margin-top: 10px;">📥 下载文件: ' + ' | '.join(links) + '</div>'
                elif status == "failed":
                    error_msg = job_data.get("error_message", job_data.get("error", "处理失败"))
                    status_html = f'<div style="color: #dc3545;">❌ {error_msg}</div>'
                else:
                    status_html = f'<div style="color: #6c757d;">📊 状态: {status}</div>'
                
                # 转录文本
                transcript_text = job_data.get("transcript_text", "")
                show_transcript = bool(transcript_text)
                
                # 将下载链接添加到状态HTML中
                if download_links_html:
                    status_html += download_links_html
                
                print(f"返回状态HTML: {status_html[:50]}..., 显示转录: {show_transcript}")
                
                # 返回结果和更新后的轮询状态
                return (status_html, job_data, transcript_text, 
                       gr.update(visible=show_transcript), 
                       not is_completed)  # 如果任务完成，停止轮询
                
            except Exception as e:
                print(f"轮询过程中发生错误: {e}")
                error_status_html = f'<div style="color: #dc3545;">❌ 轮询错误: {str(e)}</div>'
                return (error_status_html, {}, "", gr.update(visible=False), False)
        
        # 轮询定时器已移除，现在直接返回最终结果
    
    return demo

# 创建Gradio界面
gradio_app = create_gradio_interface()

# 将Gradio应用挂载到FastAPI，使用标准配置
try:
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")
    print("Gradio界面成功挂载到 /ui")
except Exception as e:
    print(f"Gradio挂载失败: {e}")
    print("尝试备用挂载方式...")
    try:
        # 备用挂载方式
        import gradio as gr
        app.mount("/ui", gradio_app)
        print("Gradio界面使用备用方式挂载")
    except Exception as e2:
        print(f"备用挂载也失败: {e2}")
        print("将在没有Gradio界面的情况下运行")

# 根路径重定向到Gradio界面
@app.get("/")
async def root():
    """根路径重定向到Gradio界面"""
    return HTMLResponse("""
    <html>
        <head>
            <title>Whisper 服务</title>
            <meta http-equiv="refresh" content="0; url=/ui">
        </head>
        <body>
            <p>正在重定向到 <a href="/ui">Whisper 界面</a>...</p>
        </body>
    </html>
    """)

# ----------------------------
# 启动服务
# ----------------------------
if __name__ == '__main__':
    import uvicorn
    
    # 设置环境变量避免字体和缓存问题
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'  # 使用0.0.0.0避免主机名验证问题
    os.environ['GRADIO_CACHE_DIR'] = os.path.join(os.getcwd(), '.gradio_cache')
    # 禁用Gradio的各种验证
    os.environ['GRADIO_SERVER_HEADERS'] = 'Access-Control-Allow-Origin: *'
    os.environ['GRADIO_SHARE'] = 'False'
    os.environ['GRADIO_ALLOW_FLAGGING'] = 'never'
    # 尝试禁用IP验证
    os.environ['GRADIO_VALIDATE_QUEUE'] = 'False'
    
    print("=" * 60)
    print("🎙️ 整合版 Whisper 语音转文字服务")
    print("=" * 60)
    print(f"🌐 API服务地址: {config.BASE_URL}")
    print(f"📱 Gradio界面: {config.GRADIO_URL}")
    print(f"📚 API文档: {config.DOCS_URL}")
    print(f"🔑 固定Token: {config.API_TOKEN}")
    print(f"🧠 默认模型: {config.DEFAULT_MODEL}")
    print(f"💻 计算设备: {config.DEFAULT_DEVICE}")
    print(f"📁 输出目录: {config.OUTPUT_FOLDER}")
    print("=" * 60)
    print("=" * 60)
    print("默认用户账号:")
    print("  Username: admin, Password: admin123")
    print("  Username: user, Password: user123")
    print("=" * 60)
    print("提示: 字体文件404错误不影响核心功能，已自动处理")
    print("=" * 60)
    
    uvicorn.run(app, host=config.HOST, port=config.PORT)
