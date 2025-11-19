from faster_whisper import WhisperModel
import json
import threading
import os
from typing import List, Dict, Any, Optional

class WhisperService:
    """Whisper 语音转文字服务类，支持模型复用"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.model_size = "small"
            self.device = "cpu"
            self.compute_type = "int8"
            self.cpu_threads = 8
            self.model = None
            self._load_model()
            self.initialized = True
    
    def _load_model(self):
        """加载 Whisper 模型"""
        try:
            print(f"Loading Whisper model: {self.model_size}")
            self.model = WhisperModel(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads
            )
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio_path: str, beam_size: int = 5) -> Dict[str, Any]:
        """
        语音转文字
        
        Args:
            audio_path: 音频文件路径
            beam_size: beam search 大小
            
        Returns:
            包含转录结果的字典
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            segments, info = self.model.transcribe(audio_path, beam_size=beam_size)
            
            result = {
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": []
            }
            
            for segment in segments:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                result["segments"].append(segment_data)
            
            return result
            
        except Exception as e:
            print(f"Transcription error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads
        }