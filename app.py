#!/usr/bin/env python3
"""
server.py (Version A - Full)

- FastAPI REST API with Bearer token auth
- Gradio UI mounted at /ui (also requires token)
- faster-whisper for ASR
- ffmpeg for audio extraction / subtitle burn / mux
- Support upload / local path / remote url inputs
- Support bilingual (original + English translate) option
- Fix for Gradio NamedString upload object via ensure_local_file()
- Returns job_id; files downloadable via /api/download?job_id=...&file=...
"""

import os
import io
import sys
import uuid
import shutil
import tempfile
import asyncio
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import timedelta, datetime

import requests
import subprocess
from faster_whisper import WhisperModel

from fastapi import FastAPI, File, UploadFile, Header, Body, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

import gradio as gr

# ----------------------------
# Configuration
# ----------------------------
API_TOKEN = os.environ.get("API_TOKEN", "changeme")  # MUST set in production
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 7860))

DEFAULT_MODEL = os.environ.get("FW_MODEL", "small")
DEFAULT_DEVICE = os.environ.get("FW_DEVICE", "cpu")
DEFAULT_COMPUTE = os.environ.get("FW_COMPUTE", "int8")  # e.g. "int8", "float16", or None

# Keep job registry to allow downloads; in production use persistent storage
JOBS: Dict[str, Dict[str, Any]] = {}

# ----------------------------
# Utilities
# ----------------------------

def get_output_dir():
    """Get or create output directory in current code directory"""
    current_dir = Path(__file__).parent
    output_dir = current_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir

def create_temp_dir(prefix: str = "job_"):
    """Create a temporary directory with date-based naming"""
    output_dir = get_output_dir()
    today = datetime.now().strftime("%Y%m%d")
    timestamp = datetime.now().strftime("%H%M%S")
    temp_dir_name = f"{prefix}{today}-{timestamp}"
    temp_dir = output_dir / temp_dir_name
    temp_dir.mkdir(exist_ok=True)
    return temp_dir

def debug_log(message: str, job_id: str = None):
    """Write debug information to log file"""
    import time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    debug_dir = Path(__file__).parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    
    log_file = debug_dir / "asr_debug.log"
    with open(log_file, "a", encoding="utf-8") as f:
        if job_id:
            f.write(f"[{timestamp}] [JOB:{job_id}] {message}\n")
        else:
            f.write(f"[{timestamp}] {message}\n")

def run_cmd(cmd: List[str]):
    """Run command and raise on error, returning stdout."""
    # Handle Windows-specific path issues
    if os.name == 'nt':  # Windows
        cmd = [str(c) for c in cmd]
    
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=os.name == 'nt')
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc.stdout

def download_url_to_file(url: str, out_path: Path, timeout=60):
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    debug_log(f"Downloading URL: {url} to {out_path}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    debug_log(f"Downloaded successfully: {out_path}")

def ffmpeg_extract_audio(input_path: Path, out_wav: Path, sample_rate=16000):
    # Ensure paths are absolute and properly escaped
    input_path = Path(input_path).resolve()
    out_wav = Path(out_wav).resolve()
    
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", str(sample_rate),
        "-vn", "-f", "wav", str(out_wav)
    ]
    run_cmd(cmd)

def format_timestamp(seconds: float):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((td.total_seconds() - int(td.total_seconds())) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def write_srt_from_segments(segments, out_srt: Path, bilingual: bool=False, translated_segments=None):
    # Ensure the output directory exists
    out_srt = Path(out_srt).resolve()
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            orig = seg.text.strip()
            if bilingual and translated_segments:
                trans = translated_segments[i-1].text.strip() if i-1 < len(translated_segments) else ""
                text_block = (orig + "\n" + trans).strip()
            else:
                text_block = orig
            f.write(f"{i}\n{start} --> {end}\n{text_block}\n\n")

def burn_hardsub(input_video: Path, srt_file: Path, out_video: Path, force_style: Optional[str]=None):
    # Ensure paths are absolute
    input_video = Path(input_video).resolve()
    srt_file = Path(srt_file).resolve()
    out_video = Path(out_video).resolve()
    
    # Ensure output directory exists
    out_video.parent.mkdir(parents=True, exist_ok=True)
    
    # Read SRT content and adjust timestamps to fit video duration
    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    # Get video duration using ffprobe
    cmd_duration = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(input_video)
    ]
    try:
        duration_result = subprocess.run(cmd_duration, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration_str = duration_result.stdout.strip()
        video_duration = float(duration_str) if duration_str else None
    except:
        video_duration = None
    
    # Create adjusted SRT file with timestamps limited to video duration
    adjusted_srt = out_video.parent / f"{out_video.stem}_adjusted.srt"
    if video_duration:
        # Parse and adjust SRT timestamps
        import re
        lines = srt_content.split('\n')
        adjusted_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.isdigit():
                # This is a subtitle number
                adjusted_lines.append(line)
                i += 1
                if i < len(lines):
                    # This should be the timestamp line
                    timestamp_line = lines[i].strip()
                    # Parse start and end times
                    match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', timestamp_line)
                    if match:
                        start_time = match.group(1)
                        end_time = match.group(2)
                        # Convert to seconds
                        def time_to_seconds(time_str):
                            h, m, s_ms = time_str.split(':')
                            s, ms = s_ms.split(',')
                            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
                        
                        start_seconds = time_to_seconds(start_time)
                        end_seconds = time_to_seconds(end_time)
                        
                        # Adjust if end time exceeds video duration
                        if end_seconds > video_duration:
                            if start_seconds >= video_duration:
                                # Skip this subtitle entirely
                                i += 2  # Skip the text line too
                                continue
                            else:
                                # Truncate to video duration
                                end_seconds = video_duration - 0.1  # Small margin
                                # Convert back to timestamp format
                                def seconds_to_time(seconds):
                                    h = int(seconds // 3600)
                                    m = int((seconds % 3600) // 60)
                                    s = int(seconds % 60)
                                    ms = int((seconds - int(seconds)) * 1000)
                                    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                                
                                end_time = seconds_to_time(end_seconds)
                                timestamp_line = f"{start_time} --> {end_time}"
                    
                    adjusted_lines.append(timestamp_line)
                    i += 1
                    if i < len(lines):
                        # Add the text line
                        adjusted_lines.append(lines[i])
                        i += 1
                # Add empty line if present
                if i < len(lines) and not lines[i].strip():
                    adjusted_lines.append('')
                    i += 1
            else:
                adjusted_lines.append(line)
                i += 1
        
        adjusted_content = '\n'.join(adjusted_lines)
        with open(adjusted_srt, 'w', encoding='utf-8') as f:
            f.write(adjusted_content)
        srt_to_use = adjusted_srt
    else:
        srt_to_use = srt_file
    
    # Use a simpler approach that works better on Windows
    # First, try to use the subtitles filter with the file path directly
    style_arg = f":force_style='{force_style}'" if force_style else ""
    
    # Create a temporary working directory with shorter paths if needed
    if os.name == 'nt' and len(str(srt_to_use)) > 200:
        # Create a temp directory with shorter path
        temp_dir = create_temp_dir(prefix="sub_")
        temp_srt = temp_dir / "subtitle.srt"
        shutil.copy2(srt_to_use, temp_srt)
        srt_to_use = temp_srt
    
    # Add debug logging
    debug_log(f"Attempting to burn subtitles from {srt_to_use} to {input_video}")
    debug_log(f"SRT file exists: {srt_to_use.exists()}")
    debug_log(f"SRT file size: {srt_to_use.stat().st_size if srt_to_use.exists() else 0} bytes")
    
    # Check if SRT file has content
    if srt_to_use.exists():
        with open(srt_to_use, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            debug_log(f"SRT content preview: {content[:200]}...")
            if not content:
                raise RuntimeError("SRT file is empty")
    
    # Use the original simple approach but with better error handling
    try:
        # First, let's verify the SRT file format and content
        with open(srt_to_use, 'r', encoding='utf-8') as f:
            srt_lines = f.readlines()
            debug_log(f"SRT file has {len(srt_lines)} lines")
            if len(srt_lines) >= 3:
                debug_log(f"First few lines: {srt_lines[:3]}")
        
        # Try the simple subtitle filter first with explicit encoding
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-vf", f"subtitles='{srt_to_use}':force_style='Fontsize=20,PrimaryColour=&Hffffff,BackColour=&H80000000,BorderStyle=1'",
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "fast",
            str(out_video)
        ]
        debug_log(f"Running command: {' '.join(cmd)}")
        run_cmd(cmd)
        debug_log(f"Successfully created output video: {out_video}")
        
        # Verify the output was created and has reasonable size
        if out_video.exists() and out_video.stat().st_size > 1000:
            debug_log(f"Output video created successfully, size: {out_video.stat().st_size} bytes")
        else:
            raise RuntimeError("Output video file is too small or not created")
            
    except RuntimeError as e:
        debug_log(f"First method failed: {e}")
        # If that fails, try with a different approach - convert SRT to ASS format
        try:
            ass_file = srt_to_use.with_suffix('.ass')
            # Convert SRT to ASS using ffmpeg
            cmd_convert = [
                "ffmpeg", "-y",
                "-i", str(srt_to_use),
                "-c:s", "ass",
                str(ass_file)
            ]
            debug_log(f"Converting SRT to ASS: {' '.join(cmd_convert)}")
            run_cmd(cmd_convert)
            
            # Now use ASS file for subtitles
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video),
                "-vf", f"ass='{ass_file}'",
                "-c:a", "copy",
                "-c:v", "libx264",
                "-preset", "fast",
                str(out_video)
            ]
            debug_log(f"Running second method with ASS: {' '.join(cmd)}")
            run_cmd(cmd)
            debug_log(f"Second method succeeded")
            
        except RuntimeError as e2:
            debug_log(f"Second method failed: {e2}")
            # Try using subtitles as a separate stream (soft subs approach)
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(input_video),
                    "-i", str(srt_to_use),
                    "-c:v", "copy",
                    "-c:a", "copy",
                    "-c:s", "mov_text",
                    "-metadata:s:s:0", "language=chi",
                    "-disposition:s:0", "default",
                    str(out_video.with_suffix('.mp4'))
                ]
                debug_log(f"Running third method (soft subs): {' '.join(cmd)}")
                run_cmd(cmd)
                debug_log(f"Third method succeeded - created soft subs")
                
                # If soft subs worked, try to burn them
                out_video_soft = out_video.with_suffix('.mp4')
                cmd_hard = [
                    "ffmpeg", "-y",
                    "-i", str(out_video_soft),
                    "-vf", "subtitles",
                    "-c:a", "copy",
                    str(out_video)
                ]
                debug_log(f"Burning soft subs to hard subs: {' '.join(cmd_hard)}")
                run_cmd(cmd_hard)
                debug_log(f"Hard subs from soft subs succeeded")
                
            except RuntimeError as e3:
                debug_log(f"Third method failed: {e3}")
                # Last resort - try with very basic parameters
                try:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(input_video),
                        "-i", str(srt_to_use),
                        "-filter_complex", "[0:v][1:s]overlay=5:5[v]",
                        "-map", "[v]",
                        "-map", "0:a?",
                        "-c:v", "libx264",
                        "-preset", "ultrafast",
                        "-crf", "28",
                        str(out_video)
                    ]
                    debug_log(f"Running fourth method (overlay): {' '.join(cmd)}")
                    run_cmd(cmd)
                    debug_log(f"Fourth method succeeded")
                except RuntimeError as e4:
                    debug_log(f"All methods failed. Last error: {e4}")
                    # Create a simple test SRT to verify the issue
                    test_srt = srt_to_use.parent / "test_subtitles.srt"
                    with open(test_srt, 'w', encoding='utf-8') as f:
                        f.write("1\n00:00:01,000 --> 00:00:03,000\nTest Subtitle\n\n")
                    
                    cmd_test = [
                        "ffmpeg", "-y",
                        "-i", str(input_video),
                        "-vf", f"subtitles='{test_srt}'",
                        "-c:a", "copy",
                        str(out_video.with_suffix('_test.mp4'))
                    ]
                    debug_log(f"Testing with simple subtitle: {' '.join(cmd_test)}")
                    run_cmd(cmd_test)
                    debug_log(f"Test with simple subtitle completed")
                    
                    raise RuntimeError(f"Failed to burn subtitles. All methods failed. Last error: {e4}")
            except RuntimeError as e3:
                debug_log(f"All methods failed. Last error: {e3}")
                raise RuntimeError(f"Failed to burn subtitles. All methods failed. Last error: {e3}")
    
    # Verify output file was created
    if not out_video.exists():
        raise RuntimeError(f"Output video file was not created: {out_video}")
    
    debug_log(f"Final output file size: {out_video.stat().st_size} bytes")

def mux_softsub(input_video: Path, srt_file: Path, out_video: Path):
    # Ensure paths are absolute
    input_video = Path(input_video).resolve()
    srt_file = Path(srt_file).resolve()
    out_video = Path(out_video).resolve()
    
    # Ensure output directory exists
    out_video.parent.mkdir(parents=True, exist_ok=True)
    
    ext = out_video.suffix.lower()
    if ext in (".mp4", ".mov", ".m4v"):
        cmd = ["ffmpeg", "-y", "-i", str(input_video), "-i", str(srt_file),
               "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text", str(out_video)]
    else:
        cmd = ["ffmpeg", "-y", "-i", str(input_video), "-i", str(srt_file),
               "-map", "0", "-map", "1", "-c", "copy", "-c:s", "srt", str(out_video)]
    run_cmd(cmd)

# ----------------------------
# Gradio / Upload helpers
# ----------------------------
def ensure_local_file(obj, tmpdir: Optional[Path]=None) -> Path:
    """
    Convert various input types (Gradio NamedString, path string, UploadFile, bytes, io.BytesIO)
    into a local Path (copied into a temp directory).
    """
    debug_log(f"ensure_local_file called with object type: {type(obj)}")
    
    if tmpdir is None:
        tmpdir = create_temp_dir(prefix="ensure_")
    else:
        tmpdir = Path(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
    
    debug_log(f"Using temp directory: {tmpdir}")

    # 1) If it's a gradio NamedString (often has .name pointing to a temp filepath)
    try:
        NamedString = getattr(gr.utils, "NamedString", None)
    except Exception:
        NamedString = None

    # If it's a Gradio NamedString-like (has .name attr that's a file path)
    # But exclude Path objects which also have .name but should be handled differently
    if hasattr(obj, "name") and isinstance(getattr(obj, "name"), str) and not isinstance(obj, Path):
        p = Path(obj.name)
        if p.exists():
            dst = tmpdir / p.name
            shutil.copy2(str(p), str(dst))
            return dst
        else:
            raise RuntimeError(f"Gradio file path not found: {obj.name}")

    # If it's a Path object
    if isinstance(obj, Path):
        p = obj.resolve()  # Get absolute path
        if p.exists():
            dst = tmpdir / p.name
            shutil.copy2(str(p), str(dst))
            return dst
        raise RuntimeError(f"Path object points to non-existent file: {obj}")

    # If it's a plain str that points to a file path:
    if isinstance(obj, str):
        p = Path(obj).resolve()  # Get absolute path
        if p.exists():
            dst = tmpdir / p.name
            shutil.copy2(str(p), str(dst))
            return dst
        # If it's a base64 data URL or raw base64, try to handle? (not implemented)
        raise RuntimeError(f"String provided but file not found: {obj}")

    # UploadFile from FastAPI (has .file and .filename)
    if hasattr(obj, "file") and hasattr(obj, "filename"):
        filename = getattr(obj, "filename")
        if not filename:
            filename = "upload.bin"
        dst = tmpdir / Path(filename).name
        with open(dst, "wb") as f:
            # obj.file might be async object; attempt to read bytes
            try:
                content = obj.file.read()
            except Exception:
                # Try .readable -> iterate
                obj.file.seek(0)
                content = obj.file.read()
            if isinstance(content, str):
                content = content.encode()
            f.write(content)
        return dst

    # If it's a tuple like (tempfile_path, filename) (older gradio versions)
    if isinstance(obj, (tuple, list)) and len(obj) >= 1:
        cand = obj[0]
        if isinstance(cand, str):
            p = Path(cand).resolve()
            if p.exists():
                # Use filename from tuple if available, otherwise use the original filename
                if len(obj) >= 2 and obj[1]:
                    filename = obj[1]
                else:
                    filename = p.name
                dst = tmpdir / filename
                shutil.copy2(str(p), str(dst))
                return dst
            else:
                raise RuntimeError(f"Tuple file path not found: {cand}")

    # If it's bytes or bytearray
    if isinstance(obj, (bytes, bytearray)):
        dst = tmpdir / "upload.bin"
        with open(dst, "wb") as f:
            f.write(obj)
        return dst

    # If it's an io.BytesIO or has .read()
    if hasattr(obj, "read"):
        # attempt to read and write
        content = obj.read()
        if isinstance(content, str):
            content = content.encode()
        dst = tmpdir / "upload_from_read"
        with open(dst, "wb") as f:
            f.write(content)
        return dst

    # If nothing matched
    raise RuntimeError(f"Unsupported uploaded object type: {type(obj)}")

# ----------------------------
# faster-whisper wrapper with cache
# ----------------------------
_MODEL_CACHE: Dict[tuple, WhisperModel] = {}

def get_fw_model(model_name: str, device: str, compute_type: Optional[str]):
    key = (model_name, device, compute_type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    
    # Handle compute_type parameter for newer versions of faster-whisper
    if compute_type is None:
        model = WhisperModel(model_name, device=device)
    else:
        # For newer versions, compute_type should be a valid string or dict
        if compute_type == "default":
            model = WhisperModel(model_name, device=device)
        else:
            try:
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
            except Exception:
                # Fallback to default if compute_type is not supported
                debug_log(f"compute_type '{compute_type}' not supported, using default")
                model = WhisperModel(model_name, device=device)
    
    _MODEL_CACHE[key] = model
    return model

def transcribe_audio(wav_path: Path, model_name: str, device: str, compute_type: Optional[str], beam_size: int=5, task: Optional[str]=None, word_timestamps: bool=False):
    model = get_fw_model(model_name, device, compute_type)
    # Ensure task is not None - default to "transcribe"
    if task is None:
        task = "transcribe"
    segments, info = model.transcribe(str(wav_path), beam_size=beam_size, task=task, word_timestamps=word_timestamps)
    return list(segments)

# ----------------------------
# FastAPI app + models
# ----------------------------
app = FastAPI(title="AutoSubtitle Service (faster-whisper + Gradio UI)")

def check_token(auth_header: Optional[str]):
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = parts[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

class TranscribeRequest(BaseModel):
    input_type: str  # upload | local | url | separate_audio
    local_path: Optional[str] = None
    url: Optional[str] = None
    model_name: Optional[str] = DEFAULT_MODEL
    device: Optional[str] = DEFAULT_DEVICE
    compute_type: Optional[str] = None
    bilingual: Optional[bool] = True
    beam_size: Optional[int] = 5
    word_timestamps: Optional[bool] = False
    burn: Optional[str] = "none"  # none|hard|soft
    out_basename: Optional[str] = None
    # For separate_audio mode
    video_local_path: Optional[str] = None
    video_url: Optional[str] = None
    audio_local_path: Optional[str] = None
    audio_url: Optional[str] = None

@app.post("/api/transcribe")
async def api_transcribe(req: TranscribeRequest = Body(...), authorization: Optional[str] = Header(None), 
                        upload_file: Optional[UploadFile] = File(None), 
                        video_file: Optional[UploadFile] = File(None),
                        audio_file: Optional[UploadFile] = File(None)):
    """
    Start a transcription job. Returns job_id and file info.
    - For input_type == "upload": include multipart file upload_file
    - For input_type == "url": provide url in req.url
    - For input_type == "local": provide server local_path (absolute) -- will be copied
    - For input_type == "separate_audio": provide video_file/audio_file or video_url/audio_url or video_local_path/audio_local_path
    """
    check_token(authorization)

    job_id = str(uuid.uuid4())
    job_tmp = create_temp_dir(prefix=f"autosub_{job_id}_")
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
                local_input = ensure_local_file(upload_file, tmpdir=job_tmp)
            elif req.input_type == "url":
                if not req.url:
                    raise RuntimeError("url required for input_type 'url'")
                filename = Path(req.url.split("?")[0].split("/")[-1] or "downloaded_input")
                local_input = job_tmp / filename.name
                download_url_to_file(req.url, local_input)
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
                    local_input = ensure_local_file(video_file, tmpdir=job_tmp)
                elif req.video_url:
                    filename = Path(req.video_url.split("?")[0].split("/")[-1] or "downloaded_video")
                    local_input = job_tmp / filename.name
                    download_url_to_file(req.video_url, local_input)
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
                    audio_input = ensure_local_file(audio_file, tmpdir=job_tmp)
                elif req.audio_url:
                    filename = Path(req.audio_url.split("?")[0].split("/")[-1] or "downloaded_audio")
                    audio_input = job_tmp / filename.name
                    download_url_to_file(req.audio_url, audio_input)
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
                # Use the separate audio file directly
                ffmpeg_extract_audio(audio_input, audio_path, sample_rate=16000)
            else:
                # Extract audio from the main input file
                ffmpeg_extract_audio(local_input, audio_path, sample_rate=16000)
            JOBS[job_id]["files"]["audio_wav"] = str(audio_path)

            model_name = req.model_name or DEFAULT_MODEL
            device = req.device or DEFAULT_DEVICE
            compute_type = req.compute_type or DEFAULT_COMPUTE
            beam = int(req.beam_size or 5)

            # original transcription
            segments = await asyncio.to_thread(transcribe_audio, audio_path, model_name, device, compute_type, beam, None, req.word_timestamps or False)
            JOBS[job_id]["files"]["segments_count"] = len(segments)

            translated_segments = None
            if req.bilingual:
                translated_segments = await asyncio.to_thread(transcribe_audio, audio_path, model_name, device, compute_type, beam, "translate", False)

            out_base = req.out_basename or f"output"
            srt_path = job_tmp / f"{out_base}.srt"
            write_srt_from_segments(segments, srt_path, bilingual=bool(req.bilingual), translated_segments=translated_segments)
            JOBS[job_id]["files"]["srt"] = str(srt_path)

            # optional burn/mux
            # Use local_input for regular modes, or video_input for separate_audio mode
            video_for_burn = local_input
            if req.input_type == "separate_audio" and local_input:
                video_for_burn = local_input
            
            if req.burn == "hard":
                out_video = job_tmp / f"{out_base}_hardsub.mp4"
                await asyncio.to_thread(burn_hardsub, video_for_burn, srt_path, out_video, None)
                JOBS[job_id]["files"]["video"] = str(out_video)
            elif req.burn == "soft":
                out_video = job_tmp / f"{out_base}_softsub.mp4"
                await asyncio.to_thread(mux_softsub, video_for_burn, srt_path, out_video)
                JOBS[job_id]["files"]["video"] = str(out_video)

            JOBS[job_id]["status"] = "finished"
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e) + "\n" + traceback.format_exc()

    # schedule background task
    asyncio.create_task(_do_job())

    return JSONResponse({"ok": True, "job_id": job_id, "tmp": JOBS[job_id]["tmp"]})

@app.get("/api/job_status")
def api_job_status(job_id: str = Query(...), authorization: Optional[str] = Header(None)):
    check_token(authorization)
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job_id not found")
    return JSONResponse({"ok": True, "job": JOBS[job_id]})

@app.get("/api/download")
def api_download(job_id: str = Query(...), file: str = Query(...), authorization: Optional[str] = Header(None)):
    """
    Download a file generated by job.
    file: one of keys in JOBS[job_id]['files'] or a filename under job tmp dir.
    Example: /api/download?job_id=<id>&file=srt
    """
    check_token(authorization)
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job_id not found")
    entry = JOBS[job_id]
    files = entry.get("files", {})
    # if file key exists
    if file in files:
        p = Path(files[file])
        if not p.exists():
            raise HTTPException(status_code=404, detail="file not found on disk")
        return FileResponse(str(p), filename=p.name)
    # else, treat 'file' as filename under tmp
    candidate = Path(entry["tmp"]) / file
    if candidate.exists():
        return FileResponse(str(candidate), filename=candidate.name)
    raise HTTPException(status_code=404, detail="file key not found in job files")

# ----------------------------
# Gradio UI
# ----------------------------
def gradio_transcribe_ui(input_file, input_type, local_path, url, model_name, device, compute_type, bilingual, burn, token_text, 
                         video_file=None, audio_file=None, video_local_path=None, audio_local_path=None, video_url=None, audio_url=None):
    # Validate token
    if token_text != API_TOKEN:
        return "ERROR: invalid token (enter correct API token)", None, None

    tmpdir = create_temp_dir(prefix="gradio_")
    try:
        input_path = None
        audio_input = None
        
        # prepare input
        if input_type == "upload":
            if input_file is None:
                return "Please upload a file", None, None
            input_path = ensure_local_file(input_file, tmpdir)
        elif input_type == "url":
            if not url:
                return "Please provide a URL", None, None
            input_path = tmpdir / Path(url.split("?")[0].split("/")[-1] or "downloaded_input")
            download_url_to_file(url, input_path)
        elif input_type == "local":
            if not local_path:
                return "Please provide a local path", None, None
            p = Path(local_path).resolve()
            if not p.exists():
                return f"Local path not found: {local_path}", None, None
            input_path = tmpdir / p.name
            shutil.copy2(str(p), str(input_path))
        elif input_type == "separate_audio":
            # Handle video file
            if video_file is not None:
                input_path = ensure_local_file(video_file, tmpdir)
            elif video_url:
                filename = Path(video_url.split("?")[0].split("/")[-1] or "downloaded_video")
                input_path = tmpdir / filename.name
                download_url_to_file(video_url, input_path)
            elif video_local_path:
                p = Path(video_local_path).resolve()
                if not p.exists():
                    return f"Video local path not found: {video_local_path}", None, None
                input_path = tmpdir / p.name
                shutil.copy2(str(p), str(input_path))
            else:
                return "Please provide a video file", None, None
            
            # Handle audio file
            if audio_file is not None:
                audio_input = ensure_local_file(audio_file, tmpdir)
            elif audio_url:
                filename = Path(audio_url.split("?")[0].split("/")[-1] or "downloaded_audio")
                audio_input = tmpdir / filename.name
                download_url_to_file(audio_url, audio_input)
            elif audio_local_path:
                p = Path(audio_local_path).resolve()
                if not p.exists():
                    return f"Audio local path not found: {audio_local_path}", None, None
                audio_input = tmpdir / p.name
                shutil.copy2(str(p), str(audio_input))
            else:
                return "Please provide an audio file", None, None
        else:
            return "Invalid input_type", None, None

        audio_path = tmpdir / "audio.wav"
        if input_type == "separate_audio":
            # Use the separate audio file directly
            ffmpeg_extract_audio(audio_input, audio_path, sample_rate=16000)
        else:
            # Extract audio from the main input file
            ffmpeg_extract_audio(input_path, audio_path, sample_rate=16000)

        model_name = model_name or DEFAULT_MODEL
        device = device or DEFAULT_DEVICE
        compute_type = compute_type or DEFAULT_COMPUTE

        segments = transcribe_audio(audio_path, model_name, device, compute_type, beam_size=5, task=None, word_timestamps=False)
        translated_segments = None
        if bilingual:
            translated_segments = transcribe_audio(audio_path, model_name, device, compute_type, beam_size=5, task="translate", word_timestamps=False)

        srt_path = tmpdir / "output.srt"
        write_srt_from_segments(segments, srt_path, bilingual=bilingual, translated_segments=translated_segments)

        out_video_path = None
        if burn == "hard":
            out_video_path = tmpdir / "out_hardsub.mp4"
            burn_hardsub(input_path, srt_path, out_video_path)
        elif burn == "soft":
            out_video_path = tmpdir / "out_softsub.mp4"
            mux_softsub(input_path, srt_path, out_video_path)

        # Return message and file paths for Gradio to show download
        if out_video_path and out_video_path.exists():
            msg = f"Done. SRT: {srt_path.name}, Video: {out_video_path.name}"
            # Return both SRT and video files
            return msg, str(srt_path), str(out_video_path)
        else:
            msg = f"Done. SRT: {srt_path.name}"
            # Return only SRT file
            return msg, str(srt_path), None
    except Exception as e:
        return f"Error: {e}", None, None

# Build Gradio interface
with gr.Blocks(title="AutoSubtitle") as demo:
    gr.Markdown("# AutoSubtitle (faster-whisper)\nEnter API token and options.")
    with gr.Row():
        token_input = gr.Textbox(label="Access Token (Bearer)", value="", placeholder="Enter API token")
        model_input = gr.Textbox(label="Model name", value=DEFAULT_MODEL)
        device_input = gr.Textbox(label="Device", value=DEFAULT_DEVICE)
        compute_input = gr.Textbox(label="Compute type (optional)", value=str(DEFAULT_COMPUTE) if DEFAULT_COMPUTE else "")
    
    with gr.Row():
        inp_type = gr.Dropdown(choices=["upload", "local", "url", "separate_audio"], value="upload", label="Input type")
    
    # Regular input options
    with gr.Group(visible=True) as regular_input_group:
        upload_file = gr.File(label="Upload file (video/audio)", file_count="single")
        local_path = gr.Textbox(label="Server local path (if input_type=local)", placeholder="/path/to/video.mp4")
        url_box = gr.Textbox(label="Remote URL (if input_type=url)", placeholder="https://example.com/video.mp4")
    
    # Separate audio input options
    with gr.Group(visible=False) as separate_audio_group:
        gr.Markdown("### Separate Audio Mode")
        with gr.Row():
            video_file = gr.File(label="Upload video file", file_count="single")
            audio_file = gr.File(label="Upload audio file", file_count="single")
        with gr.Row():
            video_local_path = gr.Textbox(label="Video local path", placeholder="/path/to/video.mp4")
            audio_local_path = gr.Textbox(label="Audio local path", placeholder="/path/to/audio.wav")
        with gr.Row():
            video_url = gr.Textbox(label="Video URL", placeholder="https://example.com/video.mp4")
            audio_url = gr.Textbox(label="Audio URL", placeholder="https://example.com/audio.wav")
    
    with gr.Row():
        bilingual_checkbox = gr.Checkbox(label="Generate bilingual subtitles (original + English)", value=True)
        burn_select = gr.Radio(choices=["none", "hard", "soft"], value="none", label="Burn subtitles (none/hard/soft)")
    run_btn = gr.Button("Run")
    result_output = gr.Textbox(label="Result")
    download_srt = gr.File(label="Download SRT (after run)")
    download_video = gr.File(label="Download Video (if available)")

    def toggle_input_visibility(input_type):
        if input_type == "separate_audio":
            return gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=False)

    def _run_gradio(upload_file, inp_type, local_path, url_box, model_input, device_input, compute_input, bilingual_checkbox, burn_select, token_input,
                   video_file, audio_file, video_local_path, audio_local_path, video_url, audio_url):
        return gradio_transcribe_ui(upload_file, inp_type, local_path, url_box, model_input, device_input, compute_input, bilingual_checkbox, burn_select, token_input,
                                   video_file, audio_file, video_local_path, audio_local_path, video_url, audio_url)

    inp_type.change(fn=toggle_input_visibility, inputs=[inp_type], outputs=[regular_input_group, separate_audio_group])
    run_btn.click(fn=_run_gradio, 
                  inputs=[upload_file, inp_type, local_path, url_box, model_input, device_input, compute_input, bilingual_checkbox, burn_select, token_input,
                          video_file, audio_file, video_local_path, audio_local_path, video_url, audio_url], 
                  outputs=[result_output, download_srt, download_video])

# mount gradio under /ui
app = gr.mount_gradio_app(app, demo, path="/ui")

@app.get("/")
def index():
    return HTMLResponse("<html><body><h3>AutoSubtitle Service</h3><p>Use the Gradio UI at <a href='/ui'>/ui</a> or API endpoints under /api.</p></body></html>")

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on {HOST}:{PORT} (API_TOKEN={'SET' if API_TOKEN else 'NOT SET'})")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
