"""
Based on https://github.com/morioka/tiny-openai-whisper-api
"""

UPLOAD_DIR = "tmp"

import os
import shutil
from datetime import timedelta
from functools import lru_cache
from typing import Optional, Annotated
import simplejson as json
from pynvml import *


import hmac
import hashlib

import numpy as np
import uvicorn
import whisper
import faster_whisper
from fastapi import FastAPI, Form, UploadFile, File, Header, Response
from fastapi import HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import time
import inspect





# 开始基础配置

MODEL_NAME = "large-v3"

WHISPER_DEFAULT_SETTINGS = {
    #"whisper_model": "base",
    "whisper_model": MODEL_NAME,
    "temperature": 0.0,
    "temperature_increment_on_fallback": 0.2,
    "no_speech_threshold": 0.6,
    "logprob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "condition_on_previous_text": True,
    "verbose": False,
    "task": "transcribe",
}


# 开始初始各种类重载
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        print("obj", obj)
        if isinstance(obj, float):
            if obj == float('inf') or obj == -float('inf'):
                return 0  # 或者返回其他你想要的值，比如 0
            elif obj != obj:  # 检查 NaN
                return 0  # 或者返回其他你想要的值，比如 0
        return super().default(obj)
    



# 开始初始化服务器配置


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# from pyinstrument import Profiler
# from fastapi import Request
# from fastapi.responses import HTMLResponse

# @app.middleware("http")
# async def profile_request(request: Request, call_next):
#     profiling = request.query_params.get("profile", False)
#     if profiling:
#         profiler = Profiler(interval=0.01, async_mode="enabled")
#         profiler.start()
#         await call_next(request)
#         profiler.stop()
#         return HTMLResponse(profiler.output_html())
#     else:
#         return await call_next(request)

@lru_cache(maxsize=1)
def get_gpu_name():
    nvmlInit()
    ret = []
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        ret.append(nvmlDeviceGetName(handle))
    nvmlShutdown()
    return ret

@lru_cache(maxsize=1)
def get_whisper_model(whisper_model: str):
    """Get a whisper model from the cache or download it if it doesn't exist"""
    model = whisper.load_model(whisper_model)

    return model

@lru_cache(maxsize=1)
def get_faster_whisper_model(whisper_model: str):
    """Get a whisper model from the cache or download it if it doesn't exist"""
    model_size = MODEL_NAME
    device, compute_type = "cuda", "float16"
    model = faster_whisper.WhisperModel(model_size, device=device, compute_type=compute_type)

    return model

@lru_cache(maxsize=1)
def get_hostname() -> str:
    import socket
    return socket.gethostname()

def transcribe(audio_path: str, whisper_model: str, **whisper_args):
    """Transcribe the audio file using whisper"""

    # Get whisper model
    # NOTE: If multiple models are selected, this may keep all of them in memory depending on the cache size
    transcriber = get_whisper_model(whisper_model)

    # Set configs & transcribe
    if whisper_args["temperature_increment_on_fallback"] is not None:
        whisper_args["temperature"] = tuple(
            np.arange(
                whisper_args["temperature"],
                1.0 + 1e-6,
                whisper_args["temperature_increment_on_fallback"],
            )
        )
    else:
        whisper_args["temperature"] = [whisper_args["temperature"]]

    del whisper_args["temperature_increment_on_fallback"]

    transcript = transcriber.transcribe(
        audio_path,
        **whisper_args,
    )

    return transcript


def faster_transcribe(audio_path :str):
    model = get_faster_whisper_model(MODEL_NAME)
    try:
        stime = time.time()
        segments, info = model.transcribe(
            audio_path, 
            word_timestamps=False,
            vad_filter=True,
            temperature=0,
            language=None,
            initial_prompt=None,
        )
        inference_time_1 = time.time() - stime
        
        # print("本次识别到的文字: %s" % segments)
        # debugSegments = copy.deepcopy(segments)
        # for s in debugSegments:
        #     print(s)

    except ValueError as e:
        # 没有识别到语言的时候可能会报 ValueError: max() arg is an empty sequence
        # 进行没有识别到语言的处理
        # print("输入语音长度 %0.3f 秒，本次没有识别到文字: %s" % (info.duration, str(e)))
        print("本次没有识别到文字: %s" % str(e))
        return []


    # 注意，segments 是一个 generator，在获取的时候似乎仍需要调用神经网络进行处理，所以这里也计算一下推理时间
    stime2 = time.time()
    segments = list(segments)
    inference_time_2 = time.time() - stime2

    inference_time = time.time() - stime

    print("推理耗时 %0.3f + %0.3f = %0.3f 秒，输入语音长度 %0.3f 秒，猜测语言 %s（概率 %0.3f)" % (inference_time_1, inference_time_2, inference_time_1 + inference_time_2, info.duration, info.language, info.language_probability))

    return {
        "segments": segments,
        "info": info,
        "inference_time": inference_time,
        "inference_time_1": inference_time_1,
        "inference_time_2": inference_time_2,
        "gpus": get_gpu_name(),
        "hostname": get_hostname(),
    }

def test_serialization(segments):
    # 在其他代码之前添加这些行

    fields = ['id', 'seek', 'start', 'end', 'text', 'tokens', 'temperature', 'avg_logprob', 'compression_ratio', 'no_speech_prob', 'words']
    
    for field in fields:
        start_time = time.time()
        try:
            json.dumps(
                [{field: getattr(segment, field, None)} for segment in segments], 
                ensure_ascii = False,
                ignore_nan = True,
                iterable_as_array = True,
                indent = None,
                separators = (',', ':'),
                cls = CustomJSONEncoder,
            )
            end_time = time.time()
            print(f"字段 '{field}' 序列化耗时: {end_time - start_time:.4f} 秒")
        except Exception as e:
            print(f"字段 '{field}' 序列化失败: {str(e)}")

    # 测试所有字段一起序列化
    start_time = time.time()
    try:
        json.dumps(
            [{f: getattr(segment, f, None) for f in fields} for segment in segments], 
            ensure_ascii = False,
            ignore_nan = True,
            iterable_as_array = True,
            indent = None,
            separators = (',', ':'),
            cls = CustomJSONEncoder,
        )
        end_time = time.time()
        print(f"所有字段一起序列化耗时: {end_time - start_time:.4f} 秒")
    except Exception as e:
        print(f"所有字段一起序列化失败: {str(e)}")

def remove_generators(obj):
    if isinstance(obj, dict):
        return {k: remove_generators(v) for k, v in obj.items() 
                if not inspect.isgenerator(v)}
    elif isinstance(obj, list):
        return [remove_generators(item) for item in obj]
    elif inspect.isgenerator(obj):
        return None  # 或者返回一个空列表 []，取决于你的需求
    else:
        return obj



stime = time.time()
print("开始预加载模型")
get_faster_whisper_model(MODEL_NAME)
print("预加载模型完成，耗时 %0.3f 秒" % (time.time() - stime))


@app.post("/ping")
async def ping():
    return {"message": "pong", "time": time.time()}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    # model: str = Form(...),
    file: UploadFile = File(...),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    settings_override: Optional[dict] = Form(None),
    authorization: Annotated[str , Header()] = "",
):
    # assert model == "whisper-1"
    stime = time.time()
    print("[transcript] %0.6f 开始处理 HTTP 请求" % (time.time() - stime))

    # 进行认证检查
    # 获取 Authorization 头
    # 其结构是: Authorization: 1234567890.abcdef12354xxxxxxx
    # 其中 1234567890 是时间戳，abcdef12345 是 HAMC_SHA256(1234567890, KIWI_API_KEY)
    # KIWI_API_KEY 由环境变量提供
    KIWI_API_KEY = os.getenv("KIWI_API_KEY")
    if KIWI_API_KEY is None:
        print("未设置 KIWI_API_KEY，不会执行 API 认证检查")
    else:
        print("获取到 Authorization 是", authorization)

        if authorization == "":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized, 未提供 Authorization 头"
            )

        auths = authorization.split(".")
        if len(auths) != 2:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized, 签名格式错误"
            )

        tsStr = auths[0]
        ts = 0
        if tsStr.isdigit():
            ts = int(tsStr)

        if time.time() - ts > 60:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized, 时间戳超出范围"
            )
        
        theirSign = auths[1]
        ourSign = hmac.new(KIWI_API_KEY.encode(), tsStr.encode(), hashlib.sha256).hexdigest()
        if theirSign != ourSign:
            # print("我方 KEY: `%s'" % KIWI_API_KEY)
            # print("我方 sign: ", ourSign)
            # print("对方 sign: ", theirSign)
            # print("我方数据: `%s'" % tsStr)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized, 签名不正确"
            )

    # print("[transcript] %0.6f 认证检查完成" % (time.time() - stime))

    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Bad Request, bad file"
        )

    if response_format is None:
        response_format = "json"
    if response_format not in ["json", "text", "srt", "verbose_json", "vtt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bad Request, bad response_format",
        )

    if temperature is None:
        temperature = 0.0
    if temperature < 0.0 or temperature > 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bad Request, bad temperature",
        )

    # print("[transcript] %0.6f 参数检查完成" % (time.time() - stime))

    filename = file.filename
    fileobj = file.file
    upload_name = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    # print("[transcript] %0.6f 上传目录初始化完成" % (time.time() - stime))


    with open(upload_name, "wb+") as upload_file:
        shutil.copyfileobj(fileobj, upload_file)

    # print("[transcript] %0.6f 上传文件复制完成" % (time.time() - stime))

    whisper_args = WHISPER_DEFAULT_SETTINGS.copy()
    if settings_override is not None:
        whisper_args.update(settings_override)

    # 普通 whisper
    # transcript = transcribe(audio_path=upload_name, **whisper_args)
    # return transcript

    # faster_whisper
    # print("[transcript] %0.6f 准备开始进行 faster_transcribe 推理" % (time.time() - stime))
    result = faster_transcribe(audio_path=upload_name)
    # print("[transcript] %0.6f faster_transcribe 推理完成" % (time.time() - stime))

    # print("[transcript] %0.6f 准备开始测试序列化速度" % (time.time() - stime))
    # test_serialization(result['segments'])
    # print("[transcript] %0.6f 序列化速度测试完成" % (time.time() - stime))

    
    # stime = time.time()

    # print("[transcript] %0.6f 准备开始序列化" % (time.time() - stime))
    resultJ = json.dumps(
        result,
        iterable_as_array = True,
        ensure_ascii = False,
        ignore_nan = True,
        indent = None,
        separators = (',', ':'),
    ).encode("utf-8")

    # print("[transcript] %0.6f 序列化完成" % (time.time() - stime))


                         
    return Response(resultJ, media_type="application/json")

    if response_format in ["text"]:
        return transcript["text"]

    if response_format in ["srt"]:
        ret = ""
        for seg in transcript["segments"]:
            td_s = timedelta(milliseconds=seg["start"] * 1000)
            td_e = timedelta(milliseconds=seg["end"] * 1000)

            t_s = f"{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}"
            t_e = f"{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}"

            ret += "{}\n{} --> {}\n{}\n\n".format(seg["id"], t_s, t_e, seg["text"])
        ret += "\n"
        return ret

    if response_format in ["vtt"]:
        ret = "WEBVTT\n\n"
        for seg in transcript["segments"]:
            td_s = timedelta(milliseconds=seg["start"] * 1000)
            td_e = timedelta(milliseconds=seg["end"] * 1000)

            t_s = f"{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}"
            t_e = f"{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}"

            ret += "{} --> {}\n{}\n\n".format(t_s, t_e, seg["text"])
        return ret

    if response_format in ["verbose_json"]:
        transcript.setdefault("task", whisper_args["task"])
        transcript.setdefault("duration", transcript["segments"][-1]["end"])
        return transcript

    return {"text": transcript["text"]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
