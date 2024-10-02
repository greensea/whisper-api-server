"""
Based on https://github.com/morioka/tiny-openai-whisper-api
"""

import os
import shutil
from datetime import timedelta
from functools import lru_cache
from typing import Optional, Annotated
from json import dumps


import numpy as np
import uvicorn
import whisper
import faster_whisper
from fastapi import FastAPI, Form, UploadFile, File, Header, Response
from fastapi import HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import time


app = FastAPI()

MODEL_NAME = "large-v3"

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        print("推理耗时 %0.3f 秒，输入语音长度 %0.3f 秒，猜测语言 %s（概率 %0.3f)" % (time.time() - stime, info.duration, info.language, info.language_probability))
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

    return {
        "segments": segments,
        "info": info,
        "inference_time": time.time() - stime,
    }   



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

UPLOAD_DIR = "tmp"

import hmac
import hashlib

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    model: str = Form(...),
    file: UploadFile = File(...),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    settings_override: Optional[dict] = Form(None),
    authorization: Annotated[str , Header()] = "",
):
    assert model == "whisper-1"

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
            print("我方 KEY: `%s'" % KIWI_API_KEY)
            print("我方 sign: ", ourSign)
            print("对方 sign: ", theirSign)
            print("我方数据: `%s'" % tsStr)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized, 签名不正确"
            )
        

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

    filename = file.filename
    fileobj = file.file
    upload_name = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    with open(upload_name, "wb+") as upload_file:
        shutil.copyfileobj(fileobj, upload_file)

    whisper_args = WHISPER_DEFAULT_SETTINGS.copy()
    if settings_override is not None:
        whisper_args.update(settings_override)

    # 普通 whisper
    # transcript = transcribe(audio_path=upload_name, **whisper_args)
    # return transcript

    # faster_whisper
    result = faster_transcribe(audio_path=upload_name)
    resultJ = dumps(
        result,
        iensure_ascii = False,
        allow_nan = True,
        indent = None,
        separators = (',', ':'),
    ).encode("utf-8")

                         
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
