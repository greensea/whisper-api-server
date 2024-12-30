from main import get_faster_whisper_model
from main import CustomJSONEncoder, WHISPER_DEFAULT_SETTINGS
import time
from typing import Dict, List, Any
import json

if __name__ == "__main__":
    # model = get_faster_whisper_model("/home/gs/pj/whisper-api-server/whisper-v3-large.model")
    print("准备开始加载模型")
    start_time = time.time()
    # model = get_faster_whisper_model("/data/git/faster-whisper-large-v3", device="cpu")
    # print("模型加载完毕，耗时：", time.time() - start_time)
    # print("model = ", model)
    


class EndpointHandler():
    def __init__(self, path=""):
        print("传入的 path 参数是: ", path)
        print("准备开始加载模型")
        start_time = time.time()
#        self.model = get_faster_whisper_model("/data/git/faster-whisper-large-v3", device="cpu")
        print("模型加载完毕，耗时：", time.time() - start_time)


    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
       data args:
            inputs (:obj: `str`)
            date (:obj: `str`)
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """
        # # get inputs
        # inputs = data.pop("inputs",data)
        # date = data.pop("date", None)

        # check if date exists and if it is a holiday
        # if date is not None and date in self.holidays:
        #   return [{"label": "happy", "score": 1}]


        # # run normal prediction
        # prediction = self.pipeline(inputs)
        # return prediction

        print("传来的数据是", data)
        return data

    def transcribe(self, file, settings_override):

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
        result = self.faster_transcribe(audio_path=upload_name)
        stime = time.time()
        resultJ = json.dumps(
            result,
            iterable_as_array = True,
            ensure_ascii = False,
            ignore_nan = True,
            indent = None,
            separators = (',', ':'),
            cls = CustomJSONEncoder,
        ).encode("utf-8")
        # print("序列化耗时: %0.3f 秒" % (time.time() - stime))


                            
        return Response(resultJ, media_type="application/json")

    def faster_transcribe(self, audio_path):
        model = get_faster_whisper_model()
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

        # test_serialization(segments)

        return {
            "segments": segments,
            "info": info,
            "inference_time": time.time() - stime,
            "gpus": get_gpu_name(),
        }   


import os
import shutil
