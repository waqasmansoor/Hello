
import whisper_at as whisper
import wget
import os
from pathlib import Path


LARGE_V2 = "large-v1"
LARGE_V1 = "large-v1"
MEDIUM  = "medium.en"
SMALL = "small.en"
BASE = "base.en"
TINY = "tiny.en"

def download_sample_audio(destination:str):
    url = 'https://www.dropbox.com/s/7eznyazmc1pmw9h/case_closed.wav?dl=1'
    if not os.path.exists(destination):
        wget.download(url, destination)
    return destination

if __name__ == "__main__":

    destination = 'audios'
    # for audio in destination:
    #     pass

    audio_path = download_sample_audio(destination + os.sep + 'case_closed.wav')
    
    model_path = 'models'
    audio_tagging_time_resolution = 10
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model = whisper.load_model(MEDIUM,download_root=model_path)
    print("Model loaded")
    result = model.transcribe(audio_path, at_time_res=audio_tagging_time_resolution)

    for segment in result["segments"]:
        print(segment['start'], 's-', segment['end'], 's', segment['text'])

