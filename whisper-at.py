
import whisper_at as whisper
import wget
import os

LARGE_V2 = "large-v1"
LARGE_V1 = "large-v1"
MEDIUM  = "medium.en"
SMALL = "small.en"
BASE = "base.en"
TINY = "tiny.en"


if __name__ == "__main__":
    url = 'https://www.dropbox.com/s/7eznyazmc1pmw9h/case_closed.wav?dl=1'
    destination = 'audios/case_closed.wav'  # Conserve l'extension correcte
    model_path = 'models'

    if not os.path.exists(destination):
        wget.download(url, destination)

    audio_tagging_time_resolution = 10
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model = whisper.load_model(MEDIUM,download_root=model_path)
    print("Model loaded")
    result = model.transcribe(destination, at_time_res=audio_tagging_time_resolution)

    for segment in result["segments"]:
        print(segment['start'], 's-', segment['end'], 's', segment['text'])

