
import whisper_at as whisper
from whisper_at.audio import load_audio
import wget
import os
from pathlib import Path
from flask import Flask,render_template,request,jsonify
import threading

app = Flask(__name__)

os.environ['PATH'] += os.pathsep + r"ffmpeg-7.1.1-essentials_build\bin"

LARGE_V2 = "large-v1"
LARGE_V1 = "large-v1"
MEDIUM  = "medium.en"
SMALL = "small.en"
BASE = "base.en"
TINY = "tiny.en"

MODEL_PATH = 'models'
SOURCE = 'audios'
model = None

def download_sample_audio(destination:str):
    url = 'https://www.dropbox.com/s/7eznyazmc1pmw9h/case_closed.wav?dl=1'
    if not os.path.exists(destination):
        wget.download(url, destination)
    return destination

def load_model(path:str):
    global model
    if not os.path.exists(path):
        os.makedirs(path)
    model = whisper.load_model(MEDIUM,download_root=path)

@app.route('/get_status',methods = ['GET'])
def get_status():
    if request.method == 'GET':
        if model:
            return jsonify({'loaded':True})
        else:
            return jsonify({'loaded':False})
    
@app.route('/get_transcribtion',methods = ['POST'])
def get_transcribtion():
    if request.method == 'POST':
        data = request.get_json()
        filename = data.get('filename')
        if filename:
            print('Audio file received',os.path.join(SOURCE,filename))
            audio_tagging_time_resolution = 10
            # audio = load_audio(os.path.join(SOURCE,filename))
            audio = os.path.join(SOURCE,filename)
            result = model.transcribe(audio, at_time_res=audio_tagging_time_resolution)
            return jsonify({'transcription':result})
        

@app.route('/',methods = ['GET','POST'])
def index():
    if request.method == 'GET':
        
        threading.Thread(target = load_model, args=(MODEL_PATH,)).start()
        
        return render_template('index.html')
    
    


if __name__ == "__main__":

    app.run(debug = True)

    
    # # for audio in destination:
    # #     pass

    # audio_path = download_sample_audio(destination + os.sep + 'case_closed.wav')
    
    
    
    
    
    # model = whisper.load_model(MEDIUM,download_root=model_path)
    # print("Model loaded")
    

    # for segment in result["segments"]:
    #     print(segment['start'], 's-', segment['end'], 's', segment['text'])

