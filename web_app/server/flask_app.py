from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import soundfile as sf
import sys
import os
import threading
import queue
import time
import io
import torchaudio
import torchaudio.transforms as T
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from embeddings import save_embeddings,vad,DATASET_DIR
from parameters import SAMPLE_RATE


app = Flask(__name__)
CORS(app)
thread_queue=queue.Queue()
os.makedirs(DATASET_DIR, exist_ok=True)



if os.path.exists('embeddings.pkl'):
    os.remove('embeddings.pkl')
if os.path.exists('faiss.index'):
    os.remove('faiss.index')
if os.path.exists('labels.pkl'):
    os.remove('labels.pkl')


class status:
    def __init__(self):
        self.status = []
    def setText(self,text):
        if len(self.status)>100:
            self.status=[]
        self.status.append(text)


s=status()
# save_embeddings(None,s,True)

def get_new_thread():
    while True:
        new_thread = thread_queue.get()
        if new_thread is None:
            break
        waveform,sr,speaker = new_thread
        read_audio(waveform,sr,speaker)
        thread_queue.task_done()

def read_audio(waveform,sr,speaker):
    
    if sr != SAMPLE_RATE:
    # # #  Resampling to 16k
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    # sf.write(filepath,data,SAMPLE_RATE)
    start=time.time()
    vad(waveform,speaker,s)
    # save_embeddings(speaker,s)
    s.setText(f"Time: {time.time()-start}")
    
main_thread = threading.Thread(target=get_new_thread,daemon=True)
main_thread.start()
    

@app.route('/test', methods=['POST'])
def get_test_data():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400
    file = request.files['audio']
    # print(file.filename,file.file)
    return jsonify({'message': 'File received'}), 200
    
@app.route('/train', methods=['POST'])
def get_train_data():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400

    file = request.files['audio']
    speaker = request.form.get("name")
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    os.makedirs(os.path.join(DATASET_DIR,speaker),exist_ok=True)
    
    buffer = io.BytesIO(file.read())
    buffer.seek(0)
    waveform, sample_rate = torchaudio.load(buffer)
    signal = waveform.mean(dim=0).unsqueeze(0)  # Convert to mono
    # rec,fsf=sf.read(file)
    thread_queue.put((signal,sample_rate,speaker))
    
    
    return jsonify({'message': 'File received'}), 200

@app.route("/status",methods=["GET"])
def get_status():
    return jsonify({"status":s.status})

#For Testing (Development Environment)
import atexit
@atexit.register
def cleanup():
    thread_queue.put(None) 
    main_thread.join()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
