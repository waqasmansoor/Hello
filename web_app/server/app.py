from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import soundfile as sf
import scipy.signal as sps
import sys
import os
import threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from embeddings import save_embeddings,DATASET_DIR



app = Flask(__name__)
CORS(app)
os.makedirs(DATASET_DIR, exist_ok=True)
SAMPLE_RATE = 16000
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
        if len(self.status)>10:
            self.status=[]
        self.status.append(text)


s=status()
save_embeddings(None,s,True)
def read_audio(file,speaker):
    filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    
    filepath = os.path.join(DATASET_DIR,speaker,filename)
    rec,fsf=sf.read(file)
    number_of_samples=round(len(rec) * float(SAMPLE_RATE) / fsf)
    # #  Resampling to 16k
    data=sps.resample(rec,number_of_samples)
    sf.write(filepath,data,SAMPLE_RATE)
    thread = threading.Thread(target=save_embeddings,args=(speaker,s))
    thread.daemon = True
    thread.start()
    
    


@app.route('/', methods=['POST'])
def receive_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400

    file = request.files['audio']
    speaker = request.form.get("name")
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    os.makedirs(os.path.join(DATASET_DIR,speaker),exist_ok=True)
    read_audio(file,speaker)
    
    return jsonify({'message': 'File received'}), 200

@app.route("/status",methods=["GET"])
def get_status():
    return jsonify({"status":s.status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=False)
