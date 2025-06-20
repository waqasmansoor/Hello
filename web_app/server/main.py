from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys
import threading
import queue
import time
import io
import torchaudio
import torchaudio.transforms as T

# Add upper directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from embeddings import save_embeddings, vad, DATASET_DIR
from parameters import SAMPLE_RATE

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

thread_queue = queue.Queue()
os.makedirs(DATASET_DIR, exist_ok=True)

# Clean up previous embeddings
# for f in ['embeddings.pkl', 'faiss.index', 'labels.pkl']:
#     if os.path.exists(f):
#         os.remove(f)

# Status tracker class
class Status:
    def __init__(self):
        self.status = []
    def setText(self, text):
        if len(self.status) > 100:
            self.status = []
        self.status.append(text)

s = Status()
#Build New Embeddings from Scratch
# save_embeddings(None,s,True)

def get_new_thread():
    while True:
        new_thread = thread_queue.get()
        if new_thread is None:
            break
        waveform, sr, speaker = new_thread
        read_audio(waveform, sr, speaker)
        thread_queue.task_done()

def read_audio(waveform, sr, speaker):
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    start = time.time()
    vad(waveform, speaker, s)
    save_embeddings(speaker, s)
    s.setText(f"Time: {time.time() - start}")

# Start background thread
main_thread = threading.Thread(target=get_new_thread, daemon=True)
main_thread.start()

@app.post("/test")
async def get_test_data(audio: UploadFile = File(...)):
    pass

@app.post("/train")
async def get_train_data(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    if not file.filename:
        return JSONResponse(content={"error": "No selected file"}, status_code=400)

    speaker_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(speaker_dir, exist_ok=True)

    # Read audio bytes
    buffer = io.BytesIO(await file.read())
    buffer.seek(0)
    waveform, sample_rate = torchaudio.load(buffer)

    # Convert to mono
    signal = waveform.mean(dim=0).unsqueeze(0)

    # Put task in thread queue
    thread_queue.put((signal, sample_rate, name))

    return JSONResponse(content={"message": "File received"})

@app.get("/status")
async def get_status():
    return {"status": s.status}

# Cleanup function for thread on exit
import atexit
@atexit.register
def cleanup():
    thread_queue.put(None)
    main_thread.join()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)