import tkinter as tk
from tkinter import ttk
import threading
import os
import whisper_at as whisper
import sys
from pathlib import Path
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import time
import subprocess
import shutil
import pygame
from pyannote.audio import Pipeline

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  

if shutil.which('ffmpeg') is None:
    showinfo("Not found","FFmpeg is not found in the Path.")
    sys.exit(1)

LARGE_V2 = "large-v2"
LARGE_V1 = "large-v1"
MEDIUM  = "medium.en"
SMALL = "small.en"
BASE = "base.en"
TINY = "tiny.en"

MODEL_PATH = 'models'
SOURCE = 'audios'
model = None
pipeline = None

speaker_map = {
    "SPEAKER_00": "Josh",
    "SPEAKER_01": "Dyllan"
}

def load_model(path:str, message_box:callable):
    global model,pipeline
    if not os.path.exists(path):
        os.makedirs(path)
    message_box('Loading model...')
    pygame.mixer.init()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        # use_auth_token=""
        )

    model = whisper.load_model(TINY,download_root=path)
    # time.sleep(5)
    # model = ''
    message_box('Model Loaded Successfully!')

def message_box(msg:str):
    status_text.after(0, lambda: (status_text.insert(tk.END, msg + '\n'), status_text.see(tk.END)))

def text_box(msg:str):
    result_text.after(0, lambda: (result_text.insert(tk.END, msg + '\n'), result_text.see(tk.END)))

def start_thread(func:callable,arg:tuple):
    global worker_thread
    worker_thread = threading.Thread(target = func, args=arg,daemon = True)
    worker_thread.start()

def split_audio(filename:str, audio_tagging_time_resolution:int):
    message_box(f'Segmenting audio file: {Path(filename).name} into {audio_tagging_time_resolution} seconds intervals...')
    if not os.path.exists(SOURCE):
        os.makedirs(SOURCE)
    
    command = [
        'ffmpeg',
        '-i',filename,
        '-f','segment',
        '-segment_time',str(audio_tagging_time_resolution),
        '-c','copy',
        SOURCE + '/'+Path(filename).stem + '_%03d' + Path(filename).suffix
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # if process.stdout:
        #     for line in process.stdout:
        #         message_box(line.strip())
        process.wait()

        if process.returncode == 0:
            return True
        else: return False
            
            
    except subprocess.CalledProcessError as e:
        message_box(f'Error: {e}')
    
def transcribe_audio(name:str, text_box: callable,message_box:callable):
    message_box('Starting transcription...')
    for p in Path(SOURCE).glob(name):
        message_box(f'Audio file: {p.name}')
        audio = os.path.join(SOURCE,p.name)
        result = model.transcribe(audio)
        diarization = pipeline(audio)
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.load(os.path.join('audios',p.name))
        pygame.mixer.music.play()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Collect all text segments that overlap significantly with the speaker turn
            speaker_texts = []
            for segment in result["segments"]:
                seg_start = segment["start"]
                seg_end = segment["end"]

                # Check for overlap
                if seg_end > turn.start and seg_start < turn.end:
                    speaker_texts.append(segment["text"])

            if speaker_texts:
                combined_text = " ".join(speaker_texts)
                text_box(
                    f"start={turn.start:.1f}s stop={turn.end:.1f}s "
                    f"[{speaker_map.get(speaker,speaker)}]: {combined_text}"
                )
def wait_for_thread():
    if worker_thread.is_alive():
        root.after(100,wait_for_thread)
    else:
        message_box('Transcription completed')

def select_file():
    global model
    audio_tagging_time_resolution = 30
    filetypes = (
        ('All files', '*.*'),
        ('MP3', '*.mp3'),
        ('WAV', '*.wav'),
        ('OGG', '*.ogg'),
        ('FLAC', '*.flac')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir=ROOT,
        filetypes=filetypes)

    if filename and model is not None:
        
        if (split_audio(filename,audio_tagging_time_resolution)):
            message_box('Audio segmentation completed successfully!')
            name = Path(filename).stem + '_*' + Path(filename).suffix
            start_thread(transcribe_audio,(name,text_box,message_box))
            wait_for_thread()

        

root = tk.Tk()
root.title('Hello')
root.geometry("1000x600")
root.configure(bg="#f0f0f0")

# ===== Top Button =====
top_button = ttk.Button(root, text="Open a File", command=select_file)
top_button.pack(pady=20)

# ===== Frame to hold both text areas side by side =====
main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

# ===== Message Text Area (smaller) =====
message_frame = tk.LabelFrame(main_frame, text="Messages", labelanchor='n', font=("Helvetica", 11, "bold"))
message_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10), pady=10)

status_text = tk.Text(message_frame, wrap=tk.WORD, font=("Consolas", 10), relief=tk.FLAT, width=30)
status_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# ===== Result Text Area (larger) =====
result_frame = tk.LabelFrame(main_frame, text="Result", labelanchor='n', font=("Helvetica", 11, "bold"))
result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)

result_text = tk.Text(result_frame, wrap=tk.WORD, font=("Consolas", 10), relief=tk.FLAT)
result_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

root.after(100, start_thread(load_model, (MODEL_PATH,message_box)))
root.mainloop()



    