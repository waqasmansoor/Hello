import whisper
import time
import torchaudio
import gc
import torch
import numpy as np
import subprocess
from whisperx.alignment import align, load_align_model
from whisperx.audio import load_audio
import pickle

from whisperx.diarize import DiarizationPipeline,assign_word_speakers

SAMPLE_RATE = 16000

def transcribe(audio_paths):
    results = []
    model = whisper.load_model("medium.en")
    start = time.time()
    for audio_path in audio_paths:
        audio = load_audio(audio_path)
        result = model.transcribe(audio,verbose = True)
        results.append((result,audio_path))
    print(f"Time to transcribe: {time.time()-start}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results
    

            

audio = "audios/tel1_000.wav"
language_code = "en"
model_dir = "models"
device = "cpu"
print_progress = True

align_model, align_metadata = load_align_model(language_code,device, model_dir = model_dir)
results = transcribe([audio])
with open("transcribe.pkl",'wb') as f:
    pickle.dump(results,f)
# tmp_results = results
# results = []
# for result,audio_path in tmp_results:
#     if len(result) > 1:
#         input_audio = audio_path
#     else:
#         input_audio = audio

#     if len(result["segments"]) > 0:
#         print(">>Performing alignment...")
#         result = align(result["segments"],align_model,align_metadata,input_audio,device,print_progress = True)
#     results.append((result,input_audio))

# with open("align.pkl",'wb') as f:
#     pickle.dump(results,f)

# del align_model
# gc.collect()
# torch.cuda.empty_cache()

tmp_results = results
results = []
print(">>Performing diarization...")
diarize_model = DiarizationPipeline(use_auth_token="HF_TOKEN", device=device)
for result, input_audio_path in tmp_results:
    print("........",input_audio_path)
    diarize_segments = diarize_model(input_audio_path, min_speakers=2, max_speakers=2)
    result = assign_word_speakers(diarize_segments, result)
    results.append((result, input_audio_path))

with open("diarize.pkl",'wb') as f:
    pickle.dump(results,f)
for result, audio_path in results:
    result["language"] = "en"
    print(result)
