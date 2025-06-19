import os
import torch
import torchaudio
import numpy as np
import pickle
from vad import apply_vad
from build_faiss import build_faiss_index
from speechbrain.pretrained import EncoderClassifier
from parameters import SAMPLE_RATE,EMBEDDINGS_PATH
import uuid
# Load pretrained ECAPA model


# Path to dataset


base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'model', 'spkrec-ecapa-voxceleb')
classifier = EncoderClassifier.from_hparams(source=model_dir)
DATASET_DIR = os.path.join(base_dir,'dataset')


def vad(signal,speaker,s):
    s.setText(f'Applying VAD...')
    signals = apply_vad(signal,SAMPLE_RATE)
    
    for signal in signals:
        filename = f"audio_{uuid.uuid4().hex}.wav"
        filepath = os.path.join(DATASET_DIR,speaker,filename)
        
        torchaudio.save(filepath,signal,SAMPLE_RATE)
    # if signal.shape[1] < 100:
    #     print(f"Skipping {filepath} — too short after VAD.")
    #     return
    # torchaudio.save(filepath, signal, SAMPLE_RATE)

def save_embeddings(name,status,extract_all=False):
    # Loop over each speaker
    speakers = os.listdir(DATASET_DIR) if extract_all else [name]
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as f:
            speaker_embeddings = pickle.load(f)
    else:
        speaker_embeddings = {}

    for speaker in speakers:
        status.setText(f'Extracting Embeddings for speaker {speaker}...')
        speaker_dir = os.path.join(DATASET_DIR, speaker)
        if not os.path.isdir(speaker_dir):
            continue

        speaker_embeddings[speaker] = []

        for filename in os.listdir(speaker_dir):
            if filename.endswith(".wav"):
                filepath = os.path.join(speaker_dir, filename)

                # Load audio
                signal, fs = torchaudio.load(filepath)
                # signal = signal.mean(dim=0).unsqueeze(0)  # Convert to mono

                # Resample to 16 kHz if needed
                if fs != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(fs, SAMPLE_RATE)
                    signal = resampler(signal)
                
                # signal = apply_vad(signal,fs)
                # if signal.shape[1] < 100:
                #     print(f"Skipping {filename} — too short after VAD.")
                #     continue
                # Extract embedding
                with torch.no_grad():
                    embedding = classifier.encode_batch(signal).squeeze().cpu().numpy()
                speaker_embeddings[speaker].append(embedding)

    

    # Save to disk
    if sum([len(x) for x in speaker_embeddings.values()]) > 0:
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(speaker_embeddings, f)
        status.setText(f'Building FAISS Index...')
        build_faiss_index(EMBEDDINGS_PATH)
        status.setText(f"Done...")

