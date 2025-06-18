import os
import torch
import torchaudio
import numpy as np
import pickle
from vad import apply_vad
from build_faiss import build_faiss_index
from speechbrain.pretrained import EncoderClassifier

# Load pretrained ECAPA model


# Path to dataset
EMBEDDINGS_PATH = "embeddings.pkl"

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'model', 'spkrec-ecapa-voxceleb')
classifier = EncoderClassifier.from_hparams(source=model_dir)
DATASET_DIR = os.path.join(base_dir,'dataset')


def save_embeddings(name,status,extract_all=False):
    # Loop over each speaker
    speakers = os.listdir(DATASET_DIR) if extract_all else [name]
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as f:
            speaker_embeddings = pickle.load(f)
    else:
        speaker_embeddings = {}

    for speaker in speakers:
        print(speaker)
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
                signal = signal.mean(dim=0).unsqueeze(0)  # Convert to mono

                # Resample to 16 kHz if needed
                if fs != 16000:
                    resampler = torchaudio.transforms.Resample(fs, 16000)
                    signal = resampler(signal)
                status.setText(f'Applying VAD...')
                signal = apply_vad(signal,fs)
                if signal.shape[1] < 100:
                    print(f"Skipping {filename} — too short after VAD.")
                    continue
                # Extract embedding
                with torch.no_grad():
                    embedding = classifier.encode_batch(signal).squeeze().cpu().numpy()
                speaker_embeddings[speaker].append(embedding)

    

    # Save to disk
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(speaker_embeddings, f)
    status.setText(f'Building FAISS Index...')
    build_faiss_index(EMBEDDINGS_PATH)
    status.setText(f"Done...")

