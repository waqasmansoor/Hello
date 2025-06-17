import numpy as np
import pickle
import faiss
import torch
from vad import apply_vad
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load classifier and FAISS



def predict_speaker(audio,samplerate,classifier):
    index = faiss.read_index("faiss.index")
    with open("labels.pkl", "rb") as f:
        all_labels = pickle.load(f)


    test_tensor = torch.tensor(audio.squeeze(), dtype=torch.float32).unsqueeze(0)

    # === VAD ===
    test_tensor = apply_vad(test_tensor, samplerate)
    if test_tensor.shape[1] < 100:
        print("⚠️ No speech detected.")
        exit()

    # === Embedding ===
    with torch.no_grad():
        test_embedding = classifier.encode_batch(test_tensor).squeeze().cpu().numpy()

    # Normalize for cosine similarity
    test_embedding = test_embedding / np.linalg.norm(test_embedding)

    # === Search FAISS ===
    D, I = index.search(np.array([test_embedding]).astype('float32'), k=1)
    top_index = I[0][0]
    score = D[0][0]
    predicted_speaker = all_labels[top_index]
    return predicted_speaker,score
    # === Output ===
    print(f"🔍 Predicted speaker: {predicted_speaker}")
    print(f"📊 Similarity score: {score:.4f}")

    # Optional: Reject if below confidence threshold
    if score < 0.75:
        print("⚠️ Speaker unknown or unclear")
