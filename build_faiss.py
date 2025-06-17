import numpy as np
import pickle
import faiss

# Load your saved embeddings

def build_faiss_index(file):
    with open(file, "rb") as f:
        speaker_embeddings = pickle.load(f)

    all_vectors = []
    all_labels = []

    for speaker, vectors in speaker_embeddings.items():
        for vec in vectors:
            vec = vec / np.linalg.norm(vec)  # normalize for cosine similarity
            all_vectors.append(vec)
            all_labels.append(speaker)

    # Convert to FAISS-compatible format
    all_vectors = np.array(all_vectors).astype('float32')

    # Build index
    index = faiss.IndexFlatIP(all_vectors.shape[1])  # inner product for cosine similarity
    index.add(all_vectors)

    # Save index and label mapping
    faiss.write_index(index, "faiss.index")
    with open("labels.pkl", "wb") as f:
        pickle.dump(all_labels, f)

    print(f"✅ FAISS index built with {len(all_vectors)} vectors.")
