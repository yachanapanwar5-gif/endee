from sentence_transformers import SentenceTransformer
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load embeddings
base_dir = os.path.dirname(os.path.dirname(__file__))
embeddings_path = os.path.join(base_dir, "embeddings", "note_vectors.npy")
embeddings = np.load(embeddings_path)

# Load notes
notes_path = os.path.join(base_dir, "data", "notes.txt")
with open(notes_path, "r", encoding="utf-8") as f:
    notes = f.readlines()

# Ask user query
query = input("Ask your question: ")

# Convert query to embedding
query_embedding = model.encode([query])

# Compute similarity
scores = cosine_similarity(query_embedding, embeddings)[0]

# Find best match
best_index = np.argmax(scores)

print("\nMost relevant answer:\n")
print(notes[best_index])