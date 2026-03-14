from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Build correct path to notes file
base_dir = os.path.dirname(os.path.dirname(__file__))
notes_path = os.path.join(base_dir, "data", "notes.txt")

# Read notes
with open(notes_path, "r", encoding="utf-8") as file:
    notes = file.readlines()

# Create embeddings
embeddings = model.encode(notes)

# Save embeddings
np.save(os.path.join(os.path.dirname(__file__), "note_vectors.npy"), embeddings)

print("Embeddings created successfully!")