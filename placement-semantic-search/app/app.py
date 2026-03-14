import streamlit as st
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Placement Preparation AI Assistant")

model = SentenceTransformer('all-MiniLM-L6-v2')

base_dir = os.path.dirname(os.path.dirname(__file__))

embeddings = np.load(os.path.join(base_dir,"embeddings","note_vectors.npy"))

with open(os.path.join(base_dir,"data","notes.txt"),"r",encoding="utf-8") as f:
    notes = f.readlines()

query = st.text_input("Ask a question")

if query:

    query_embedding = model.encode([query])

    scores = cosine_similarity(query_embedding, embeddings)[0]

    best_index = scores.argmax()

    st.write("Answer:")
    st.success(notes[best_index])