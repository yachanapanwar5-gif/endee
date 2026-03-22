# Simple demo of semantic search using embeddings

from sentence_transformers import SentenceTransformer
import numpy as np

# Sample data (leads)
data = [
    "Customer interested in buying laptop",
    "Looking for mobile phone deals",
    "Wants insurance plan",
    "Searching for CRM software"
]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert data to embeddings
embeddings = model.encode(data)

def search(query):
    query_embedding = model.encode([query])
    
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    index = np.argmax(similarities)
    
    return data[index]

# Test
print(search("customer wants to buy phone"))
