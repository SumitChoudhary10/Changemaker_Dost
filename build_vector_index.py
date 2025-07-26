import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

print("Running...")
# --- This is the key change ---
# We specify the correct path to the text file inside the 'data' folder.
CLEANED_TEXT_PATH = os.path.join("data", "ashoka_cleaned.txt")

# print(f"Loading the cleaned text from '{CLEANED_TEXT_PATH}'...")
with open(CLEANED_TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Split the text by double newlines (paragraphs) and filter out short/empty ones.
text_chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 100]
# print(f"Text split into {len(text_chunks)} chunks.")


# print("Loading the embedding model...")
model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')

print("Creating embeddings for text chunks... (This may take a moment)")
embeddings = model.encode(text_chunks, show_progress_bar=True)
print(f"Embeddings created with shape: {embeddings.shape}")

# Build the FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
# print(f"Total vectors in the index: {index.ntotal}")

# --- We will save the new files in the main project folder ---
INDEX_SAVE_PATH = "ashoka_index.faiss"
CHUNKS_SAVE_PATH = "text_chunks.pkl"

print(f"Saving the FAISS index to '{INDEX_SAVE_PATH}'")
faiss.write_index(index, INDEX_SAVE_PATH)

print(f"Saving the text chunks to '{CHUNKS_SAVE_PATH}'")
with open(CHUNKS_SAVE_PATH, "wb") as f:
    pickle.dump(text_chunks, f)

print("\n--- Setup Complete! ---")
print(f"Your vector index ('{INDEX_SAVE_PATH}') and text chunks ('{CHUNKS_SAVE_PATH}') are ready.")