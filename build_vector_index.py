import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
import spacy
import google.generativeai as genai
import time

print("working...")
load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")
genai.configure(api_key=GOOGLE_API_KEY)
EMBEDDING_MODEL_NAME = "text-embedding-004"


def get_embedding_from_google(text_payload):
    """Calls the Google AI API to get an embedding for a text chunk."""
    embedding = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,
        content=text_payload,
        task_type="RETRIEVAL_DOCUMENT" 
    )
    return embedding['embedding']

def create_smart_chunks(text, nlp_model, sentences_per_chunk=7):
    doc = nlp_model(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if len(chunk.strip()) > 50: 
            chunks.append(chunk)
    return chunks

# print("Running build_vector_index.py using Google's Embedding API...")

# print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

CLEANED_TEXT_PATH = os.path.join("data", "ashoka_cleaned.txt")
with open(CLEANED_TEXT_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

text_chunks = create_smart_chunks(raw_text, nlp)
# print(f"Text successfully split into {len(text_chunks)} chunks.")

if not text_chunks:
    print("No chunks were created. Please check the input text.")
    exit()

print("Creating embeddings... (This will take a moment)")
all_embeddings = []
for i, chunk in enumerate(text_chunks):
    # print(f"Processing chunk {i+1}/{len(text_chunks)}...")
    try:
        embedding = get_embedding_from_google(chunk)
        all_embeddings.append(embedding)
        time.sleep(1) 
    except Exception as e:
        print(f"Could not process chunk {i+1}. Error: {e}")
        break 

if len(all_embeddings) != len(text_chunks):
    print("Embedding generation failed. Please check the error messages.")
    exit()

embeddings_np = np.array(all_embeddings).astype('float32')
print(f"Embeddings created with final shape: {embeddings_np.shape}")

d = embeddings_np.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings_np)
# print(f"Total vectors in the final index: {index.ntotal}")

faiss.write_index(index, "ashoka_index.faiss")
with open("text_chunks.pkl", "wb") as f:
    pickle.dump(text_chunks, f)

print("\n-- Index built successfully --")
