import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class QuestionRequest(BaseModel):
    question: str

app = FastAPI()

print("Loading models and data...")
try:
    index = faiss.read_index("ashoka_index.faiss")
    with open("text_chunks.pkl", "rb") as f:
        text_chunks = pickle.load(f)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file!")
    genai.configure(api_key=api_key)
    generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Models and data loaded successfully.")
except Exception as e:
    print(f"An error occurred during startup: {e}")

def search(query, k=4):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(query_embedding, k)
    return [text_chunks[i] for i in I[0]]

def generate_answer_stream(query, context_chunks):
    
    context = "\n\n".join(context_chunks)
    prompt = f"""
    You are a helpful assistant. Answer the user's question based ONLY on the provided context.
    If the context does not contain the answer, state that you couldn't find the information.
    Context: {context}
    User's Question: {query}
    Answer:
    """
    try:
        response_stream = generative_model.generate_content(prompt, stream=True)
        for chunk in response_stream:
            yield chunk.text
    except Exception as e:
        print(f"Error during streaming: {e}")
        yield "Sorry, an error occurred while generating the answer."

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    user_question = request.question
    relevant_chunks = search(user_question)
    return StreamingResponse(
        generate_answer_stream(user_question, relevant_chunks),
        media_type="text/plain"
    )

@app.get("/")
async def read_root():
    return {"message": "Ashoka Bot Backend is running!"}