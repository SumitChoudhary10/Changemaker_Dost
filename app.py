import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")
genai.configure(api_key=GOOGLE_API_KEY)

EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATIVE_MODEL_NAME = "gemini-1.5-flash-latest"

class QuestionRequest(BaseModel):
    question: str

app = FastAPI()

# print("Loading FAISS index and text chunks...")
try:
    index = faiss.read_index("ashoka_index.faiss")
    with open("text_chunks.pkl", "rb") as f:
        text_chunks = pickle.load(f)
    print("Data loaded successfully. Backend is ready.")
except Exception as e:
    print(f"An error occurred during startup: {e}")

def search(query, k=4):
    query_embedding_dict = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,
        content=query,
        task_type="RETRIEVAL_QUERY"
    )
    query_embedding = np.array([query_embedding_dict['embedding']]).astype('float32')
    D, I = index.search(query_embedding, k)
    return [text_chunks[i] for i in I[0]]

def generate_full_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the context does not contain the answer, state that you couldn't find the information.\n\nContext:\n{context}\n\nUser's Question:\n{query}\n\nAnswer:"
    
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text

@app.post("/dialogflow-webhook")
async def dialogflow_webhook(request: dict):
    try:
        user_question = request['queryResult']['queryText']

        relevant_chunks = search(user_question)
        final_answer = generate_full_answer(user_question, relevant_chunks)

        response_data = {
            "fulfillmentText": final_answer
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Error in Dialogflow webhook: {e}")
        error_response = {
            "fulfillmentText": "Sorry, I encountered an error while trying to find an answer. Please try again."
        }
        return JSONResponse(content=error_response)


@app.get("/")
async def read_root():
    return {"message": "Ashoka Bot Backend is running!"}
