import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")
genai.configure(api_key=GOOGLE_API_KEY)

# Define model names
EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATIVE_MODEL_NAME = "gemini-1.5-flash-latest"

# Pydantic model for request validation
class QuestionRequest(BaseModel):
    question: str

# Initialize the FastAPI application
app = FastAPI()

# --- Load data at startup (No heavy AI models!) ---
print("Loading FAISS index and text chunks...")
try:
    index = faiss.read_index("ashoka_index.faiss")
    with open("text_chunks.pkl", "rb") as f:
        text_chunks = pickle.load(f)
    print("Data loaded successfully. Backend is ready.")
except Exception as e:
    print(f"An error occurred during startup: {e}")

# --- Define the Functions ---
def search(query, k=4):
    """Finds the most relevant text chunks for a given query."""
    query_embedding_dict = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,
        content=query,
        task_type="RETRIEVAL_QUERY"
    )
    query_embedding = np.array([query_embedding_dict['embedding']]).astype('float32')
    
    D, I = index.search(query_embedding, k)
    return [text_chunks[i] for i in I[0]]

def generate_answer_stream(query, context_chunks):
    """Generates a streamed answer from the Google AI model with a smarter prompt."""
    context = "\n\n".join(context_chunks)
    
    # --- THIS IS THE NEW, SMARTER PROMPT ---
    prompt = f"""
    You are 'Changemaker Dost', a friendly and knowledgeable AI assistant about Ashoka.
    Your goal is to answer the user's question. Follow these steps:
    1.  First, carefully read the provided "Official Context" from the Ashoka website.
    2.  If the answer is fully contained within the Official Context, use it to write a clear and natural-sounding answer.
    3.  If the Official Context does not contain the answer, then use your own general knowledge to answer the question about Ashoka.
    4.  Never say phrases like "Based on the provided text" or "The context does not contain...". Just provide the best possible answer in a conversational way.

    Official Context:
    ---
    {context}
    ---

    User's Question:
    {query}

    Answer:
    """
    
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response_stream = model.generate_content(prompt, stream=True)
    for chunk in response_stream:
        yield chunk.text

# --- API Endpoints ---
@app.post("/dialogflow-webhook")
async def dialogflow_webhook(request: dict):
    """Handles requests from Dialogflow."""
    try:
        user_question = request['queryResult']['queryText']
        relevant_chunks = search(user_question)
        
        # We need a non-streamed version for Dialogflow
        full_answer = ""
        model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        prompt = f"""
        You are 'Changemaker Dost', a friendly and knowledgeable AI assistant about Ashoka.
        Your goal is to answer the user's question. Follow these steps:
        1.  First, carefully read the provided "Official Context" from the Ashoka website.
        2.  If the answer is fully contained within the Official Context, use it to write a clear and natural-sounding answer.
        3.  If the Official Context does not contain the answer, then use your own general knowledge to answer the question about Ashoka.
        4.  Never say phrases like "Based on the provided text" or "The context does not contain...". Just provide the best possible answer in a conversational way.

        Official Context:
        ---
        {"\n\n".join(relevant_chunks)}
        ---

        User's Question:
        {user_question}

        Answer:
        """
        response = model.generate_content(prompt)
        full_answer = response.text

        response_data = {"fulfillmentText": full_answer}
        return response_data

    except Exception as e:
        print(f"Error in Dialogflow webhook: {e}")
        error_response = {"fulfillmentText": "Sorry, I encountered an error. Please try again."}
        return error_response

@app.get("/")
async def read_root():
    """A simple health check endpoint."""
    return {"message": "Ashoka Bot Backend is running!"}
