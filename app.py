import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse 
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import random
import json

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

# --- THE CMI QUESTION BANK ---
# This is a large, expandable library of questions for the assessment.
QUESTION_BANK = {
    "empathy": [
        "Describe a time you had to work with someone who was very different from you. How did you try to understand their perspective?",
        "Think about a problem in your community or college. Who is most affected by this problem, and what do you think their daily experience is like?",
        "Tell me about a time you changed your mind about something important after listening to someone else's point of view.",
        "When you see someone struggling, what is your first instinct? Describe a time you acted on that instinct.",
        "How do you ensure that everyone in a group feels heard, especially the quietest person?"
    ],
    "teamwork": [
        "Tell me about a group project that was challenging. What role did you play, and how did you help the team move forward together?",
        "Imagine you and a friend have a great idea, but you disagree on the first step. How would you handle this to ensure you can still work together?",
        "Describe a time you had to rely on a teammate to get something done. How did you build that trust?",
        "What, in your opinion, is the most important quality of a good team member? Give an example of when you demonstrated this quality.",
        "How do you handle a situation where a team member is not contributing their fair share?"
    ],
    "leadership": [
        "Describe a situation where you saw something that was unfair or inefficient. What did you do about it, even if it was a small action?",
        "If you were given the resources to improve one thing in your college, what would it be and what would be your very first step?",
        "Leadership isn't always about being in charge. Tell me about a time you led by example.",
        "Think of a leader you admire. What qualities do they have that you would like to develop in yourself?",
        "What is a change you would like to see in your community, and how would you inspire others to join you in making that change?"
    ],
    "initiative": [
        "Tell me about an idea you had that you were excited about. What steps did you take to bring it to life, even if it wasn't perfect?",
        "Imagine you've been working on a project and it fails completely. What is your process for learning from that failure and deciding what to do next?",
        "Describe a time you taught yourself a new skill to solve a problem or complete a project.",
        "When you have a big assignment, do you prefer to start right away or wait? Explain your reasoning.",
        "What is something you have started on your own, without anyone asking you to? It could be a club, a small project, or even a new habit."
    ]
}

# --- Pydantic Models for Request Validation ---
class QuestionRequest(BaseModel):
    question: str

# --- Initialize the FastAPI App ---
app = FastAPI()

# --- Load Ashoka Q&A Data at Startup ---
print("Loading FAISS index and text chunks for Ashoka Q&A...")
try:
    index = faiss.read_index("ashoka_index.faiss")
    with open("text_chunks.pkl", "rb") as f:
        text_chunks = pickle.load(f)
    print("Ashoka Q&A data loaded successfully.")
except Exception as e:
    print(f"An error occurred during startup loading Ashoka data: {e}")

# --- Core Functions (Ashoka Q&A and CMI) ---

def search(query, k=3):
    """Finds the most relevant text chunks for a given query."""
    query_embedding_dict = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = np.array([query_embedding_dict['embedding']]).astype('float32')
    D, I = index.search(query_embedding, k)
    return [text_chunks[i] for i in I[0]]

def generate_full_answer(query, context_chunks):
    """Generates a complete answer for the Ashoka Q&A webhook."""
    context = "\n\n".join(context_chunks)
    prompt = f"""
    You are 'Changemaker Dost', a friendly and knowledgeable AI assistant about Ashoka.
    Your goal is to answer the user's question. Follow these steps:
    1. First, carefully read the provided "Official Context" from the Ashoka website.
    2. If the answer is fully contained within the Official Context, use it to write a clear and natural-sounding answer.
    3. If the Official Context does not contain the answer, then use your own general knowledge to answer the question about Ashoka.
    4. Never say phrases like "Based on the provided text". Just provide the best possible answer in a conversational way.

    Official Context: --- {context} ---
    User's Question: {query}
    Answer:
    """
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text

def analyze_cmi_response(category, question, answer):
    """Uses the AI to analyze the user's CMI answer and return a score."""
    prompt = f"""
    You are an expert in evaluating changemaking skills based on the CMI framework (Empathy, Teamwork, Leadership, Initiative).
    The user was asked the following question to assess their '{category}':
    ---
    Question: "{question}"
    ---
    The user responded with the following answer:
    ---
    Answer: "{answer}"
    ---
    Analyze their response. Provide a score from 1 to 10 for their demonstrated '{category}' skill. Also provide a brief, one-sentence justification for your score.
    Respond ONLY with a valid JSON object in the format: {{"score": <score_number>, "justification": "<one_sentence_justification>"}}
    """
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    try:
        # Clean up the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error decoding AI analysis response: {e}")
        return {"score": 0, "justification": "Could not analyze the response."}

# --- API Endpoints ---

@app.post("/dialogflow-webhook")
async def dialogflow_webhook(request: dict):
    """
    This single webhook now handles ALL requests from Dialogflow
    and routes them based on the incoming intent.
    """
    try:
        intent_name = request['queryResult']['intent']['displayName']
        print(f"Received intent: {intent_name}")

        # --- Router for Ashoka Q&A ---
        if "ask_about_ashoka" in intent_name:
            user_question = request['queryResult']['queryText']
            relevant_chunks = search(user_question)
            final_answer = generate_full_answer(user_question, relevant_chunks)
            response_data = {"fulfillmentText": final_answer}
            return JSONResponse(content=response_data)

        # --- Router for CMI Assessment ---
        elif intent_name == "cmi_assessment_START":
            # Start of the assessment. Pick the first question.
            question = random.choice(QUESTION_BANK["empathy"])
            response_data = {
                "fulfillmentText": f"Great! Let's start with Empathy. Here is your first question:\n\n{question}",
                "outputContexts": [
                    {
                        "name": f"{request['session']}/contexts/awaiting_cmi_empathy_response",
                        "lifespanCount": 1,
                        "parameters": {"current_question": question}
                    }
                ]
            }
            return JSONResponse(content=response_data)

        elif intent_name == "cmi_assessment_empathy_RESPONSE":
            # User answered the empathy question. Analyze it and ask the next one.
            user_answer = request['queryResult']['queryText']
            # Get the question that was asked from the input context
            contexts = request['queryResult']['outputContexts']
            current_question = ""
            for context in contexts:
                if "awaiting_cmi_empathy_response" in context['name'] and 'parameters' in context:
                    current_question = context['parameters'].get('current_question', "")
            
            # Analyze the answer (we can store this later)
            analysis = analyze_cmi_response("Empathy", current_question, user_answer)
            print(f"Empathy Analysis: {analysis}")

            # Ask the next question for Teamwork
            question = random.choice(QUESTION_BANK["teamwork"])
            response_data = {
                "fulfillmentText": f"Thank you for sharing. Now, let's think about Teamwork:\n\n{question}",
                "outputContexts": [
                    {
                        "name": f"{request['session']}/contexts/awaiting_cmi_teamwork_response",
                        "lifespanCount": 1,
                        "parameters": {"current_question": question}
                    }
                ]
            }
            return JSONResponse(content=response_data)

        # ... (We will add the other CMI response intents later) ...

        else:
            # Fallback for any other intent
            return JSONResponse(content={"fulfillmentText": "I'm not sure how to handle that."})

    except Exception as e:
        print(f"Error in main webhook: {e}")
        return JSONResponse(content={"fulfillmentText": "Sorry, an error occurred on my end."})


@app.get("/")
async def read_root():
    """A simple health check endpoint."""
    return {"message": "Ashoka Bot Backend is running!"}
