import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import random
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

load_dotenv()

# --- Firebase Initialization (Final, Most Robust Method) ---
print("Initializing Firebase...")
try:
    # Load the service account key file directly into a dictionary
    with open("serviceAccountKey.json") as f:
        service_account_info = json.load(f)
    
    # Pass the dictionary directly to the credentials object
    cred = credentials.Certificate(service_account_info)
    
    # Check if the app is already initialized to prevent errors during hot-reloads
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        
    db = firestore.client()
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")


# --- Google AI Configuration ---
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")
genai.configure(api_key=GOOGLE_API_KEY)

# Define model names
EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATIVE_MODEL_NAME = "gemini-1.5-flash-latest"

# --- THE CMI QUESTION BANK ---
QUESTION_BANK = {
    "empathy": ["Describe a time you had to work with someone very different from you. How did you try to understand their perspective?", "Think about a problem in your community. Who is most affected, and what do you think their daily experience is like?", "Tell me about a time you changed your mind after listening to someone else's point of view."],
    "teamwork": ["Tell me about a challenging group project. What role did you play, and how did you help the team move forward?", "Imagine you and a friend disagree on the first step of a great idea. How would you handle this?", "Describe a time you had to rely on a teammate. How did you build that trust?"],
    "leadership": ["Describe a situation where you saw something unfair. What did you do about it, even if it was a small action?", "If you could improve one thing in your college, what would it be and what would be your first step?", "Tell me about a time you led by example."],
    "initiative": ["Tell me about an idea you were excited about. What steps did you take to bring it to life?", "Imagine a project you're working on fails completely. How do you learn from it and decide what to do next?", "Describe a time you taught yourself a new skill to solve a problem."]
}

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


# --- Core Functions ---
def search(query, k=3):
    """Finds the most relevant text chunks for a given query."""
    query_embedding_dict = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = np.array([query_embedding_dict['embedding']]).astype('float32')
    D, I = index.search(query_embedding, k)
    return [text_chunks[i] for i in I[0]]

def generate_full_answer(query, context_chunks):
    """Generates a complete answer for the Ashoka Q&A webhook."""
    context = "\n\n".join(context_chunks)
    prompt = f"You are 'Changemaker Dost', a friendly AI assistant. Answer the user's question about Ashoka. Prioritize information from the Official Context. If the answer isn't there, use your general knowledge. Never say 'Based on the provided text'.\n\nOfficial Context: --- {context} ---\nUser's Question: {query}\nAnswer:"
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text

def analyze_cmi_response(category, question, answer):
    """Uses the AI to analyze the user's CMI answer and return a score."""
    prompt = f"You are an expert in evaluating changemaking skills (Empathy, Teamwork, Leadership, Initiative). The user was asked a question to assess their '{category}':\nQuestion: \"{question}\"\nThe user responded:\nAnswer: \"{answer}\"\nAnalyze their response. Provide a score from 1 to 10 for their '{category}' skill and a one-sentence justification. Respond ONLY with a valid JSON object like: {{\"score\": <score_number>, \"justification\": \"<one_sentence_justification>\"}}"
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    try:
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error decoding AI analysis: {e}")
        return {"score": 0, "justification": "Could not analyze response."}

def save_assessment_to_firestore(session_id, results):
    """Saves the assessment and returns the new document's ID."""
    try:
        user_id = session_id.split('/')[-1]
        doc_ref = db.collection('assessments').document() # Let Firestore create a unique ID
        doc_ref.set({
            'userId': user_id,
            'assessmentDate': datetime.now(),
            'empathy': results.get('empathy', {}),
            'teamwork': results.get('teamwork', {}),
            'leadership': results.get('leadership', {}),
            'initiative': results.get('initiative', {})
        })
        print(f"Successfully saved assessment {doc_ref.id} for user {user_id}")
        return doc_ref.id # Return the unique ID
    except Exception as e:
        print(f"Error saving to Firestore: {e}")
        raise e

def get_params_from_context(request, context_name_suffix):
    """Helper to extract parameters from a specific incoming context."""
    contexts = request['queryResult']['outputContexts']
    for context in contexts:
        if context_name_suffix in context['name'] and 'parameters' in context:
            return context['parameters']
    return {}

# --- API Endpoints ---
@app.post("/dialogflow-webhook")
async def dialogflow_webhook(request: dict):
    try:
        intent_name = request['queryResult']['intent']['displayName']
        session_id = request['session']
        
        # --- Router for Ashoka Q&A ---
        if "ask_about_ashoka" in intent_name:
            user_question = request['queryResult']['queryText']
            relevant_chunks = search(user_question)
            final_answer = generate_full_answer(user_question, relevant_chunks)
            return JSONResponse(content={"fulfillmentText": final_answer})

        # --- Router for CMI Assessment ---
        elif intent_name == "cmi_assessment_START":
            question = random.choice(QUESTION_BANK["empathy"])
            initial_results = {}
            return JSONResponse(content={
                "fulfillmentText": f"Great! Let's start with Empathy.\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_empathy_response", "lifespanCount": 1, "parameters": {"current_question": question, "results": initial_results}}]
            })

        elif intent_name == "cmi_assessment_empathy_RESPONSE":
            user_answer = request['queryResult']['queryText']
            context_params = get_params_from_context(request, "awaiting_cmi_empathy_response")
            current_question = context_params.get("current_question", "")
            results = context_params.get("results", {})
            analysis = analyze_cmi_response("Empathy", current_question, user_answer)
            results['empathy'] = analysis
            question = random.choice(QUESTION_BANK["teamwork"])
            return JSONResponse(content={
                "fulfillmentText": f"Thank you. Now, for Teamwork:\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_teamwork_response", "lifespanCount": 1, "parameters": {"current_question": question, "results": results}}]
            })
        
        elif intent_name == "cmi_assessment_teamwork_RESPONSE":
            user_answer = request['queryResult']['queryText']
            context_params = get_params_from_context(request, "awaiting_cmi_teamwork_response")
            current_question = context_params.get("current_question", "")
            results = context_params.get("results", {})
            analysis = analyze_cmi_response("Teamwork", current_question, user_answer)
            results['teamwork'] = analysis
            question = random.choice(QUESTION_BANK["leadership"])
            return JSONResponse(content={
                "fulfillmentText": f"Interesting. Next, let's reflect on Leadership:\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_leadership_response", "lifespanCount": 1, "parameters": {"current_question": question, "results": results}}]
            })

        elif intent_name == "cmi_assessment_leadership_RESPONSE":
            user_answer = request['queryResult']['queryText']
            context_params = get_params_from_context(request, "awaiting_cmi_leadership_response")
            current_question = context_params.get("current_question", "")
            results = context_params.get("results", {})
            analysis = analyze_cmi_response("Leadership", current_question, user_answer)
            results['leadership'] = analysis
            question = random.choice(QUESTION_BANK["initiative"])
            return JSONResponse(content={
                "fulfillmentText": f"Almost done! Finally, let's talk about Initiative:\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_initiative_response", "lifespanCount": 1, "parameters": {"current_question": question, "results": results}}]
            })

        elif intent_name == "cmi_assessment_initiative_RESPONSE":
            user_answer = request['queryResult']['queryText']
            context_params = get_params_from_context(request, "awaiting_cmi_initiative_response")
            current_question = context_params.get("current_question", "")
            results = context_params.get("results", {})
            analysis = analyze_cmi_response("Initiative", current_question, user_answer)
            results['initiative'] = analysis
            
            assessment_id = save_assessment_to_firestore(session_id, results)
            
            dashboard_url = f"https://changemaker-dost-api.onrender.com/dashboard?id={assessment_id}"
            
            return JSONResponse(content={
                "fulfillmentText": f"Thank you for completing the CMI reflection! You can view your personal dashboard using this secure link: {dashboard_url}"
            })
        
        elif intent_name == "cmi_show_dashboard":
            return JSONResponse(content={"fulfillmentText": "You can view your dashboard using the link provided after your assessment."})

    except Exception as e:
        print(f"Error in main webhook: {e}")
        return JSONResponse(content={"fulfillmentText": "Sorry, an error occurred on my end."})

# --- NEW ENDPOINTS FOR THE DASHBOARD ---
@app.get("/dashboard")
async def get_dashboard_page():
    """Serves the dashboard.html file."""
    return FileResponse('dashboard.html')

@app.get("/assessment/{assessment_id}")
async def get_assessment_data(assessment_id: str):
    """Fetches a specific assessment's data from Firestore."""
    try:
        doc_ref = db.collection('assessments').document(assessment_id)
        doc = doc_ref.get()
        if doc.exists:
            return JSONResponse(content=doc.to_dict())
        else:
            return JSONResponse(content={"error": "Assessment not found"}, status_code=404)
    except Exception as e:
        print(f"Error fetching assessment {assessment_id}: {e}")
        return JSONResponse(content={"error": "Could not fetch assessment data"}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "Ashoka Bot Backend is running!"}
