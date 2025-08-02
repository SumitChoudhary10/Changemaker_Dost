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
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# --- Firebase Initialization ---
print("Initializing Firebase...")
try:
    # Use the secret key file you downloaded
    cred = credentials.Certificate("serviceAccountKey.json")
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
# ... (The Ashoka Q&A functions search() and generate_full_answer() are unchanged) ...
def search(query, k=3):
    # This function is unchanged
    pass
def generate_full_answer(query, context_chunks):
    # This function is unchanged
    pass

# --- CMI Analysis and Data Storage ---
def analyze_cmi_response(category, question, answer):
    # This function is unchanged
    pass

def save_assessment_to_firestore(session_id, results):
    """Saves the completed assessment results to Firestore."""
    try:
        # We use a part of the Dialogflow session ID as a simple user ID for now
        user_id = session_id.split('/')[-1]
        
        # Create a new document in the 'assessments' collection
        doc_ref = db.collection('assessments').document()
        doc_ref.set({
            'userId': user_id,
            'assessmentDate': datetime.now(),
            'empathy': results.get('empathy', {}),
            'teamwork': results.get('teamwork', {}),
            'leadership': results.get('leadership', {}),
            'initiative': results.get('initiative', {})
        })
        print(f"Successfully saved assessment for user {user_id}")
    except Exception as e:
        print(f"Error saving to Firestore: {e}")


# --- API Endpoints ---
@app.post("/dialogflow-webhook")
async def dialogflow_webhook(request: dict):
    try:
        intent_name = request['queryResult']['intent']['displayName']
        session_id = request['session']
        print(f"Received intent: {intent_name}")

        # --- Router for Ashoka Q&A ---
        if "ask_about_ashoka" in intent_name:
            # ... (This part is unchanged) ...
            pass

        # --- Router for CMI Assessment ---
        elif intent_name == "cmi_assessment_START":
            question = random.choice(QUESTION_BANK["empathy"])
            # We will now pass an empty results object through the contexts
            initial_results = {}
            return JSONResponse(content={
                "fulfillmentText": f"Great! Let's start with Empathy.\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_empathy_response", "lifespanCount": 1, "parameters": {"current_question": question, "results": initial_results}}]
            })

        elif intent_name == "cmi_assessment_empathy_RESPONSE":
            # ... (Code to get user_answer and current_question is unchanged) ...
            # Get the results object from the incoming context
            results = {} # Get results from context
            analysis = analyze_cmi_response("Empathy", current_question, user_answer)
            results['empathy'] = analysis
            
            question = random.choice(QUESTION_BANK["teamwork"])
            return JSONResponse(content={
                "fulfillmentText": f"Thank you. Now, for Teamwork:\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_teamwork_response", "lifespanCount": 1, "parameters": {"current_question": question, "results": results}}]
            })

        elif intent_name == "cmi_assessment_teamwork_RESPONSE":
            # ... (Code to get user_answer and current_question is unchanged) ...
            results = {} # Get results from context
            analysis = analyze_cmi_response("Teamwork", current_question, user_answer)
            results['teamwork'] = analysis

            question = random.choice(QUESTION_BANK["leadership"])
            return JSONResponse(content={
                "fulfillmentText": f"Interesting. Next, Leadership:\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_leadership_response", "lifespanCount": 1, "parameters": {"current_question": question, "results": results}}]
            })

        elif intent_name == "cmi_assessment_leadership_RESPONSE":
            # ... (Code to get user_answer and current_question is unchanged) ...
            results = {} # Get results from context
            analysis = analyze_cmi_response("Leadership", current_question, user_answer)
            results['leadership'] = analysis

            question = random.choice(QUESTION_BANK["initiative"])
            return JSONResponse(content={
                "fulfillmentText": f"Almost done! Finally, Initiative:\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_initiative_response", "lifespanCount": 1, "parameters": {"current_question": question, "results": results}}]
            })

        elif intent_name == "cmi_assessment_initiative_RESPONSE":
            # ... (Code to get user_answer and current_question is unchanged) ...
            results = {} # Get results from context
            analysis = analyze_cmi_response("Initiative", current_question, user_answer)
            results['initiative'] = analysis
            
            # Now we save the final results to the database
            save_assessment_to_firestore(session_id, results)

            return JSONResponse(content={
                "fulfillmentText": "Thank you for completing the CMI reflection! Your results have been saved. You can ask to see your dashboard at any time."
            })

        else:
            return JSONResponse(content={"fulfillmentText": "I'm not sure how to handle that."})

    except Exception as e:
        print(f"Error in main webhook: {e}")
        return JSONResponse(content={"fulfillmentText": "Sorry, an error occurred on my end."})

# ... (The root endpoint @app.get("/") is unchanged) ...
