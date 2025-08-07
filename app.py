# FILE: app.py (Final Version with Firestore REST API)
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
import requests # We will use requests for database interaction
from dotenv import load_dotenv
import random
import json
from datetime import datetime
import google.generativeai as genai
import faiss
import pickle
import numpy as np

# NOTE: We are no longer using the 'firebase-admin' library

load_dotenv()

# --- Google AI and Firestore Configuration ---
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")
genai.configure(api_key=GOOGLE_API_KEY)

# Get Project ID from your service account key file
try:
    with open("serviceAccountKey.json") as f:
        service_account_info = json.load(f)
        PROJECT_ID = service_account_info.get('project_id')
    if not PROJECT_ID:
        raise ValueError("Could not find 'project_id' in serviceAccountKey.json")
except FileNotFoundError:
    raise FileNotFoundError("serviceAccountKey.json not found. It is needed to get your Project ID.")

# Define model names
EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATIVE_MODEL_NAME = "gemini-1.5-flash-latest"

# --- THE CMI QUESTION BANK (Unchanged) ---
QUESTION_BANK = {
    "empathy": ["Describe a time you had to work with someone very different from you. How did you try to understand their perspective?", "Think about a problem in your community. Who is most affected, and what do you think their daily experience is like?", "Tell me about a time you changed your mind after listening to someone else's point of view."],
    "teamwork": ["Tell me about a challenging group project. What role did you play, and how did you help the team move forward?", "Imagine you and a friend disagree on the first step of a great idea. How would you handle this?", "Describe a time you had to rely on a teammate. How did you build that trust?"],
    "leadership": ["Describe a situation where you saw something unfair. What did you do about it, even if it was a small action?", "If you could improve one thing in your college, what would it be and what would be your first step?", "Tell me about a time you led by example."],
    "initiative": ["Tell me about an idea you were excited about. What steps did you take to bring it to life?", "Imagine a project you're working on fails completely. How do you learn from it and decide what to do next?", "Describe a time you taught yourself a new skill to solve a problem."]
}

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

def get_params_from_context(request, context_name_suffix):
    """Helper to extract parameters from a specific incoming context."""
    contexts = request['queryResult']['outputContexts']
    for context in contexts:
        if context_name_suffix in context['name'] and 'parameters' in context:
            return context['parameters']
    return {}

# --- NEW Firestore Functions using REST API ---
def format_for_firestore(data_dict):
    """Converts a Python dictionary to Firestore's REST API format."""
    fields = {}
    for key, value in data_dict.items():
        if isinstance(value, str):
            fields[key] = {"stringValue": value}
        elif isinstance(value, int):
            fields[key] = {"integerValue": str(value)}
        elif isinstance(value, float):
            fields[key] = {"integerValue": str(int(value))}
        elif isinstance(value, dict):
            fields[key] = {"mapValue": {"fields": format_for_firestore(value)}}
        elif isinstance(value, datetime):
            fields[key] = {"timestampValue": value.isoformat() + "Z"}
    return fields

def save_assessment_via_rest(user_id, results):
    """Saves the assessment to Firestore using the persistent user ID."""
    document_data = {
        'userId': user_id,
        'assessmentDate': datetime.now(),
        'empathy': results.get('empathy', {}),
        'teamwork': results.get('teamwork', {}),
        'leadership': results.get('leadership', {}),
        'initiative': results.get('initiative', {})
    }
    
    url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/assessments?key={GOOGLE_API_KEY}"
    payload = {"fields": format_for_firestore(document_data)}
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        
        doc_path = response.json().get('name', '')
        assessment_id = doc_path.split('/')[-1]
        print(f"Successfully saved assessment {assessment_id} via REST API.")
        return assessment_id
    except requests.exceptions.RequestException as e:
        print(f"Error saving to Firestore via REST API: {e}")
        if e.response is not None:
            print(f"Response Body: {e.response.text}")
        raise Exception("Could not save assessment to the database.")


def unformat_from_firestore(fields):
    """Converts Firestore's REST API format back to a Python dictionary."""
    if not fields:
        return {}
    data = {}
    for key, value_dict in fields.items():
        if not value_dict:
            continue
        value_type = list(value_dict.keys())[0]
        if value_type == 'mapValue':
            data[key] = unformat_from_firestore(value_dict[value_type].get('fields'))
        elif value_type == 'integerValue':
            data[key] = int(value_dict[value_type])
        elif value_type == 'timestampValue':
            data[key] = value_dict[value_type]
        else:
            data[key] = value_dict[value_type]
    return data

def fetch_user_history_via_rest(user_id):
    """Fetches all assessments for a specific user using the REST API."""
    # This URL is for running a query against the database
    url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents:runQuery?key={GOOGLE_API_KEY}"
    
    # This payload tells Firestore to find all documents for the user and order them by date
    query_payload = {
        "structuredQuery": {
            "from": [{"collectionId": "assessments"}],
            "where": {
                "fieldFilter": {
                    "field": {"fieldPath": "userId"},
                    "op": "EQUAL",
                    "value": {"stringValue": user_id}
                }
            },
            "orderBy": [{
                "field": {"fieldPath": "assessmentDate"},
                "direction": "DESCENDING"
            }]
        }
    }
    try:
        response = requests.post(url, json=query_payload)
        response.raise_for_status()
        
        results = response.json()
        assessments = []
        for doc in results:
            if 'document' in doc:
                # We re-use our helper function to clean up the data
                formatted_doc = unformat_from_firestore(doc['document'].get('fields', {}))
                assessments.append(formatted_doc)
        
        return assessments
    except requests.exceptions.RequestException as e:
        print(f"Error fetching history for user {user_id}: {e}")
        if e.response is not None:
            print(f"Response Body: {e.response.text}")
        return None 
    

# --- NEW FUNCTION FOR PERSONALIZED ADVICE ---
def generate_improvement_tips(assessment_data):
    """Generates personalized improvement tips based on the lowest CMI score."""
    if not assessment_data:
        return "I couldn't find your latest assessment to give you tips. Please complete one first!"

    skills = ['empathy', 'teamwork', 'leadership', 'initiative']
    lowest_skill = ''
    lowest_score = 11  # Start with a score higher than max

    for skill in skills:
        score = assessment_data.get(skill, {}).get('score', 10)
        if score < lowest_score:
            lowest_score = score
            lowest_skill = skill

    if not lowest_skill:
        return "You're doing great in all areas! Keep practicing these skills in your daily life."

    prompt = f"""
    You are 'Changemaker Dost', an encouraging and helpful AI coach.
    A user has completed their CMI reflection and their lowest scoring area is '{lowest_skill}' with a score of {lowest_score}/10.
    Based on this, provide 2-3 actionable, practical, and encouraging advice for a college student in India to improve their '{lowest_skill}' skill.
    Frame the response as if you are speaking directly to the user. Start with something like, "It looks like a great area to focus on is {lowest_skill.capitalize()}. Here are a few ideas to help you grow:".
    Keep the tone positive and empowering. Use bullet points for the advice.
    """
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text


# --- API Endpoints ---
@app.post("/dialogflow-webhook")
async def dialogflow_webhook(request: dict):
    try:
        intent_name = request['queryResult']['intent']['displayName']
        session_id = request['session']
        user_id = session_id.split('/')[-1]
        
        if "ask_about_ashoka" in intent_name:
            user_question = request['queryResult']['queryText']
            relevant_chunks = search(user_question)
            final_answer = generate_full_answer(user_question, relevant_chunks)
            return JSONResponse(content={"fulfillmentText": final_answer})

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
            
            assessment_id = save_assessment_via_rest(user_id, results)
            dashboard_url = f"https://changemaker-dost-api.onrender.com/dashboard?id={assessment_id}"
            
            # --- THIS IS THE NEW, CLICKABLE LINK RESPONSE ---
            response_data = {
                "fulfillmentMessages": [
                    {
                        "text": {
                            "text": [
                                "Thank you for completing the CMI reflection! You can view your personal dashboard using this secure link:"
                            ]
                        }
                    },
                    {
                        "payload": {
                            "richContent": [
                                [
                                    {
                                        "type": "button",
                                        "icon": {
                                            "type": "open_in_new",
                                            "color": "#0d6efd"
                                        },
                                        "text": "View My Dashboard",
                                        "link": dashboard_url
                                    }
                                ],
                                [
                                    {
                                        "type" : "chips",
                                        "options" : [
                                            {
                                                "text" : "Run CMI assessment again"
                                            },
                                            {
                                                "text" : "How can I improve my CMI score?"
                                            }
                                        ]
                                    }
                                ]
                            ]
                        }
                    }
                ]
            }
            return JSONResponse(content=response_data)
        
        # --- NEW INTENT HANDLER FOR IMPROVEMENT TIPS ---
        elif intent_name == "cmi_improve_score":
            history = fetch_user_history_via_rest(user_id)
            if not history:
                return JSONResponse(content={"fulfillmentText": "I can't find your assessment results. Please complete one first so I can give you personalized advice."})
            
            latest_assessment = history[0]
            improvement_tips = generate_improvement_tips(latest_assessment)
            return JSONResponse(content={"fulfillmentText": improvement_tips})
        
        else:
            return JSONResponse(content={"fulfillmentText": "I'm not sure how to handle that."})

    except Exception as e:
        print(f"Error in main webhook: {e}")
        return JSONResponse(content={"fulfillmentText": "Sorry, an error occurred on my end."})

# --- Dashboard Endpoints ---
@app.get("/dashboard")
async def get_dashboard_page():
    return FileResponse('dashboard.html')

@app.get("/assessment/{assessment_id}")
async def get_assessment_data(assessment_id: str):
    url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/assessments/{assessment_id}?key={GOOGLE_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        raw_data = response.json().get('fields', {})
        formatted_data = unformat_from_firestore(raw_data)
        return JSONResponse(content=formatted_data)
    else:
        return JSONResponse(content={"error": "Assessment not found"}, status_code=404)

# --- NEW ENDPOINT TO GET USER HISTORY ---
@app.get("/history/{user_id}")
async def get_user_history(user_id: str):
    """Fetches all assessments for a specific user using the REST API."""
    # This URL is for running a query against the database
    url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents:runQuery?key={GOOGLE_API_KEY}"
    
    # This payload tells Firestore to find all documents for the user and order them by date
    query_payload = {
        "structuredQuery": {
            "from": [{"collectionId": "assessments"}],
            "where": {
                "fieldFilter": {
                    "field": {"fieldPath": "userId"},
                    "op": "EQUAL",
                    "value": {"stringValue": user_id}
                }
            },
            "orderBy": [{
                "field": {"fieldPath": "assessmentDate"},
                "direction": "DESCENDING"
            }]
        }
    }
    try:
        response = requests.post(url, json=query_payload)
        response.raise_for_status()
        
        results = response.json()
        assessments = []
        for doc in results:
            if 'document' in doc:
                # We re-use our helper function to clean up the data
                formatted_doc = unformat_from_firestore(doc['document'].get('fields', {}))
                assessments.append(formatted_doc)
        
        return JSONResponse(content=assessments)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching history for user {user_id}: {e}")
        if e.response is not None:
            print(f"Response Body: {e.response.text}")
        return JSONResponse(content={"error": "Could not fetch user history"}, status_code=500)
    

@app.get("/")
def read_root():
    return {"message": "Ashoka Bot Backend is running!"}
