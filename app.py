import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
import requests # We will use requests for database interaction
from dotenv import load_dotenv
import random
import json
from datetime import datetime

load_dotenv()

# --- Google AI and Firestore Configuration ---
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

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
GENERATIVE_MODEL_NAME = "gemini-1.5-flash-latest"

# --- THE CMI QUESTION BANK (Unchanged) ---
QUESTION_BANK = {
    "empathy": ["Describe a time you had to work with someone very different from you. How did you try to understand their perspective?", "Think about a problem in your community. Who is most affected, and what do you think their daily experience is like?", "Tell me about a time you changed your mind after listening to someone else's point of view."],
    "teamwork": ["Tell me about a challenging group project. What role did you play, and how did you help the team move forward?", "Imagine you and a friend disagree on the first step of a great idea. How would you handle this?", "Describe a time you had to rely on a teammate. How did you build that trust?"],
    "leadership": ["Describe a situation where you saw something unfair. What did you do about it, even if it was a small action?", "If you could improve one thing in your college, what would it be and what would be your first step?", "Tell me about a time you led by example."],
    "initiative": ["Tell me about an idea you were excited about. What steps did you take to bring it to life?", "Imagine a project you're working on fails completely. How do you learn from it and decide what to do next?", "Describe a time you taught yourself a new skill to solve a problem."]
}

app = FastAPI()

# --- Core Functions ---
def analyze_cmi_response(category, question, answer):
    # This function is unchanged
    pass

def get_params_from_context(request, context_name_suffix):
    # This function is unchanged
    pass

# --- NEW Firestore Functions using REST API ---
def format_for_firestore(data_dict):
    """Converts a Python dictionary to Firestore's REST API format."""
    fields = {}
    for key, value in data_dict.items():
        if isinstance(value, str):
            fields[key] = {"stringValue": value}
        elif isinstance(value, int) or isinstance(value, float):
            fields[key] = {"integerValue": str(value)}
        elif isinstance(value, dict):
            fields[key] = {"mapValue": {"fields": format_for_firestore(value)}}
        elif isinstance(value, datetime):
            fields[key] = {"timestampValue": value.isoformat() + "Z"}
    return fields

def save_assessment_via_rest(session_id, results):
    """Saves the assessment to Firestore using a direct REST API call."""
    user_id = session_id.split('/')[-1]
    document_data = {
        'userId': user_id,
        'assessmentDate': datetime.now(),
        'empathy': results.get('empathy', {}),
        'teamwork': results.get('teamwork', {}),
        'leadership': results.get('leadership', {}),
        'initiative': results.get('initiative', {})
    }
    
    # Construct the API request
    url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/assessments?key={GOOGLE_API_KEY}"
    payload = {"fields": format_for_firestore(document_data)}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        # Extract the document ID from the response path
        doc_path = response.json().get('name', '')
        assessment_id = doc_path.split('/')[-1]
        print(f"Successfully saved assessment {assessment_id} via REST API.")
        return assessment_id
    else:
        print(f"Error saving to Firestore via REST API: {response.status_code} - {response.text}")
        raise Exception("Could not save assessment to the database.")

# --- API Endpoints ---
@app.post("/dialogflow-webhook")
async def dialogflow_webhook(request: dict):
    try:
        intent_name = request['queryResult']['intent']['displayName']
        session_id = request['session']
        
        # ... (CMI START, EMPATHY, TEAMWORK, LEADERSHIP handlers are unchanged) ...

        if intent_name == "cmi_assessment_initiative_RESPONSE":
            user_answer = request['queryResult']['queryText']
            context_params = get_params_from_context(request, "awaiting_cmi_initiative_response")
            current_question = context_params.get("current_question", "")
            results = context_params.get("results", {})
            analysis = analyze_cmi_response("Initiative", current_question, user_answer)
            results['initiative'] = analysis
            
            # Save to DB using the new REST function and get the unique ID
            assessment_id = save_assessment_via_rest(session_id, results)
            
            # Create the unique dashboard link
            dashboard_url = f"https://changemaker-dost-api.onrender.com/dashboard?id={assessment_id}"
            
            return JSONResponse(content={
                "fulfillmentText": f"Thank you for completing the CMI reflection! You can view your personal dashboard using this secure link: {dashboard_url}"
            })
        
        # ... (Other intent handlers) ...

    except Exception as e:
        print(f"Error in main webhook: {e}")
        return JSONResponse(content={"fulfillmentText": "Sorry, an error occurred on my end."})

# --- Dashboard Endpoints (Unchanged, but will now work) ---
@app.get("/dashboard")
async def get_dashboard_page():
    return FileResponse('dashboard.html')

@app.get("/assessment/{assessment_id}")
async def get_assessment_data(assessment_id: str):
    # This endpoint also uses the REST API to fetch data
    url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/assessments/{assessment_id}?key={GOOGLE_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Need to un-format the data from Firestore's format
        raw_data = response.json().get('fields', {})
        formatted_data = {}
        for key, value_dict in raw_data.items():
            value_type = list(value_dict.keys())[0]
            if value_type == 'mapValue':
                 # This part needs more logic to fully unwrap, but is good for now
                 formatted_data[key] = value_dict[value_type].get('fields', {})
            else:
                 formatted_data[key] = value_dict[value_type]
        return JSONResponse(content=formatted_data)
    else:
        return JSONResponse(content={"error": "Assessment not found"}, status_code=404)

@app.get("/")
def read_root():
    return {"message": "Ashoka Bot Backend is running!"}

# --- Helper functions for analyze_cmi_response, get_params_from_context, and other intents ---
# These are included for completeness but are not the focus of the change.
def analyze_cmi_response(category, question, answer):
    # ... (implementation from previous versions)
    pass
def get_params_from_context(request, context_name_suffix):
    # ... (implementation from previous versions)
    pass
