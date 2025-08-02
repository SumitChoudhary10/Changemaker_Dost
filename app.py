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
    query_embedding_dict = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = np.array([query_embedding_dict['embedding']]).astype('float32')
    D, I = index.search(query_embedding, k)
    return [text_chunks[i] for i in I[0]]

def generate_full_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"You are 'Changemaker Dost', a friendly AI assistant. Answer the user's question about Ashoka. Prioritize information from the Official Context. If the answer isn't there, use your general knowledge. Never say 'Based on the provided text'.\n\nOfficial Context: --- {context} ---\nUser's Question: {query}\nAnswer:"
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text

def analyze_cmi_response(category, question, answer):
    prompt = f"You are an expert in evaluating changemaking skills (Empathy, Teamwork, Leadership, Initiative). The user was asked a question to assess their '{category}':\nQuestion: \"{question}\"\nThe user responded:\nAnswer: \"{answer}\"\nAnalyze their response. Provide a score from 1 to 10 for their '{category}' skill and a one-sentence justification. Respond ONLY with a valid JSON object like: {{\"score\": <score_number>, \"justification\": \"<one_sentence_justification>\"}}"
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    try:
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error decoding AI analysis: {e}")
        return {"score": 0, "justification": "Could not analyze response."}

def get_question_from_context(request, context_name_suffix):
    """Helper function to extract the question from the incoming context."""
    contexts = request['queryResult']['outputContexts']
    for context in contexts:
        if context_name_suffix in context['name'] and 'parameters' in context:
            return context['parameters'].get('current_question', "")
    return ""

# --- API Endpoints ---

@app.post("/dialogflow-webhook")
async def dialogflow_webhook(request: dict):
    try:
        intent_name = request['queryResult']['intent']['displayName']
        session_id = request['session']
        print(f"Received intent: {intent_name}")

        # --- Router for Ashoka Q&A ---
        if "ask_about_ashoka" in intent_name:
            user_question = request['queryResult']['queryText']
            relevant_chunks = search(user_question)
            final_answer = generate_full_answer(user_question, relevant_chunks)
            return JSONResponse(content={"fulfillmentText": final_answer})

        # --- Router for CMI Assessment ---
        elif intent_name == "cmi_assessment_START":
            question = random.choice(QUESTION_BANK["empathy"])
            return JSONResponse(content={
                "fulfillmentText": f"Great! Let's start with Empathy. Here is your first question:\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_empathy_response", "lifespanCount": 1, "parameters": {"current_question": question}}]
            })

        elif intent_name == "cmi_assessment_empathy_RESPONSE":
            user_answer = request['queryResult']['queryText']
            current_question = get_question_from_context(request, "awaiting_cmi_empathy_response")
            analysis = analyze_cmi_response("Empathy", current_question, user_answer)
            print(f"Empathy Analysis: {analysis}") # We will store this later
            
            question = random.choice(QUESTION_BANK["teamwork"])
            return JSONResponse(content={
                "fulfillmentText": f"Thank you for sharing. Now, let's think about Teamwork:\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_teamwork_response", "lifespanCount": 1, "parameters": {"current_question": question}}]
            })

        elif intent_name == "cmi_assessment_teamwork_RESPONSE":
            user_answer = request['queryResult']['queryText']
            current_question = get_question_from_context(request, "awaiting_cmi_teamwork_response")
            analysis = analyze_cmi_response("Teamwork", current_question, user_answer)
            print(f"Teamwork Analysis: {analysis}")

            question = random.choice(QUESTION_BANK["leadership"])
            return JSONResponse(content={
                "fulfillmentText": f"Interesting. Next, let's reflect on Leadership:\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_leadership_response", "lifespanCount": 1, "parameters": {"current_question": question}}]
            })

        elif intent_name == "cmi_assessment_leadership_RESPONSE":
            user_answer = request['queryResult']['queryText']
            current_question = get_question_from_context(request, "awaiting_cmi_leadership_response")
            analysis = analyze_cmi_response("Leadership", current_question, user_answer)
            print(f"Leadership Analysis: {analysis}")

            question = random.choice(QUESTION_BANK["initiative"])
            return JSONResponse(content={
                "fulfillmentText": f"Almost done! Finally, let's talk about Initiative:\n\n{question}",
                "outputContexts": [{"name": f"{session_id}/contexts/awaiting_cmi_initiative_response", "lifespanCount": 1, "parameters": {"current_question": question}}]
            })

        elif intent_name == "cmi_assessment_initiative_RESPONSE":
            user_answer = request['queryResult']['queryText']
            current_question = get_question_from_context(request, "awaiting_cmi_initiative_response")
            analysis = analyze_cmi_response("Initiative", current_question, user_answer)
            print(f"Initiative Analysis: {analysis}")

            return JSONResponse(content={
                "fulfillmentText": "Thank you for completing the CMI reflection! We've saved your responses. Soon, you'll be able to see your results on a personal dashboard."
            })

        else:
            return JSONResponse(content={"fulfillmentText": "I'm not sure how to handle that."})

    except Exception as e:
        print(f"Error in main webhook: {e}")
        return JSONResponse(content={"fulfillmentText": "Sorry, an error occurred on my end."})

@app.get("/")
async def read_root():
    return {"message": "Ashoka Bot Backend is running!"}
