import faiss
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import google.generativeai as genai
import os


# IMPORTANT: Replace API_KEY if needed
API_KEY = 'AIzaSyB0YSeFv6ttS2VSLP7yuzJZkOccB20F1ak' 
genai.configure(api_key=API_KEY)



# print("Loading the FAISS index...")
index = faiss.read_index("ashoka_index.faiss")

# print("Loading the text chunks...")
with open("text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)

print("Loading the embedding model...")
model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')


def search(query, k=4):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k)
    retrieved_chunks = [text_chunks[i] for i in I[0]]
    return retrieved_chunks

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    
    prompt = f"""
    You are a helpful assistant for the Changemaker Dost chatbot.
    Answer the following user's question based ONLY on the provided context from the Ashoka website.
    If the context does not contain the answer, state clearly that you couldn't find the information on the website.

    Context:
    {context}

    User's Question:
    {query}

    Answer:
    """
    
    try:
        # --- THIS IS THE ONLY LINE THAT HAS CHANGED ---
        # Using a newer, recommended model name.
        llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"



if __name__ == '__main__':
    # Ask a question
    user_question = "Tell me more about Yashveer singh and about ashoka youngs changemakers program"

    print(f"\nUser Question: {user_question}")
    
    # Retrieve the relevant context
    relevant_chunks = search(user_question)
    
    # Generate a clean answer based on the context
    print("\nGenerating answer...")
    final_answer = generate_answer(user_question, relevant_chunks)
    
    print("\n-- Answer --")
    print(final_answer)