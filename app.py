from flask import Flask, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import dotenv
from flask_cors import CORS

# Load environment variables
dotenv.load_dotenv()

# Create Flask app
app = Flask(__name__)

# Configure CORS to allow requests from the specified frontend origin
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173", "allow_headers": ["Content-Type"]}})

# Initialize the language model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define the prompt template with detailed instructions
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a virtual medical assistant with expertise in gathering health information and providing preliminary symptom analysis. Use this approach:

            1. Start by asking for the patient's name, age, gender, and location to understand basic demographics.
            2. Ask up to 15 concise, symptom-focused questions to gather detailed health information. Limit each question to one or two symptoms for clarity. Examples include:
               - "How long have you had these symptoms?"
               - "Do you have fever, headache, or chills?"
               - "Have you experienced nausea or vomiting?"
               - "Do you have any abdominal pain or discomfort?"

            3. Avoid asking multiple follow-ups at once; instead, let the patient answer each question one at a time to ensure a clear response.

            4. After the 15 questions, based on the patterns and symptoms gathered, provide a probable diagnosis. Mention common possible conditions, like gastroenteritis, flu, or viral infections, based on the combination of symptoms.

            5. Avoid alarming language. Emphasize that the assessment is preliminary and encourage consulting a healthcare provider for accurate diagnosis and treatment.

            6. Example dialogue:
               - Patient: "I have a fever and stomach pain."
               - Assistant: "Thank you. Could you also let me know if you're experiencing nausea or vomiting? If so, how long have you had these symptoms?"

            7. Once all questions are asked and responses received, offer a predictive diagnosis, like "Based on your symptoms of fever, stomach pain, and nausea, it could be a gastrointestinal infection or viral illness."

            Start by asking the patient’s basic information.
            """,
        ),
        (
            "placeholder", 
            "{chat_history}"
        ),
        (
            "user", 
            "{input}"
        ),
    ]
)


# Initialize message history
demo_ephemeral_chat_history = ChatMessageHistory()

# Create the chain for the chatbot
chain = prompt | llm

# Create a Runnable with message history
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Define chatbot function
def chat_bot(input):
    response = chain_with_message_history.invoke(
        {"input": input},
        {"configurable": {"session_id": "unused"}},
    )
    return response.content

# Define the chatbot route
@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    input_text = data.get("input_text")
    response = chat_bot(input_text)  # Call the chat bot logic
    return jsonify({"response": response})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
