import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Streamlit Page Configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Custom Styling for Background and Chat Messages
st.markdown("""
    <style>
        /* Set background color */
        body {
            background-color: #f5f5f5;
        }

        /* Sidebar Enhancements */
        [data-testid="stSidebar"] {
            background-color: #2E3B4E;
            color: white;
        }

        /* Chat container */
        .chat-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            max-width: 80%;
            margin: auto;
        }

        /* Chat message styling */
        .user-message {
            text-align: right;
            color: white;
            background-color: #007AFF;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            float: right;
            margin: 10px;
        }

        .bot-message {
            text-align: left;
            color: black;
            background-color: #EAEAEA;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            float: left;
            margin: 10px;
        }

        /* Clearfix to prevent layout break */
        .clearfix {
            clear: both;
        }

        /* Sidebar Buttons */
        .sidebar-button {
            background-color: #FF4B4B;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin-top: 20px;
        }

        .sidebar-header {
            font-size: 18px;
            font-weight: bold;
            color: white;
            padding-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# SSL Fix for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents from JSON file
file_path = os.path.abspath("C:/Users/dhars/.ipynb_checkpoints/implementation_of_chatbot_using_NLP/intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Train the NLP Model
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Function to get chatbot response
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I don't understand."

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Layout - Top: Conversation History, Bottom: Clear Button
st.sidebar.markdown('<div class="sidebar-header">📜 Conversation History</div>', unsafe_allow_html=True)

# Show Conversation History in Sidebar
if os.path.exists("chat_log.csv"):
    with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)  # Skip header
        for row in csv_reader:
            st.sidebar.text(f"{row[0]}")
            st.sidebar.text(f"🤖 {row[1]}")
            st.sidebar.text(f"🕒 {row[2]}")
            st.sidebar.markdown("---")
else:
    st.sidebar.warning("No chat history yet.")

# Sidebar - Clear Chat Button (Bottom)
if st.sidebar.button("🗑️ Clear Chat History"):
    os.remove("chat_log.csv") if os.path.exists("chat_log.csv") else None
    st.session_state.chat_history = []  # Reset chat history
    st.sidebar.success("Chat history cleared!")
    st.rerun()

# 📌 Home Screen
st.title("🤖 AI Chatbot")
st.write("Welcome! Type a message to start the conversation.")

# User input field
user_input = st.chat_input("Type your message...")

if user_input:
    # Store user message
    st.session_state.chat_history.append({"role": "user", "message": user_input})

    # Get chatbot response
    bot_response = chatbot(user_input)

    # Store chatbot response
    st.session_state.chat_history.append({"role": "bot", "message": bot_response})

    # Save to conversation history
    with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([user_input, bot_response, timestamp])

    st.rerun()  # Refresh UI to display chat

# Display chat messages with icons
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for entry in st.session_state.chat_history:
    role, message = entry["role"], entry["message"]
    if role == "user":
        st.markdown(f"<div class='user-message'> {message}</div><div class='clearfix'></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-message'>🤖 {message}</div><div class='clearfix'></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ℹ️ About Section (No Changes)
st.sidebar.markdown("---")
st.sidebar.markdown("ℹ️ **About the Chatbot**")
st.sidebar.markdown("""
The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) and Logistic Regression.

### **Project Overview:**
1. NLP techniques and Logistic Regression algorithm are used to train the chatbot on labeled intents.
2. Streamlit is used to build a web-based chatbot interface.

### **Dataset:**
- Intents: The user intent (e.g., "greeting", "budget", "about").
- Entities: Extracted words from user input.
- Text: The actual user input.

### **Streamlit Chatbot Interface:**
- Built using Streamlit for interactive web-based chatting.

### **Conclusion:**
This chatbot understands and responds based on trained intents. It can be enhanced with more data and deep learning models in the future.
""")

# Run the Streamlit App
if __name__ == '__main__':
    st.rerun()
