# 🤖 AI Chatbot using NLP & Logistic Regression  

## 📌 Overview  
This project is an AI-powered chatbot built using **Natural Language Processing (NLP)** and **Logistic Regression** to classify user intents and generate relevant responses. The chatbot is deployed with an interactive web-based UI using **Streamlit**.

## 🚀 Features  
✅ **Intent-Based Chatbot:** Uses machine learning to classify user queries and respond accordingly.  
✅ **NLP Processing:** Implements **TF-IDF Vectorization** for text processing.  
✅ **Machine Learning Model:** Uses **Logistic Regression** for intent classification.  
✅ **User-Friendly UI:** Interactive chat interface built with **Streamlit**.  
✅ **Chat History Logging:** Stores previous conversations in a CSV file.  
✅ **Customizable Interface:** Styled using **CSS** for a modern look.  

## 🛠️ Tech Stack  
- **Python** (Core language)  
- **Streamlit** (UI framework)  
- **NLTK** (Natural Language Processing)  
- **scikit-learn** (Machine Learning)  
- **TfidfVectorizer** (Feature extraction)  
- **Logistic Regression** (Classification)  
- **CSV & JSON** (Data storage)  

## 📂 Project Structure  
```
📁 chatbot_project/
│── 📜 intents.json               # Predefined intents and responses  
│── 📜 chatbot.py                  # Main chatbot script  
│── 📜 requirements.txt            # Dependencies  
│── 📜 README.md                   # Documentation  
│── 📜 chat_log.csv                # Chat history log  
```

## ⚙️ Installation & Setup  
### 1️⃣ Clone the Repository  
```sh
git clone https://github.com/your-username/chatbot_project.git
cd chatbot_project
```

### 2️⃣ Install Dependencies  
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Chatbot  
```sh
streamlit run chatbot.py
```

## 📊 How It Works  
1. **Text Processing:** The chatbot tokenizes and vectorizes the user's input.  
2. **Classification:** The trained Logistic Regression model predicts the intent.  
3. **Response Selection:** The chatbot returns a predefined response based on the predicted intent.  
4. **Chat Logging:** User queries and responses are stored in a CSV file.  

## 📝 Future Enhancements  
🔹 Add **Deep Learning models (LSTMs, Transformers)** for better accuracy.  
🔹 Improve **Intent Detection** with more training data.  
🔹 Integrate with **Databases or APIs** for dynamic responses.  
🔹 Implement **Speech-to-Text** for voice interactions.  

## 🤝 Contributing  
Want to improve this chatbot? Feel free to fork, contribute, and submit pull requests!  

## 📜 License  
This project is open-source and available under the **MIT License**.  

---

Would you like to add any more details, such as images, demo links, or API integrations? 🚀


