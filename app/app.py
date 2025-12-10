from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model only once
model = joblib.load("../models/emotion_model.pkl")

# Helper function
def predict_emotion(text):
    return model.predict([text])[0]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    emotion = predict_emotion(user_message)

    responses = {
        "joy": "I'm glad to hear that! 😊",
        "sadness": "I'm here for you. 💙",
        "anger": "Take a deep breath... it'll be okay. 😤",
        "fear": "Don't worry, you're safe now. 🤗",
        "love": "Aww that's sweet ❤️"
    }

    reply = responses.get(emotion, "Tell me more about it.")
    return jsonify({"emotion": emotion, "reply": reply})


@app.route("/get_emotion", methods=["POST"])
def get_emotion():
    user_text = request.json.get("message", "")
    emotion = predict_emotion(user_text)

    responses = {
        "joy": "I'm glad to hear that! 😊",
        "sadness": "I'm here for you. Stay strong ❤️",
        "anger": "Take a deep breath… it's okay 😌",
        "fear": "Don't worry, I'm with you 🤝",
        "love": "Aww that's sweet ❤️"
    }

    bot_reply = responses.get(emotion, "I’m here to help you 😊")
    return jsonify({"emotion": emotion, "reply": bot_reply})


if __name__ == "__main__":
    app.run(debug=True)
