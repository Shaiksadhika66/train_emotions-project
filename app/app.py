from flask import Flask, request, jsonify, render_template
import joblib


from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/emotion_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    prediction = model.predict([user_message])[0]

    responses = {
        "joy": "I'm glad to hear that! 😊",
        "sadness": "I'm here for you. 💙",
        "anger": "Take a deep breath... it'll be okay. 😤",
        "fear": "Don't worry, you're safe now. 🤗"
    }

    return jsonify({
        "emotion": prediction,
        "reply": responses.get(prediction, "Tell me more about it.")
    })

if __name__ == "__main__":
    app.run(debug=True)
