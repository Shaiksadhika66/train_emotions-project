import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 🔥 Improved dataset with more emotion variety
data = {
    "joy": [
        "I am very happy today",
        "This is awesome!",
        "I'm feeling great",
        "I love this so much",
        "Everything is going amazing",
        "This made my day"
    ],
    "sadness": [
        "I am sad",
        "Feeling low",
        "I'm crying",
        "I feel lonely",
        "This makes me want to cry",
        "I'm depressed today"
    ],
    "anger": [
        "I'm so mad",
        "This is frustrating",
        "I hate this so much",
        "I'm angry right now",
        "That really annoys me",
        "I'm furious about this"
    ],
    "fear": [
        "I'm scared",
        "This is terrifying",
        "I feel nervous",
        "I'm afraid something will go wrong",
        "This makes me anxious",
        "I'm worried"
    ]
}

# Prepare data for training
texts = []
labels = []

for emotion, sentences in data.items():
    for s in sentences:
        texts.append(s)
        labels.append(emotion)

# 🧠 Use TF-IDF + Logistic Regression pipeline (better accuracy)
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

model.fit(texts, labels)

# Save model
joblib.dump(model, "models/emotion_model.pkl")

print("✅ Model retrained and saved successfully!")
