import requests

url = "http://127.0.0.1:5000/chat"
data = {"message": "I am feeling happy today"}

response = requests.post(url, json=data)
print("Response from Chatbot:", response.json())
