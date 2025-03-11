import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "query": "what is the weather in beijing?"
    }
)
print(response.json())