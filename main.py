from fastapi import FastAPI, Body
from agents.router_agent import RouterAgent
import uvicorn
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize RouterAgent with API key from config
router_agent = RouterAgent(api_key=os.environ.get("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest = Body(...)):
    response = await router_agent.process_query(request.query)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 