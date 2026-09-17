from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from agent import process_agent_request

app = FastAPI(title="YAML Configured Gemini Agent")

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allows all origins
    allow_credentials=False,  # Must be False when allow_origins is ["*"]
    allow_methods=["*"],      # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],      # Allows all headers
)
# ---------------------------

class AgentInput(BaseModel):
    message: str
    call_id: str  # Added call_id parameter from the request payload

class AgentOutput(BaseModel):
    output: str

@app.post("/agentic/waregv", response_model=AgentOutput)
async def handle_request(payload: AgentInput):
    # Pass both the message and call_id to your agent processor
    final_response = process_agent_request(payload.message, payload.call_id)
    return AgentOutput(output=final_response)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)