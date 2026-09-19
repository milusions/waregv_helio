from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from agent import process_agent_request

app = FastAPI(title="WareGV Helio")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=False,  
    allow_methods=["*"],     
    allow_headers=["*"],     
)


class AgentInput(BaseModel):
    message: str
    call_id: str 

class AgentOutput(BaseModel):
    output: str

@app.post("/agentic/waregv", response_model=AgentOutput)
async def handle_request(payload: AgentInput):

    final_response = process_agent_request(payload.message, payload.call_id)
    return AgentOutput(output=final_response or "No response generated.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)