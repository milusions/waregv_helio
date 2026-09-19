import os
import json
import yaml
import logging
from datetime import datetime
from typing import Optional, Dict, List
from openai import OpenAI
from configs.tools import handle_tool
import uuid


# Setup agent logger
LOG_PATH = "log/agent.log"
logger = logging.getLogger("AgentLogger")
logger.setLevel(logging.INFO)
logger.propagate = False
os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

config_data = None
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "configs", "config.yaml")
with open(config_path, "r") as file:
    config_data = yaml.safe_load(file)


def build_system_prompt() -> str:
    global config_data

    prompt = config_data.get("agent_instructions", "") + "\n\nHere are the APIs available to you:\n"
    
    for i, api in enumerate(config_data.get("apis", [])):
        prompt += f"{i+1}. {api['name']}:\n"
        prompt += f"   - Purpose: {api['purpose']}\n"
        prompt += f"   - URL: {api['url']}\n"
        prompt += f"   - Method: {api['method']}\n"
        prompt += f"   - Input format (payload): {api['input_format']}\n"
        prompt += f"   - Output format: {api['output_format']}\n\n"
        
    return prompt


SYSTEM_PROMPT = build_system_prompt()
tools = config_data.get("tool_description", "")

sessions: Dict[str, List[dict]] = {}

conversation_id = str(uuid.uuid4())

def process_agent_request(user_message: str, call_id: str) -> str:
    """
    Handles agentic loop with support for multi-step tool calling.
    Logs user message, call_id, timestamp, conversation_id, thought process, tool calls/inputs, and final response.
    """
    global sessions,conversation_id
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(
        "--- NEW REQUEST --- | Time: %s | Conversation ID: %s | Call ID: %s | User Message: %s",
        current_time, conversation_id, call_id, user_message
    )
    
    if call_id not in sessions:
        conversation_id = str(uuid.uuid4())
        sessions[call_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    
    conversation_history = sessions[call_id]
    conversation_history.append({"role": "user", "content": user_message})
    
    while True:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=conversation_history,
            tools=tools,
            temperature=0.0
        )
        
        response_message = response.choices[0].message
        
        # Capture model's thought process / reasoning content if available
        thought_process = getattr(response_message, "reasoning_content", None) or response_message.content or "No direct text content/thought produced."
        logger.info(
            "[%s] [Call ID: %s] Thought Process / Model Output: %s",
            conversation_id, call_id, thought_process
        )
        
        if not response_message.tool_calls:
            conversation_history.append(response_message)
            final_content = response_message.content or "I have processed your request, but no text response was returned."
            logger.info(
                "[%s] [Call ID: %s] Final Response Sent to User: %s",
                conversation_id, call_id, final_content
            )
            return final_content
            
        conversation_history.append(response_message)
        
        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            
            logger.info(
                "[%s] [Call ID: %s] Tool Call Triggered -> Function: %s | Inputs: %s",
                conversation_id, call_id, func_name, args
            )
            
            tool_result = handle_tool(func_name, args)
                
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": str(tool_result)
            })