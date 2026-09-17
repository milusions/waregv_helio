import os
import json
import yaml
from typing import Optional, Dict, List
from openai import OpenAI
from tools import call_rest_api, get_favourites, set_favourite

# 1. Initialize OpenAI client pointing to Gemini
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 2. Function to load and build the system prompt from config.yaml
def build_system_prompt() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir,"configs", "config.yaml")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    prompt = config.get("agent_instructions", "") + "\n\nHere are the APIs available to you:\n"
    
    for i, api in enumerate(config.get("apis", [])):
        prompt += f"{i+1}. {api['name']}:\n"
        prompt += f"   - Purpose: {api['purpose']}\n"
        prompt += f"   - URL: {api['url']}\n"
        prompt += f"   - Method: {api['method']}\n"
        prompt += f"   - Input format (payload): {api['input_format']}\n"
        prompt += f"   - Output format: {api['output_format']}\n\n"
        
    return prompt

SYSTEM_PROMPT = build_system_prompt()

# 3. Define all tool schemas for Gemini function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "call_rest_api",
            "description": "Makes an HTTP request to an external REST API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The REST API URL"},
                    "method": {"type": "string", "description": "HTTP method (GET, POST, etc.)"},
                    "payload": {"type": "object", "description": "JSON dictionary of input data"}
                },
                "required": ["url", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_favourites",
            "description": "Retrieves all saved favorite locations from config.yaml.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_favourite",
            "description": "Saves or updates a new favorite location label and its coordinates in config.yaml.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "The name/label of the favorite location"},
                    "x": {"type": "number", "description": "X coordinate"},
                    "y": {"type": "number", "description": "Y coordinate"},
                    "yaw_w": {"type": "number", "description": "Orientation yaw_w (default 1.0)"}
                },
                "required": ["label", "x", "y"]
            }
        }
    }
]

# Dictionary-based session store mapping call_id -> list of message dictionaries
sessions: Dict[str, List[dict]] = {}

def process_agent_request(user_message: str, call_id: str) -> str:
    """Handles agentic loop with support for multi-step tool calling per call_id."""
    global sessions
    
    # Initialize session history if new
    if call_id not in sessions:
        sessions[call_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    
    conversation_history = sessions[call_id]
    
    # Append the new user message to session history
    conversation_history.append({"role": "user", "content": user_message})
    
    # Multi-step tool execution loop: continues as long as Gemini wants to call tools
    while True:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=conversation_history,
            tools=tools,
            temperature=0.0
        )
        
        response_message = response.choices[0].message
        
        # If Gemini didn't request a tool call, it's providing the final text response
        if not response_message.tool_calls:
            conversation_history.append(response_message)
            return response_message.content
            
        # Append the assistant message containing the tool calls to history
        conversation_history.append(response_message)
        
        # Execute each requested tool and append its result back into history
        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            
            tool_result = ""
            if func_name == "call_rest_api":
                tool_result = call_rest_api(
                    url=args.get("url"), 
                    method=args.get("method"), 
                    payload=args.get("payload")
                )
            elif func_name == "get_favourites":
                tool_result = json.dumps(get_favourites())
            elif func_name == "set_favourite":
                tool_result = set_favourite(
                    label=args.get("label"),
                    x=args.get("x"),
                    y=args.get("y"),
                    yaw_w=args.get("yaw_w", 1.0)
                )
                
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": str(tool_result)
            })