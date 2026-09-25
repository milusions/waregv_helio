# agent.py
import os
import yaml
import logging
import sys
import json
from datetime import datetime
from typing import Optional, Dict, List
from openai import OpenAI
from configs.tools import handle_tool

# --- ANSI Colors for Beautiful Terminal ---
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
DIM = "\033[2m"

# Setup agent logger
LOG_PATH = os.environ.get("AGENT_LOG_PATH", "log/agent.log")

class BeautifulConsoleFormatter(logging.Formatter):
    """Formats terminal output to be beautiful and conversational while ignoring raw log syntax."""
    def format(self, record):
        msg = record.getMessage()
        time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        prefix = f"{DIM}[{time_str}]{RESET} "

        if msg.startswith("--- NEW REQUEST ---"):
            parts = msg.split(" | ")
            if len(parts) >= 4:
                conv_id = parts[2].replace("Conversation ID: ", "")
                user_msg = parts[3].replace("User Message: ", "")
                return f"\n{BOLD}{CYAN}╭─── Conversation ID: {conv_id} ───╮{RESET}\n{prefix}{BOLD}{GREEN}Input:{RESET} {user_msg}"
        
        elif "] New Helio conversation created." in msg:
            return f"{prefix}{YELLOW}✨ New conversation initialized. ✨{RESET}"
            
        elif "Final Response Sent to User:" in msg:
            parts = msg.split("Final Response Sent to User: ")
            response = parts[1] if len(parts) > 1 else ""
            return f"{prefix}{BOLD}{MAGENTA}Output:{RESET} {response}\n{BOLD}{CYAN}╰────────────────────────────────────────╯{RESET}"
            
        elif "] Tool Execution:" in msg:
            parts = msg.split("] Tool Execution: ")
            return f"{prefix}{DIM}⚙️ Executing tool: {parts[1]}{RESET}"

        elif "Model output:" in msg:
            return f"{prefix}{DIM}Processing model output...{RESET}"

        return f"{prefix}{msg}"

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
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(BeautifulConsoleFormatter())
    logger.addHandler(ch)

client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "configs", "config.yaml")
with open(config_path, "r") as file:
    config_data = yaml.safe_load(file)

SYSTEM_PROMPT = config_data.get("agent_instructions", "You are Helio.")
sessions: Dict[str, List[dict]] = {}

def _emit(event_callback, event_type: str, **payload):
    if event_callback:
        event_callback({"type": event_type, **payload})

def process_agent_request(user_message: str, conversation_id: str, event_callback=None) -> str:
    """Persistent agent loop handling both text generation and recursive tool execution."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(
        "--- NEW REQUEST --- | Time: %s | Conversation ID: %s | User Message: %s",
        current_time, conversation_id, user_message,
    )

    if conversation_id not in sessions:
        sessions[conversation_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("[%s] New Helio conversation created.", conversation_id)

    history = sessions[conversation_id]
    history.append({"role": "user", "content": user_message})
    _emit(event_callback, "status", stage="thinking", message="Helio is processing the request.")

    tools = config_data.get("tool_description", [])

    while True:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=history,
            temperature=0.0,
            tools=tools if tools else None,
        )
        
        response_message = response.choices[0].message
        
        # Check if the model requested any function calls
        if response_message.tool_calls:
            tool_calls_data = []
            for tc in response_message.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
                
            # Append the assistant's tool request to history
            history.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": tool_calls_data
            })
            
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                    
                logger.info("[%s] Tool Execution: %s(args=%s)", conversation_id, func_name, args)
                _emit(event_callback, "status", stage="tool", message=f"Accessing system data: {func_name}")
                
                # Execute the tool
                result = handle_tool(func_name, args)
                
                # Guarantee result is stringified for the LLM
                result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                
                # Append the tool's output back to history
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result_str
                })
            
            # The loop continues, sending the new history (including tool results) back to the LLM
        else:
            # The model returned a final text response
            final_content = response_message.content or "I have processed your request, but no text response was returned."
            
            logger.info("[%s] Model output: %s", conversation_id, final_content)
            _emit(event_callback, "model", stage="model", message=final_content)

            history.append({"role": "assistant", "content": final_content})

            logger.info("[%s] Final Response Sent to User: %s", conversation_id, final_content)
            _emit(event_callback, "status", stage="complete", message="Response ready.")
            
            return final_content