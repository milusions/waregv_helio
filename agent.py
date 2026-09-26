import asyncio
import json
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional

import aiohttp
import yaml
from openai import OpenAI

from configs.tools import handle_tool

# --- Patch missing aiohttp attribute before importing openai ---
if not hasattr(aiohttp, "SocketTimeoutError"):
    aiohttp.SocketTimeoutError = getattr(
        aiohttp, 
        "ServerTimeoutError", 
        asyncio.TimeoutError if "asyncio" in sys.modules else Exception
    )

# --- ANSI Colors for Beautiful Terminal ---
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
DIM = "\033[2m"

LOG_PATH = os.environ.get("AGENT_LOG_PATH", "log/agent.log")


class BeautifulConsoleFormatter(logging.Formatter):
    """Formats terminal output to be beautiful and conversational while ignoring raw log syntax."""
    
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        prefix = f"{DIM}[{time_str}]{RESET} "

        if msg.startswith("--- NEW REQUEST ---"):
            parts = msg.split(" | ")
            if len(parts) >= 4:
                conv_id = parts[2].replace("Conversation ID: ", "")
                user_msg = parts[3].replace("User Message: ", "")
                return f"\n{BOLD}{CYAN}╭─── Conversation ID: {conv_id} ───╮{RESET}\n{prefix}{BOLD}{GREEN}Input:{RESET} {user_msg}"
        
        if "] New Helio conversation created." in msg:
            return f"{prefix}{YELLOW}✨ New conversation initialized. ✨{RESET}"
            
        if "Final Response Sent to User:" in msg:
            parts = msg.split("Final Response Sent to User: ")
            response = parts[1] if len(parts) > 1 else ""
            return f"{prefix}{BOLD}{MAGENTA}Output:{RESET} {response}\n{BOLD}{CYAN}╰────────────────────────────────────────╯{RESET}"
            
        if "] Tool Execution:" in msg:
            parts = msg.split("] Tool Execution: ")
            return f"{prefix}{DIM}⚙️ Executing tool: {parts[1]}{RESET}"

        if "Model output:" in msg:
            return f"{prefix}{DIM}Processing model output...{RESET}"

        return f"{prefix}{msg}"


# --- Initialize Logging System ---
logger = logging.getLogger("AgentLogger")
logger.setLevel(logging.INFO)
logger.propagate = False
os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

if not logger.handlers:
    # File handling (Raw formats for log files)
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(file_handler)
    
    # Stream handling (Beautiful CLI view)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(BeautifulConsoleFormatter())
    logger.addHandler(console_handler)


# --- Configuration & Client Initialization ---
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "configs", "config.yaml")

try:
    with open(config_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file) or {}
except Exception as exc:
    logger.error("Failed to load configuration file at %s: %s", config_path, exc)
    config_data = {}

SYSTEM_PROMPT = config_data.get("agent_instructions", "You are Helio.")
TOOLS_SCHEMA = config_data.get("tool_description", [])

# Initialize OpenAI client targeting the Gemini API endpoint
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Active, persistent chat memories keyed by conversation_id
sessions: dict[str, list[dict[str, Any]]] = {}


def _emit(event_callback: Optional[Callable[[dict[str, Any]], None]], event_type: str, **payload: Any) -> None:
    """Safely triggers an upstream callback with standard format payloads."""
    if event_callback:
        event_callback({"type": event_type, **payload})


def _process_tool_calls(
    tool_calls: list[Any], 
    history: list[dict[str, Any]], 
    conversation_id: str, 
    event_callback: Optional[Callable[[dict[str, Any]], None]]
) -> None:
    """Extracts, logs, triggers, and saves results of LLM requested tools."""
    tool_calls_payload = []
    
    # Pre-parse structure to build the assistant message in history
    for tc in tool_calls:
        tool_calls_payload.append({
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
        })
        
    history.append({
        "role": "assistant",
        "content": None,  # Explicit None when function payloads are present
        "tool_calls": tool_calls_payload
    })
    
    # Sequentially execute requested tools
    for tool_call in tool_calls:
        func_name = tool_call.function.name
        
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}
            
        logger.info("[%s] Tool Execution: %s(args=%s)", conversation_id, func_name, args)
        _emit(event_callback, "status", stage="tool", message=f"Accessing system data: {func_name}")
        
        # Core utility caller
        result = handle_tool(func_name, args)
        result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        
        history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": func_name,
            "content": result_str
        })


def process_agent_request(
    user_message: str, 
    conversation_id: str, 
    event_callback: Optional[Callable[[dict[str, Any]], None]] = None
) -> str:
    """Persistent agent loop handling both text generation and recursive tool execution."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(
        "--- NEW REQUEST --- | Time: %s | Conversation ID: %s | User Message: %s",
        current_time, conversation_id, user_message,
    )

    # Initialize session history tracking if absent
    if conversation_id not in sessions:
        sessions[conversation_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("[%s] New Helio conversation created.", conversation_id)

    history = sessions[conversation_id]
    history.append({"role": "user", "content": user_message})
    _emit(event_callback, "status", stage="thinking", message="Helio is processing the request.")

    while True:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=history, # type: ignore
            temperature=0.0,
            tools=TOOLS_SCHEMA if TOOLS_SCHEMA else None,
        )
        
        response_message = response.choices[0].message
        
        # Scenario A: The model asks for data tools
        if response_message.tool_calls:
            _process_tool_calls(response_message.tool_calls, history, conversation_id, event_callback)
            # Continues while-loop to send outputs back to model
            continue
            
        # Scenario B: The model produced its finalized text response
        final_content = response_message.content or "I have processed your request, but no text response was returned."
        
        logger.info("[%s] Model output: %s", conversation_id, final_content)
        _emit(event_callback, "model", stage="model", message=final_content)

        history.append({"role": "assistant", "content": final_content})

        logger.info("[%s] Final Response Sent to User: %s", conversation_id, final_content)
        _emit(event_callback, "status", stage="complete", message="Response ready.")
        
        return final_content