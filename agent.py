import os
import yaml
import logging
from datetime import datetime
from typing import Optional, Dict, List
from openai import OpenAI


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
    """Persistent agent loop. Browser identity is conversation_id only."""
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

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=history,
        temperature=0.0,
    )
    
    response_message = response.choices[0].message
    final_content = response_message.content or "I have processed your request, but no text response was returned."
    
    logger.info("[%s] Model output: %s", conversation_id, final_content)
    _emit(event_callback, "model", stage="model", message=final_content)

    # Append assistant response to preserve conversation history
    history.append({"role": "assistant", "content": final_content})

    logger.info("[%s] Final Response Sent to User: %s", conversation_id, final_content)
    _emit(event_callback, "status", stage="complete", message="Response ready.")
    
    return final_content