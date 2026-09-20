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

    while True:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=history,
            tools=tools,
            temperature=0.0,
        )
        response_message = response.choices[0].message
        model_text = response_message.content or ""

        if model_text:
            logger.info("[%s] Model output: %s", conversation_id, model_text)
            _emit(event_callback, "model", stage="model", message=model_text)

        if not response_message.tool_calls:
            history.append(response_message)
            final_content = response_message.content or "I have processed your request, but no text response was returned."
            logger.info("[%s] Final Response Sent to User: %s", conversation_id, final_content)
            _emit(event_callback, "status", stage="complete", message="Response ready.")
            return final_content

        history.append(response_message)

        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            logger.info("[%s] Tool Call Triggered -> Function: %s | Inputs: %s", conversation_id, func_name, args)
            _emit(event_callback, "tool_call", stage="tool_call", tool=func_name, inputs=args, message=f"Calling {func_name}")
            _emit(event_callback, "status", stage="tool_running", message=f"Running {func_name}...")

            try:
                tool_result = handle_tool(func_name, args)
                tool_error = False
            except Exception as exc:
                tool_result = {"error": str(exc)}
                tool_error = True
                logger.exception("[%s] Tool failed: %s", conversation_id, func_name)

            result_text = str(tool_result)
            logger.info("[%s] Tool Result <- Function: %s | Result: %s", conversation_id, func_name, result_text)
            _emit(
                event_callback,
                "tool_result",
                stage="tool_result",
                tool=func_name,
                result=tool_result,
                success=not tool_error,
                message=f"{func_name} returned a result.",
            )

            # This internal tool_call_id is required by the model API. It is
            # not the browser's call/session identifier and is never surfaced.
            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": result_text,
            })

        _emit(event_callback, "status", stage="thinking", message="Helio is evaluating the tool result.")
