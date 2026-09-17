import os
import logging
import yaml
import requests
from typing import Optional, Dict, Any

CONFIG_PATH = "configs/config.yaml"
LOG_PATH = "log/tools.log"

# Create a dedicated, isolated logger for your tools
logger = logging.getLogger("RoverTools")
logger.setLevel(logging.INFO)

# CRITICAL: Stop logs from propagating to the global root logger 
# (This prevents watchfiles and uvicorn logs from bleeding into your file)
logger.propagate = False

# Add a file handler specifically for tools.log (avoiding duplicate handlers on reload)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def _load_config() -> dict:
    """Helper to load the current config.yaml file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def _save_config(config_data: dict) -> None:
    """Helper to save changes back to config.yaml."""
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config_data, f, sort_keys=False)

def get_favourites() -> dict:
    """Retrieves all favorite locations stored in config.yaml."""
    logger.info("TOOL_CALL: get_favourites invoked.")
    try:
        config = _load_config()
        favs = config.get("favorite_locations", {}) or {}
        logger.info(f"TOOL_SUCCESS: get_favourites output -> {favs}")
        return favs
    except Exception as e:
        err_msg = f"Failed to retrieve favorites: {str(e)}"
        logger.error(f"TOOL_ERROR: get_favourites failed -> {err_msg}")
        return {}

def set_favourite(label: str, x: float, y: float, yaw_w: float = 1.0) -> str:
    """Adds or updates a favorite location in config.yaml."""
    logger.info(f"TOOL_CALL: set_favourite invoked with input -> label: {label}, x: {x}, y: {y}, yaw_w: {yaw_w}")
    try:
        config = _load_config()
        if "favorite_locations" not in config or config["favorite_locations"] is None:
            config["favorite_locations"] = {}
            
        config["favorite_locations"][label] = {
            "x": float(x),
            "y": float(y),
            "yaw_w": float(yaw_w)
        }
        
        _save_config(config)
        result = f"Successfully saved favorite location '{label}'."
        logger.info(f"TOOL_SUCCESS: set_favourite output -> {result}")
        return result
    except Exception as e:
        err_msg = f"Failed to save favorite location: {str(e)}"
        logger.error(f"TOOL_ERROR: set_favourite failed -> {err_msg}")
        return err_msg

def call_rest_api(url: str, method: str, payload: Optional[dict] = None) -> str:
    """Makes an HTTP request to an external REST API."""
    logger.info(f"TOOL_CALL: call_rest_api invoked with input -> url: {url}, method: {method}, payload: {payload}")
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=payload, timeout=10)
        else:
            response = requests.request(method.upper(), url, json=payload, timeout=10)
            
        response.raise_for_status()
        output = response.text
        logger.info(f"TOOL_SUCCESS: call_rest_api output -> Status {response.status_code}, Response: {output}")
        return output
    except Exception as e:
        err_msg = f"API Call Failed: {str(e)}"
        logger.error(f"TOOL_ERROR: call_rest_api failed -> {err_msg}")
        return err_msg