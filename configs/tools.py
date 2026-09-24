import os
import json
import logging
from typing import Optional, Dict, Any, List

import yaml
import requests

CONFIG_PATH = "configs/config.yaml"
CHECKPOINTS_PATH = "configs/checkpoints.yaml"
LEARNING_PATH = "configs/learning.yaml"
LOG_PATH = "log/tools.log"

logger = logging.getLogger("RoverTools")
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


def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def call_rest_api(url: str, method: str, payload: Optional[dict] = None, params: Optional[dict] = None) -> str:
    try:
        method = method.upper()
        if method == "GET":
            r = requests.get(url, params=params, timeout=10)
        else:
            r = requests.request(method, url, json=payload, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"API Call Failed: {e}"


def _api(method: str, path: str, payload: Optional[dict] = None, params: Optional[dict] = None) -> Any:
    config = _load_yaml(CONFIG_PATH)
    base = config.get("rest_base_url", "http://localhost:8000").rstrip("/")
    raw = call_rest_api(base + path, method, payload, params)
    try:
        return json.loads(raw)
    except Exception:
        return raw


# ---------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------
def get_checkpoints() -> dict:
    try:
        data = _load_yaml(CHECKPOINTS_PATH)
        return data.get("checkpoints", {}) or {}
    except Exception as e:
        logger.error("get_checkpoints: %s", e)
        return {}


def set_checkpoint(label: str, x: float, y: float, yaw_deg: float = 0.0) -> str:
    try:
        data = _load_yaml(CHECKPOINTS_PATH)
        locations = data.get("checkpoints", {})
        if locations is None:
            locations = {}
        locations[label] = {"x": float(x), "y": float(y), "yaw_deg": float(yaw_deg)}
        data["checkpoints"] = locations
        _save_yaml(CHECKPOINTS_PATH, data)
        logger.info("Saved checkpoint '%s' at x=%s, y=%s, yaw_deg=%s", label, x, y, yaw_deg)
        return f"Successfully saved checkpoint location '{label}'."
    except Exception as e:
        logger.error("set_checkpoint: %s", e)
        return f"Failed to save checkpoint location: {e}"


def navigate_to_checkpoint(label: str) -> Any:
    checkpoints = get_checkpoints()
    if label not in checkpoints:
        return {
            "status": "error",
            "message": f"Checkpoint '{label}' not found.",
            "available_checkpoints": list(checkpoints.keys())
        }
    p = checkpoints[label]
    return navigate_to_pose(p["x"], p["y"], p.get("yaw_deg", 0.0))


# ---------------------------------------------------------
# Online Learning Management (learning.yaml)
# ---------------------------------------------------------
def get_learned_skills() -> dict:
    try:
        data = _load_yaml(LEARNING_PATH)
        return data.get("skills", {}) or {}
    except Exception as e:
        logger.error("get_learned_skills: %s", e)
        return {}


def save_learned_skill(skill_name: str, description: str, steps: List[str]) -> str:
    try:
        data = _load_yaml(LEARNING_PATH)
        skills = data.get("skills", {})
        if skills is None:
            skills = {}
        skills[skill_name] = {
            "description": description,
            "steps": steps
        }
        data["skills"] = skills
        _save_yaml(LEARNING_PATH, data)
        logger.info("Saved new learned skill '%s'", skill_name)
        return f"Successfully saved learned skill '{skill_name}' to learning.yaml."
    except Exception as e:
        logger.error("save_learned_skill: %s", e)
        return f"Failed to save learned skill: {e}"


# ---------------------------------------------------------
# Rover REST API Tools
# ---------------------------------------------------------
def get_system_mode() -> Any:
    return _api("GET", "/system/mode")


def set_system_mode(mode: str, map_name: str = "small_warehouse") -> Any:
    return _api("POST", "/system/mode", {"mode": mode, "map_name": map_name})


def set_initial_pose(x: float, y: float, yaw_deg: float = 0.0) -> Any:
    return _api("POST", "/set_initial_pose", {"x": x, "y": y, "yaw_deg": yaw_deg})


def navigate_to_pose(x: float, y: float, yaw_deg: float = 0.0) -> Any:
    return _api("POST", "/navigate_to_pose", {"x": x, "y": y, "yaw_deg": yaw_deg})


def follow_waypoints(waypoints: List[dict]) -> Any:
    return _api("POST", "/follow_waypoints", {"waypoints": waypoints})


def abort_mission() -> Any:
    return _api("POST", "/abort", {})


def save_map(name: str = "map") -> Any:
    return _api("GET", "/map/save", params={"name": name})


def get_robot_pose() -> Any:
    return _api("GET", "/amcl_pose")


# ---------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------
def handle_tool(func_name, args):
    logger.info("Calling tool '%s' with args: %s", func_name, args)
    
    dispatch = {
        "get_checkpoints": get_checkpoints,
        "set_checkpoint": set_checkpoint,
        "navigate_to_checkpoint": navigate_to_checkpoint,
        "get_learned_skills": get_learned_skills,
        "save_learned_skill": save_learned_skill,
        "get_system_mode": get_system_mode,
        "set_system_mode": set_system_mode,
        "set_initial_pose": set_initial_pose,
        "navigate_to_pose": navigate_to_pose,
        "follow_waypoints": follow_waypoints,
        "abort_mission": abort_mission,
        "save_map": save_map,
        "get_robot_pose": get_robot_pose,
    }
    if func_name not in dispatch:
        logger.error("Unknown tool requested: %s", func_name)
        return f"Unknown tool: {func_name}"
    try:
        result = dispatch[func_name](**args)
        logger.info("Tool '%s' executed successfully.", func_name)
        return result
    except TypeError as e:
        logger.error("Invalid arguments for %s: %s", func_name, e)
        return f"Invalid arguments for {func_name}: {e}"
    except Exception as e:
        logger.exception("Tool failed: %s", func_name)
        return f"Tool failed: {e}"