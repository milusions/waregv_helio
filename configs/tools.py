import os
import json
import logging
from typing import Optional, Dict, Any, List

import yaml
import requests

CONFIG_PATH = "configs/config.yaml"
CHECKPOINTS_PATH = "configs/checkpoints.yaml"
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
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_checkpoints() -> dict:
    try:
        data = _load_yaml(CHECKPOINTS_PATH)
        return data.get("checkpoints", data.get("checkpoint_locations", {})) or {}
    except Exception as e:
        logger.error("get_checkpoints: %s", e)
        return {}


def set_checkpoint(label: str, x: float, y: float, yaw_w: float = 1.0) -> str:
    try:
        data = _load_yaml(CHECKPOINTS_PATH)
        locations = data.get("checkpoints")
        if locations is None:
            locations = data.get("checkpoint_locations", {})
        if locations is None:
            locations = {}
        locations[label] = {"x": float(x), "y": float(y), "yaw_w": float(yaw_w)}
        data["checkpoints"] = locations
        data.pop("checkpoint_locations", None)
        _save_yaml(CHECKPOINTS_PATH, data)
        return f"Successfully saved checkpoint location '{label}'."
    except Exception as e:
        logger.error("set_checkpoint: %s", e)
        return f"Failed to save checkpoint location: {e}"


def call_rest_api(url: str, method: str, payload: Optional[dict] = None) -> str:
    try:
        method = method.upper()
        if method == "GET":
            r = requests.get(url, params=payload, timeout=10)
        else:
            r = requests.request(method, url, json=payload, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"API Call Failed: {e}"


def _api(method: str, path: str, payload: Optional[dict] = None) -> Any:
    config = _load_yaml(CONFIG_PATH)
    base = config.get("rest_base_url", "http://localhost:8000").rstrip("/")
    raw = call_rest_api(base + path, method, payload)
    try:
        return json.loads(raw)
    except Exception:
        return raw


def navigate_to_pose(x: float, y: float, yaw_deg: float = 0.0) -> Any:
    return _api("POST", "/navigate_to_pose", {"x": x, "y": y, "yaw_deg": yaw_deg})


def navigate_to_checkpoint(label: str) -> Any:
    checkpoints = get_checkpoints()
    if label not in checkpoints:
        return {"status": "error", "message": f"Checkpoint '{label}' not found",
                "available_checkpoints": list(checkpoints.keys())}
    p = checkpoints[label]
    return navigate_to_pose(p["x"], p["y"], p.get("yaw_deg", 0.0))


def set_initial_pose(x: float, y: float, yaw_deg: float = 0.0) -> Any:
    return _api("POST", "/set_initial_pose", {"x": x, "y": y, "yaw_deg": yaw_deg})


def abort_mission() -> Any:
    return _api("POST", "/abort", {})


def get_system_mode() -> Any:
    return _api("GET", "/system/mode")


def set_system_mode(mode: str, map_name: str = "small_warehouse") -> Any:
    return _api("POST", "/system/mode", {"mode": mode, "map_name": map_name})


def load_map(map_name: str) -> Any:
    return _api("POST", "/system/mode/load_map", {"map_name": map_name})


def save_map(map_name: str) -> Any:
    return _api("POST", "/system/save_map", {"map_name": map_name})


def start_slam_update(map_name: str) -> Any:
    return _api("POST", "/system/slam_update/load", {"map_name": map_name})


def follow_waypoints(waypoints: List[dict]) -> Any:
    return _api("POST", "/follow_waypoints", {"waypoints": waypoints})


def get_robot_pose() -> Any:
    return _api("GET", "/robot/pose")


def get_navigation_status() -> Any:
    return _api("GET", "/navigation/status")


def get_distance_to_goal() -> Any:
    return _api("GET", "/navigation/status")


def get_map_info() -> Any:
    return _api("GET", "/map/info")


def get_navigation_plan() -> Any:
    return _api("GET", "/navigation/plan")


def get_odom() -> Any:
    return _api("GET", "/telemetry/odom")


def get_wheel_states() -> Any:
    return _api("GET", "/telemetry/wheels")


def get_rover_status() -> Any:
    return _api("GET", "/rover/status")


def get_available_maps() -> Any:
    return _api("GET", "/maps")


def manual_drive(linear: float, angular: float) -> Any:
    return _api("POST", "/manual_drive", {"linear": linear, "angular": angular})


def stop_manual_drive() -> Any:
    return _api("POST", "/manual_drive", {"linear": 0.0, "angular": 0.0})


def get_slam_update_status() -> Any:
    return _api("GET", "/slam/status")


def handle_tool(func_name, args):
    logger.info("Calling tool '%s' with args: %s", func_name, args)
    
    dispatch = {
        "call_rest_api": call_rest_api,
        "get_checkpoints": get_checkpoints,
        "set_checkpoint": set_checkpoint,
        "navigate_to_pose": navigate_to_pose,
        "navigate_to_checkpoint": navigate_to_checkpoint,
        "set_initial_pose": set_initial_pose,
        "abort_mission": abort_mission,
        "get_system_mode": get_system_mode,
        "set_system_mode": set_system_mode,
        "load_map": load_map,
        "save_map": save_map,
        "start_slam_update": start_slam_update,
        "follow_waypoints": follow_waypoints,
        "get_robot_pose": get_robot_pose,
        "get_navigation_status": get_navigation_status,
        "get_distance_to_goal": get_distance_to_goal,
        "get_map_info": get_map_info,
        "get_navigation_plan": get_navigation_plan,
        "get_odom": get_odom,
        "get_wheel_states": get_wheel_states,
        "get_rover_status": get_rover_status,
        "get_available_maps": get_available_maps,
        "manual_drive": manual_drive,
        "stop_manual_drive": stop_manual_drive,
        "get_slam_update_status": get_slam_update_status,
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