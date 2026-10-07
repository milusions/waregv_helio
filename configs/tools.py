import os
import json
import math
import logging
from typing import Optional, Dict, Any, List

import yaml
import requests

CONFIG_PATH = "configs/config.yaml"
CHECKPOINTS_PATH = "configs/checkpoints.yaml"
LEARNING_PATH = "configs/learning.yaml"
WAYPOINTS_PATH = "configs/waypoints.yaml"
HOME_PATH = "configs/home.yaml"
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


# =========================================================
# YAML helpers
# =========================================================
def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
        return {}


def _save_yaml(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# =========================================================
# Low-level REST plumbing
# =========================================================
def call_rest_api(
    url: str,
    method: str,
    payload: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: int = 15,
) -> str:
    try:
        method = method.upper()
        if method == "GET":
            r = requests.get(url, params=params, timeout=timeout)
        else:
            r = requests.request(method, url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"API Call Failed: {e}"


def _api(
    method: str,
    path: str,
    payload: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: int = 15,
) -> Any:
    config = _load_yaml(CONFIG_PATH)
    base = config.get("rest_base_url", "http://localhost:8000").rstrip("/")
    raw = call_rest_api(base + path, method, payload, params, timeout=timeout)
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _api_safe(result: Any) -> Any:
    """Normalize responses so the LLM always sees structured data."""
    if isinstance(result, str) and result.startswith("API Call Failed"):
        return {"ok": False, "error": result}
    return result


# =========================================================
# Geometry helpers (used for teaching skills)
# =========================================================
def _normalize_angle_deg(angle: float) -> float:
    """Wrap any angle into (-180, 180]."""
    a = math.fmod(angle + 180.0, 360.0)
    if a < 0:
        a += 360.0
    return a - 180.0


def compute_pose_offset(
    x: float,
    y: float,
    yaw_deg: float,
    forward_m: float = 0.0,
    right_m: float = 0.0,
    yaw_delta_deg: float = 0.0,
) -> dict:
    """
    Compute a new pose given a motion relative to the current pose.

    forward_m      : move forward along current heading (+x of body)
    right_m        : move to the robot's right (+y of body)
    yaw_delta_deg  : rotate in place (positive = counter-clockwise)
    """
    yaw_rad = math.radians(yaw_deg)

    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    dx = forward_m * cos_y - right_m * sin_y
    dy = forward_m * sin_y + right_m * cos_y

    new_x = x + dx
    new_y = y + dy
    new_yaw = _normalize_angle_deg(yaw_deg + yaw_delta_deg)

    return {
        "x": round(new_x, 3),
        "y": round(new_y, 3),
        "yaw_deg": round(new_yaw, 3),
    }


def _get_robot_pose_dict() -> Optional[dict]:
    """Fetch the rover's current pose as a plain dict, or None on failure."""
    pose = get_robot_pose()
    if isinstance(pose, dict) and pose.get("ok") and pose.get("pose"):
        p = pose["pose"]
        return {
            "x": float(p["position"]["x"]),
            "y": float(p["position"]["y"]),
            "yaw_deg": float(p.get("yaw_deg", 0.0)),
        }
    return None


# =========================================================
# Checkpoint Management (checkpoints.yaml)
# =========================================================
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
        locations = data.get("checkpoints", {}) or {}
        locations[label] = {"x": float(x), "y": float(y), "yaw_deg": float(yaw_deg)}
        data["checkpoints"] = locations
        _save_yaml(CHECKPOINTS_PATH, data)
        logger.info("Saved checkpoint '%s' at (%s, %s, %s)", label, x, y, yaw_deg)
        return f"Successfully saved checkpoint location '{label}'."
    except Exception as e:
        logger.error("set_checkpoint: %s", e)
        return f"Failed to save checkpoint location: {e}"


def set_checkpoint_here(label: str) -> str:
    """Remember the current rover pose under a label."""
    pose = _get_robot_pose_dict()
    if pose is None:
        return "Cannot save checkpoint: robot pose is unavailable."
    return set_checkpoint(label, pose["x"], pose["y"], pose["yaw_deg"])


def delete_checkpoint(label: str) -> str:
    try:
        data = _load_yaml(CHECKPOINTS_PATH)
        locations = data.get("checkpoints", {}) or {}
        if label not in locations:
            return f"Checkpoint '{label}' does not exist."
        del locations[label]
        data["checkpoints"] = locations
        _save_yaml(CHECKPOINTS_PATH, data)
        return f"Deleted checkpoint '{label}'."
    except Exception as e:
        return f"Failed to delete checkpoint: {e}"


def navigate_to_checkpoint(label: str) -> Any:
    checkpoints = get_checkpoints()
    if label not in checkpoints:
        return {
            "status": "error",
            "message": f"Checkpoint '{label}' not found.",
            "available_checkpoints": list(checkpoints.keys()),
        }
    p = checkpoints[label]
    return navigate_to_pose(p["x"], p["y"], p.get("yaw_deg", 0.0))


# =========================================================
# Home Position (home.yaml)
# =========================================================
def set_home_here(label: str = "home") -> str:
    """Persist the current rover pose as the home location."""
    pose = _get_robot_pose_dict()
    if pose is None:
        return "Cannot set home: robot pose is unavailable."
    try:
        data = _load_yaml(HOME_PATH)
        data["home"] = {
            "label": label,
            "x": pose["x"],
            "y": pose["y"],
            "yaw_deg": pose["yaw_deg"],
        }
        _save_yaml(HOME_PATH, data)
        return f"Home position saved at ({pose['x']}, {pose['y']}, {pose['yaw_deg']}°)."
    except Exception as e:
        return f"Failed to set home: {e}"


def get_home() -> dict:
    return _load_yaml(HOME_PATH).get("home", {}) or {}


def return_to_home() -> Any:
    home = get_home()
    if not home:
        return {
            "status": "error",
            "message": "No home position has been set. Use set_home_here first.",
        }
    return navigate_to_pose(home["x"], home["y"], home.get("yaw_deg", 0.0))


# =========================================================
# Waypoint Library (waypoints.yaml)
# =========================================================
def get_waypoint_routes() -> dict:
    return _load_yaml(WAYPOINTS_PATH).get("routes", {}) or {}


def save_waypoint_route(route_name: str, waypoints: List[dict]) -> str:
    """
    Save a named ordered route. Each waypoint must contain x, y,
    and optionally yaw_deg.
    """
    try:
        if not waypoints:
            return "Cannot save an empty route."
        normalized = []
        for wp in waypoints:
            normalized.append({
                "x": float(wp["x"]),
                "y": float(wp["y"]),
                "yaw_deg": float(wp.get("yaw_deg", 0.0)),
            })
        data = _load_yaml(WAYPOINTS_PATH)
        routes = data.get("routes", {}) or {}
        routes[route_name] = normalized
        data["routes"] = routes
        _save_yaml(WAYPOINTS_PATH, data)
        return f"Saved route '{route_name}' with {len(normalized)} waypoints."
    except Exception as e:
        return f"Failed to save route: {e}"


def delete_waypoint_route(route_name: str) -> str:
    data = _load_yaml(WAYPOINTS_PATH)
    routes = data.get("routes", {}) or {}
    if route_name not in routes:
        return f"Route '{route_name}' not found."
    del routes[route_name]
    data["routes"] = routes
    _save_yaml(WAYPOINTS_PATH, data)
    return f"Deleted route '{route_name}'."


def follow_saved_route(route_name: str) -> Any:
    routes = get_waypoint_routes()
    if route_name not in routes:
        return {
            "status": "error",
            "message": f"Route '{route_name}' not found.",
            "available_routes": list(routes.keys()),
        }
    return follow_waypoints(routes[route_name])


# =========================================================
# Online Learning Management (learning.yaml)
# =========================================================
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
        skills = data.get("skills", {}) or {}
        skills[skill_name] = {
            "description": description,
            "steps": steps,
        }
        data["skills"] = skills
        _save_yaml(LEARNING_PATH, data)
        logger.info("Saved new learned skill '%s'", skill_name)
        return f"Successfully saved learned skill '{skill_name}' to learning.yaml."
    except Exception as e:
        logger.error("save_learned_skill: %s", e)
        return f"Failed to save learned skill: {e}"


def delete_learned_skill(skill_name: str) -> str:
    data = _load_yaml(LEARNING_PATH)
    skills = data.get("skills", {}) or {}
    if skill_name not in skills:
        return f"Skill '{skill_name}' not found."
    del skills[skill_name]
    data["skills"] = skills
    _save_yaml(LEARNING_PATH, data)
    return f"Deleted learned skill '{skill_name}'."


# =========================================================
# Rover REST API Tools (direct mirror of the backend)
# =========================================================
def get_system_mode() -> Any:
    return _api_safe(_api("GET", "/system/mode"))


def set_system_mode(mode: str, map_name: str = "small_warehouse") -> Any:
    return _api_safe(_api("POST", "/system/mode", {"mode": mode, "map_name": map_name}))


def set_initial_pose(x: float, y: float, yaw_deg: float = 0.0) -> Any:
    return _api_safe(_api("POST", "/set_initial_pose", {"x": x, "y": y, "yaw_deg": yaw_deg}))


def navigate_to_pose(x: float, y: float, yaw_deg: float = 0.0) -> Any:
    """
    Dispatch a Nav2 goal. Returns IMMEDIATELY with:
        {"ok": true, "status": "dispatched", "goal_id": "..."}
    The rover keeps moving in the background. Poll with get_goal_status().
    """
    return _api_safe(_api("POST", "/navigate_to_pose", {"x": x, "y": y, "yaw_deg": yaw_deg}))


def follow_waypoints(waypoints: List[dict]) -> Any:
    """
    Dispatch a FollowWaypoints goal. Returns IMMEDIATELY with a goal_id.
    """
    return _api_safe(_api("POST", "/follow_waypoints", {"waypoints": waypoints}))


def get_goal_status(goal_id: str) -> Any:
    """
    Poll the state of a previously dispatched goal.
    States: dispatched | executing | succeeded | failed | canceled |
            aborted | rejected.
    """
    return _api_safe(_api("GET", f"/goal/{goal_id}"))


def list_goals() -> Any:
    """Return all active + recent goals with their states."""
    return _api_safe(_api("GET", "/goals"))


def abort_mission() -> Any:
    return _api_safe(_api("POST", "/abort", {}))


def save_map(name: str = "map") -> Any:
    """
    Trigger a download of an existing map as a .zip.
    Returns a status descriptor; the actual binary is served via the UI.
    """
    try:
        config = _load_yaml(CONFIG_PATH)
        base = config.get("rest_base_url", "http://localhost:8000").rstrip("/")
        r = requests.get(f"{base}/map/save", params={"name": name}, timeout=30)
        if r.status_code == 200:
            return {
                "ok": True,
                "name": name,
                "size_bytes": len(r.content),
                "content_type": r.headers.get("Content-Type", ""),
            }
        return {"ok": False, "status": r.status_code, "detail": r.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def list_maps() -> Any:
    return _api_safe(_api("GET", "/maps"))


def get_map_info(map_name: str) -> Any:
    return _api_safe(_api("GET", f"/maps/{map_name}"))


def map_exists(map_name: str) -> Any:
    return _api_safe(_api("GET", "/map/exists", params={"name": map_name}))


def save_map_to_disk(
    name: str,
    pgm_bytes: bytes,
    yaml_bytes: Optional[bytes] = None,
    overwrite: bool = False,
) -> Any:
    """
    Upload a PGM (+ optional YAML) to the backend and persist it as:
        maps/<name>/<name>.pgm
        maps/<name>/<name>.yaml
    """
    try:
        config = _load_yaml(CONFIG_PATH)
        base = config.get("rest_base_url", "http://localhost:8000").rstrip("/")
        files = {"pgm": (f"{name}.pgm", pgm_bytes, "application/octet-stream")}
        if yaml_bytes is not None:
            files["yaml"] = (f"{name}.yaml", yaml_bytes, "text/yaml")
        data = {"name": name, "overwrite": str(overwrite).lower()}
        r = requests.post(
            f"{base}/map/save_to_disk",
            data=data,
            files=files,
            timeout=60,
        )
        try:
            return r.json()
        except Exception:
            return {"ok": r.status_code == 200, "status": r.status_code, "detail": r.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def load_map(
    name: str,
    pgm_bytes: bytes,
    yaml_bytes: Optional[bytes] = None,
    overwrite: bool = False,
) -> Any:
    """Alias of save_map_to_disk via the /map/load endpoint."""
    try:
        config = _load_yaml(CONFIG_PATH)
        base = config.get("rest_base_url", "http://localhost:8000").rstrip("/")
        files = {"pgm": (f"{name}.pgm", pgm_bytes, "application/octet-stream")}
        if yaml_bytes is not None:
            files["yaml"] = (f"{name}.yaml", yaml_bytes, "text/yaml")
        data = {"name": name, "overwrite": str(overwrite).lower()}
        r = requests.post(
            f"{base}/map/load",
            data=data,
            files=files,
            timeout=60,
        )
        try:
            return r.json()
        except Exception:
            return {"ok": r.status_code == 200, "status": r.status_code, "detail": r.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_robot_pose() -> Any:
    return _api_safe(_api("GET", "/amcl_pose"))


# =========================================================
# High-level motion primitives
# =========================================================
def move_relative(
    forward_m: float = 0.0,
    right_m: float = 0.0,
    yaw_delta_deg: float = 0.0,
) -> Any:
    """
    Move relative to the robot's CURRENT pose.

    forward_m     : + is forward, - is backward (meters)
    right_m       : + is right,   - is left     (meters)
    yaw_delta_deg : + is counter-clockwise      (degrees)

    Returns the SAME shape as navigate_to_pose: a goal_id + status.
    """
    pose = _get_robot_pose_dict()
    if pose is None:
        return {
            "status": "error",
            "message": "Robot pose unavailable; cannot compute relative move.",
        }
    target = compute_pose_offset(
        pose["x"], pose["y"], pose["yaw_deg"],
        forward_m=forward_m,
        right_m=right_m,
        yaw_delta_deg=yaw_delta_deg,
    )
    logger.info(
        "move_relative: from (%s,%s,%s) -> (%s,%s,%s)",
        pose["x"], pose["y"], pose["yaw_deg"],
        target["x"], target["y"], target["yaw_deg"],
    )
    return navigate_to_pose(target["x"], target["y"], target["yaw_deg"])


def move_forward(distance_m: float) -> Any:
    return move_relative(forward_m=distance_m)


def move_backward(distance_m: float) -> Any:
    return move_relative(forward_m=-abs(distance_m))


def strafe_left(distance_m: float) -> Any:
    return move_relative(right_m=-abs(distance_m))


def strafe_right(distance_m: float) -> Any:
    return move_relative(right_m=abs(distance_m))


def rotate_in_place(yaw_delta_deg: float) -> Any:
    return move_relative(yaw_delta_deg=yaw_delta_deg)


def turn_left(angle_deg: float = 90.0) -> Any:
    return move_relative(yaw_delta_deg=abs(angle_deg))


def turn_right(angle_deg: float = 90.0) -> Any:
    return move_relative(yaw_delta_deg=-abs(angle_deg))


# =========================================================
# Utility / diagnostics
# =========================================================
def describe_rover_state() -> Any:
    """
    Aggregate read-only status: pose, mode, home, checkpoints,
    routes, learned skills, and currently active goals.
    """
    return {
        "pose": get_robot_pose(),
        "mode": get_system_mode(),
        "home": get_home(),
        "checkpoints": list(get_checkpoints().keys()),
        "routes": list(get_waypoint_routes().keys()),
        "skills": list(get_learned_skills().keys()),
        "goals": list_goals(),
    }


def stop_all() -> Any:
    """Immediate abort of all active missions."""
    return abort_mission()


# =========================================================
# Tool Dispatcher
# =========================================================
def handle_tool(func_name: str, args: Optional[dict] = None):
    args = args or {}
    logger.info("Calling tool '%s' with args: %s", func_name, args)

    dispatch = {
        # --- Checkpoints ---
        "get_checkpoints": get_checkpoints,
        "set_checkpoint": set_checkpoint,
        "set_checkpoint_here": set_checkpoint_here,
        "delete_checkpoint": delete_checkpoint,
        "navigate_to_checkpoint": navigate_to_checkpoint,

        # --- Home ---
        "set_home_here": set_home_here,
        "get_home": get_home,
        "return_to_home": return_to_home,

        # --- Waypoint routes ---
        "get_waypoint_routes": get_waypoint_routes,
        "save_waypoint_route": save_waypoint_route,
        "delete_waypoint_route": delete_waypoint_route,
        "follow_saved_route": follow_saved_route,

        # --- Learning ---
        "get_learned_skills": get_learned_skills,
        "save_learned_skill": save_learned_skill,
        "delete_learned_skill": delete_learned_skill,

        # --- System / pose ---
        "get_system_mode": get_system_mode,
        "set_system_mode": set_system_mode,
        "set_initial_pose": set_initial_pose,
        "get_robot_pose": get_robot_pose,

        # --- Navigation (non-blocking) ---
        "navigate_to_pose": navigate_to_pose,
        "follow_waypoints": follow_waypoints,
        "get_goal_status": get_goal_status,
        "list_goals": list_goals,
        "abort_mission": abort_mission,
        "stop_all": stop_all,

        # --- Motion primitives ---
        "move_relative": move_relative,
        "move_forward": move_forward,
        "move_backward": move_backward,
        "strafe_left": strafe_left,
        "strafe_right": strafe_right,
        "rotate_in_place": rotate_in_place,
        "turn_left": turn_left,
        "turn_right": turn_right,

        # --- Maps ---
        "list_maps": list_maps,
        "get_map_info": get_map_info,
        "map_exists": map_exists,
        "save_map": save_map,
        "save_map_to_disk": save_map_to_disk,
        "load_map": load_map,

        # --- Diagnostics ---
        "describe_rover_state": describe_rover_state,
        "compute_pose_offset": compute_pose_offset,
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