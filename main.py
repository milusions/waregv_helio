import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent import process_agent_request

# Configure logging for better observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("helio_server")

app = FastAPI(title="WareGV Helio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check() -> dict[str, str]:
    """Verifies that the server is alive and operational."""
    return {"status": "ok", "message": "Helio server is reachable!"}


async def _handle_agent_stream(
    websocket: WebSocket,
    conversation_id: str,
    message: str,
) -> None:
    """Dispatches the message to the synchronous agent in a thread pool

    while safely forwarding emited async events back to the WebSocket.
    """
    loop = asyncio.get_running_loop()
    
    # Track futures of background tasks emitted from the thread
    pending_tasks: list[asyncio.Future] = []

    def emit_event(event: dict[str, Any]) -> None:
        """Synchronous callback injected into the agent thread to push updates."""
        payload = {"conversation_id": conversation_id, **event}
        coro = websocket.send_json(payload)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        pending_tasks.append(future)

    try:
        # Offload the blocking agent execution to a separate worker thread
        final_response = await asyncio.to_thread(
            process_agent_request,
            message,
            conversation_id,
            emit_event,
        )

        # Flush any remaining intermediate stream frames
        if pending_tasks:
            await asyncio.gather(
                *(asyncio.wrap_future(task) for task in pending_tasks),
                return_exceptions=True,
            )

        # Dispatch the final consolidated outcome
        await websocket.send_json({
            "type": "final",
            "conversation_id": conversation_id,
            "output": final_response or "No response generated.",
        })

    except Exception as error:
        logger.exception("Error processing agent request for session %s", conversation_id)
        await websocket.send_json({
            "type": "error",
            "conversation_id": conversation_id,
            "message": f"Processing error: {str(error)}",
        })


@app.websocket("/ws/helio")
async def helio_websocket(websocket: WebSocket) -> None:
    """Manages a persistent, stateful WebSocket connection for a single user conversation."""
    await websocket.accept()
    conversation_id = str(uuid.uuid4())
    logger.info("WebSocket connection established. Session: %s", conversation_id)

    await websocket.send_json({
        "type": "session",
        "conversation_id": conversation_id,
        "status": "connected",
    })

    try:
        while True:
            raw_data = await websocket.receive_text()
            
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "conversation_id": conversation_id,
                    "message": "Malformed payload. Expected valid JSON.",
                })
                continue

            if payload.get("type") != "message":
                continue

            message = str(payload.get("message", "")).strip()
            if not message:
                continue

            await websocket.send_json({
                "type": "status",
                "conversation_id": conversation_id,
                "stage": "received",
                "message": "Request received.",
            })

            # Encapsulate processing logic to keep the main message loop tidy
            await _handle_agent_stream(websocket, conversation_id, message)

    except WebSocketDisconnect:
        logger.info("Client disconnected gracefully. Session: %s", conversation_id)
    except Exception as error:
        logger.error("Unexpected connection drop on session %s: %s", conversation_id, error)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        ws="websockets",
    )
