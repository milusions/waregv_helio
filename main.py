from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import uuid
import uvicorn

from agent import process_agent_request

app = FastAPI(title="WareGV Helio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/helio")
async def helio_websocket(websocket: WebSocket):
    """One WebSocket connection owns one persistent Helio conversation."""
    await websocket.accept()
    conversation_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()

    await websocket.send_json({
        "type": "session",
        "conversation_id": conversation_id,
        "status": "connected",
    })

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "conversation_id": conversation_id, "message": "Invalid WebSocket JSON."})
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

            pending_events = []

            async def send_event(event):
                await websocket.send_json({"conversation_id": conversation_id, **event})

            def emit(event):
                future = asyncio.run_coroutine_threadsafe(send_event(event), loop)
                pending_events.append(future)

            try:
                final_response = await asyncio.to_thread(
                    process_agent_request,
                    message,
                    conversation_id,
                    emit,
                )
                if pending_events:
                    await asyncio.gather(
                        *(asyncio.wrap_future(f) for f in pending_events),
                        return_exceptions=True,
                    )
                await websocket.send_json({
                    "type": "final",
                    "conversation_id": conversation_id,
                    "output": final_response or "No response generated.",
                })
            except Exception as exc:
                await websocket.send_json({
                    "type": "error",
                    "conversation_id": conversation_id,
                    "message": str(exc),
                })
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
