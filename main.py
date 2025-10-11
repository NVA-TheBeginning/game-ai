import asyncio
import json
from typing import Any, Dict
import websockets


PORT = 8765


def decide_action(state: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Very simple baseline policy:
    - During spawn phase: spawn on the first empty neighbor.
    - Otherwise: attack first enemy neighbor (or empty neighbor) with 30% troops.
    The client throttles state messages to once per game tick.
    """
    # tick = state.get("tick")
    in_spawn = state.get("inSpawnPhase")
    candidates = state.get("candidates") or {}
    empty = candidates.get("emptyNeighbors") or []
    enemy = candidates.get("enemyNeighbors") or []

    if in_spawn and empty:
        target = empty[0]
        return {"type": "spawn", "x": target["x"], "y": target["y"]}

    target = enemy[0] if enemy else (empty[0] if empty else None)
    if target is not None:
        return {
            "type": "attack",
            "x": target["x"],
            "y": target["y"],
            "ratio": 0.3,
        }

    return None


async def handle(ws: websockets.WebSocketServerProtocol):
    hello = await ws.recv()
    try:
        msg = json.loads(hello)
        print("Bot connected:", msg)
    except Exception:
        print("Bot connected; failed reading hello message")

    last_tick = None
    async for message in ws:
        try:
            state = json.loads(message)
        except Exception:
            continue
        if state.get("type") != "state":
            continue

        tick = state.get("tick")
        if tick == last_tick:
            continue
        last_tick = tick

        action = decide_action(state)
        if action is not None:
            await ws.send(json.dumps(action))


async def main():
    print(f"Starting bot server on ws://127.0.0.1:{PORT}")
    async with websockets.serve(handle, "127.0.0.1", PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
