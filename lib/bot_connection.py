import asyncio
import json
import random
import string
from typing import Any

import websockets

from lib.constants import (
    AUTOSAVE_INTERVAL,
    EPSILON_DECAY,
    EPSILON_MIN,
    GRAPH_ENABLED,
    SERVER_WS,
)
from lib.metrics import GameMetrics
from lib.utils import Action


def make_id(length: int = 8) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


class BotConnection:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.client_id = make_id(8)
        self.persistent_id = make_id(8)
        self.username = f"rl-bot-{self.client_id[:4]}"
        self.player_id: str | None = None
        self.current_game_id: str | None = None
        self.metrics = GameMetrics() if GRAPH_ENABLED else None
        self.running = False

    async def send_intent(
        self, ws, intent: dict[str, Any], log_prefix: str = "INTENT"
    ) -> None:
        try:
            payload = json.dumps(intent, separators=(",", ":"))
        except Exception:
            payload = str(intent)
        try:
            await ws.send(payload)
        except Exception as e:
            print(f"Failed to send {log_prefix}:", e)

    async def _ensure_player_id(self) -> str:
        if self.player_id is None:
            self.player_id = make_id(8)
        return self.player_id

    async def send_action_intent(self, ws, action: dict[str, Any]) -> None:
        if action.get("type") == Action.SPAWN.value:
            player_id = await self._ensure_player_id()
            intent_msg = {
                "type": "intent",
                "clientID": self.client_id,
                "gameID": self.current_game_id,
                "intent": {
                    "type": Action.SPAWN.value,
                    "clientID": self.client_id,
                    "playerID": player_id,
                    "flag": None,
                    "name": self.username,
                    "playerType": "BOT",
                    "x": action.get("x"),
                    "y": action.get("y"),
                },
            }
            await self.send_intent(ws, intent_msg, "SPAWN")

        elif action.get("type") == Action.ATTACK.value:
            state = self.env.current_state or {}
            players_map = {
                p.get("smallID"): p.get("playerID") for p in state.get("players", [])
            }
            candidates = state.get("candidates", {})
            target = None
            for e in candidates.get("enemyNeighbors", []) + candidates.get(
                "emptyNeighbors", []
            ):
                if e.get("x") == action.get("x") and e.get("y") == action.get("y"):
                    target = e
                    break
            target_player_id = None
            if target is not None and target.get("ownerSmallID") is not None:
                target_player_id = players_map.get(target.get("ownerSmallID"))

            troops_ratio = action.get("ratio", 0.5)
            my_troops = (state.get("me") or {}).get("troops")
            try:
                troops = int(max(0, min(1, float(troops_ratio))) * int(my_troops or 0))
            except Exception:
                troops = 0

            intent_attack = {
                "type": "intent",
                "clientID": self.client_id,
                "gameID": self.current_game_id,
                "intent": {
                    "type": Action.ATTACK.value,
                    "clientID": self.client_id,
                    "attackerID": self.player_id,
                    "targetID": target_player_id,
                    "troops": troops,
                },
            }
            await self.send_intent(ws, intent_attack, "ATTACK")

    async def receive_loop(self, ws):
        try:
            async for message in ws:
                try:
                    state = json.loads(message)
                except Exception:
                    continue

                msg_type = state.get("type")
                if msg_type == "created":
                    self.current_game_id = state.get("gameID")
                    print(f"Started new game {self.current_game_id}")
                    if self.metrics:
                        self.metrics.start_game()
                    continue
                if msg_type == "start":
                    print(f"Received start message for game {self.current_game_id}")
                    continue
                if msg_type != "state":
                    continue

                self.env.update_state(state)

        except Exception as e:
            print(f"Receive loop error: {e}")
        finally:
            self.running = False

    async def action_sender_loop(self, ws):
        try:
            while self.running:
                action = await self.env._action_queue.get()

                if action.get("type") != Action.NONE.value:
                    await self.send_action_intent(ws, action)

                self.env._action_queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Action sender loop error: {e}")

    async def agent_loop(self):
        try:
            print("Waiting for initial state...")
            await self.env._state_event.wait()
            print("Initial state received. Starting agent loop.")

            while self.running:
                action = await self.agent.best_action()
                await self.agent.do(action)
                self.agent.epsilon = max(
                    EPSILON_MIN, self.agent.epsilon * EPSILON_DECAY
                )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Agent loop error: {e}")

    async def autosave_loop(self):
        try:
            while self.running:
                await asyncio.sleep(AUTOSAVE_INTERVAL)
                await self.agent.save()
        except asyncio.CancelledError:
            pass

    async def process_state_tick(self, ws, state: dict[str, Any]) -> None:
        tick = state.get("tick")
        if not isinstance(tick, int) or tick == self.last_tick:
            return
        self.last_tick = tick

        if self.metrics is not None:
            self.metrics.update_tick(tick)

        me = state.get("me", {})
        in_spawn = bool(state.get("inSpawnPhase", False))

        handled_spawn = await self._handle_auto_spawn(ws, state)
        if handled_spawn:
            return

        self._update_env(state)
        await self._process_prev_transition(state, tick)
        await self._choose_and_send_action(ws, state)
        await self._maybe_autosave()

    async def run(self) -> None:
        backoff = 0.5

        while True:
            self.client_id = make_id(8)
            self.persistent_id = make_id(8)
            self.username = f"bot-{self.client_id[:4]}"
            self.player_id = None
            self.current_game_id = None
            self.running = True

            print(f"Connecting to server {SERVER_WS} with clientID={self.client_id}")
            try:
                async with websockets.connect(SERVER_WS) as ws:
                    hello = {
                        "type": "hello",
                        "clientID": self.client_id,
                        "persistentID": self.persistent_id,
                        "username": self.username,
                    }
                    await self.send_intent(ws, hello, "HELLO")

                    receive_task = asyncio.create_task(self.receive_loop(ws))
                    sender_task = asyncio.create_task(self.action_sender_loop(ws))
                    agent_task = asyncio.create_task(self.agent_loop())
                    autosave_task = asyncio.create_task(self.autosave_loop())

                    await receive_task

                    self.running = False
                    sender_task.cancel()
                    agent_task.cancel()
                    autosave_task.cancel()

                    await asyncio.gather(
                        sender_task, agent_task, autosave_task, return_exceptions=True
                    )

            except Exception as e:
                print("Connection error or game ended:", repr(e))
                backoff = min(backoff * 2, 10.0)
            finally:
                print("Game ended, saving qtable...\n")
                await self.agent.save()
                print(f"Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
