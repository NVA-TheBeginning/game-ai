import asyncio
import json
import random
import string

import websockets

from lib.connection_handler import ConnectionHandler
from lib.constants import (
    AUTOSAVE_INTERVAL,
    DEBUG_MODE,
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


class BotConnection(ConnectionHandler):
    def __init__(self, agent, env):
        super().__init__(agent, env)
        self.client_id = "agent001"
        self.persistent_id = make_id(8)
        self.username = "rl-bot-agent001"
        self.player_id: str | None = None
        self.current_game_id: str | None = None
        self.has_spawned: bool = False
        self.metrics = GameMetrics() if GRAPH_ENABLED else None

    def debug_print_state(self, state: dict) -> None:
        if DEBUG_MODE:
            print("\n" + "=" * 80)
            print(f"=== RAW STATE (Tick {state.get('tick', '?')}) ===")
            print(json.dumps(state, indent=2))
            print("=" * 80 + "\n")

    async def send_intent(self, ws, intent: dict, log_prefix: str = "INTENT") -> None:
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

    async def send_action(self, ws, action: dict) -> None:
        if action.get("type") == Action.SPAWN.value:
            if self.has_spawned:
                print("WARNING: Attempting to spawn but already spawned this game!")
                return
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
            self.has_spawned = True
            await self.send_intent(ws, intent_msg, "SPAWN")

        elif action.get("type") == Action.ATTACK.value:
            state = self.env.current_state or {}
            players_map = {
                p.get("smallID"): p.get("playerID") for p in state.get("players", [])
            }
            candidates = state.get("candidates", [])
            target = None

            if isinstance(candidates, dict):
                all_candidates = candidates.get("enemyNeighbors", []) + candidates.get(
                    "emptyNeighbors", []
                )
            else:
                all_candidates = candidates

            for e in all_candidates:
                if e.get("x") == action.get("x") and e.get("y") == action.get("y"):
                    target = e
                    break
            target_player_id = None
            if target is not None and target.get("ownerSmallID") is not None:
                target_player_id = players_map.get(target.get("ownerSmallID"))

            troops_ratio = action.get("ratio", 0.5)
            my_troops = (state.get("me") or {}).get("population")
            try:
                troops = int(max(0, min(1, float(troops_ratio))) * int(my_troops or 0))
            except Exception:
                troops = 0

            print(
                f"\nATTACK: Sending {troops} troops ({troops_ratio:.0%} of {my_troops}) to ({action.get('x')}, {action.get('y')}), targetID={target_player_id}"
            )

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
            print(f"Sending attack intent: {intent_attack}")
            await self.send_intent(ws, intent_attack, "ATTACK")

    async def process_message(self, message: str) -> None:
        try:
            state = json.loads(message)
        except Exception:
            return

        msg_type = state.get("type")
        if msg_type == "created":
            self.current_game_id = state.get("gameID")
            self.has_spawned = False
            print(f"Started new game {self.current_game_id}")
            self.agent.total_reward = 0
            if self.metrics:
                self.metrics.start_game()
            return
        if msg_type == "start":
            print(f"Received start message for game {self.current_game_id}")
            return
        if msg_type != "state":
            return

        self.debug_print_state(state)
        self.env.update_state(state)

    async def on_agent_action(self, _action: dict) -> None:
        if self.metrics:
            self.metrics.add_reward(self.agent.reward)

        self.agent.epsilon = max(EPSILON_MIN, self.agent.epsilon * EPSILON_DECAY)

    async def autosave_loop(self):
        try:
            while self.running:
                await asyncio.sleep(AUTOSAVE_INTERVAL)
                await self.agent.save()
        except asyncio.CancelledError:
            pass

    async def run(self) -> None:
        backoff = 0.5

        try:
            while True:
                self.player_id = None
                self.current_game_id = None
                self.has_spawned = False
                self.running = True

                print(
                    f"Connecting to server {SERVER_WS} with clientID={self.client_id}"
                )
                try:
                    async with websockets.connect(SERVER_WS) as ws:
                        hello = {
                            "type": "hello",
                            "clientID": self.client_id,
                            "persistentID": self.persistent_id,
                            "username": self.username,
                        }
                        await self.send_intent(ws, hello, "HELLO")

                        autosave_task = asyncio.create_task(self.autosave_loop())
                        await self.run_connection(ws, [autosave_task])

                except Exception as e:
                    print("Connection error or game ended:", repr(e))
                    backoff = min(backoff * 2, 10.0)
                finally:
                    print("\nGame ended, saving qtable...")
                    await self.cleanup()

                    if self.metrics:
                        final_tick = (self.env.current_state or {}).get("tick", 0)
                        self.metrics.end_game(final_tick)
                        graph_path = self.metrics.generate_graphs()
                        if graph_path:
                            print(f"Graph saved to: {graph_path}")
                        summary = self.metrics.get_summary()
                        if summary:
                            print(
                                f"Total games: {summary['total_games']}, Avg score: {summary['avg_score']:.2f}, Avg duration: {summary['avg_duration']:.1f} ticks"
                            )

                    print(f"Reconnecting in {backoff:.1f}s...")
                    await asyncio.sleep(backoff)

        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            self.running = False
            await self.cleanup()
            print("Q-table saved. Exiting.")
