import asyncio
import json
import random
import string

import websockets
from websockets import ConnectionClosed

from lib.connection_handler import ConnectionHandler
from lib.constants import (
    AUTOSAVE_INTERVAL,
    CONQUEST_WIN_THRESHOLD,
    DEBUG_MODE,
    GRAPH_ENABLED,
    SERVER_WS,
    SPAWN_PHASE_DURATION,
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
        self.game_count = 0
        self.previous_owned_count: int = 0

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

    def extract_player_id_from_state(self, state: dict) -> str | None:
        me = state.get("me", {})
        me_small_id = me.get("smallID")

        if me_small_id is None or me_small_id < 0:
            return None

        players = state.get("players", [])
        for player in players:
            if player.get("smallID") == me_small_id:
                return player.get("playerID")

        return None

    def sync_player_state(self, state: dict) -> None:
        state_player_id = self.extract_player_id_from_state(state)
        if state_player_id:
            self.player_id = state_player_id
            me = state.get("me", {})
            if me.get("ownedCount", 0) > 0 or me.get("population", 0) > 0:
                self.has_spawned = True

    async def send_action(self, ws, action: dict) -> None:
        state = self.env.current_state or {}
        self.sync_player_state(state)

        action_type = action.get("type")

        if action_type == Action.SPAWN.value:
            await self.handle_spawn_action(ws, action)

        elif action_type == Action.ATTACK.value:
            await self.handle_attack_action(ws, action, state)

    async def handle_spawn_action(self, ws, action: dict) -> None:
        if self.has_spawned:
            return

        player_id = self.player_id or make_id(8)
        self.player_id = player_id

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
                "playerType": "HUMAN",
                "x": action.get("x"),
                "y": action.get("y"),
            },
        }
        self.has_spawned = True
        await self.send_intent(ws, intent_msg, "SPAWN")

    async def handle_attack_action(self, ws, action: dict, state: dict) -> None:
        if self.player_id is None:
            print("Warning: Cannot send attack intent - player_id not set")
            return

        target = self.find_attack_target(action, state)
        target_player_id = self.resolve_target_player_id(target)
        troops = self.calculate_attack_troops(action, state)

        intent_attack = {
            "type": "intent",
            "clientID": self.client_id,
            "gameID": self.current_game_id,
            "intent": {
                "type": Action.ATTACK.value,
                "clientID": self.client_id,
                "attackerID": self.player_id,
                "targetID": target_player_id,
                "x": target.get("x") if target else None,
                "y": target.get("y") if target else None,
                "troops": troops,
            },
        }
        await self.send_intent(ws, intent_attack, "ATTACK")

    def find_attack_target(self, action: dict, state: dict) -> dict | None:
        candidates = state.get("candidates", [])

        neighbor_index = action.get("neighbor_index")
        if (
            neighbor_index is None
            or neighbor_index < 0
            or neighbor_index >= len(candidates)
        ):
            return None

        return candidates[neighbor_index]

    def resolve_target_player_id(self, target: dict | None) -> str | None:
        if target is None:
            return None

        return target.get("ownerPlayerID")

    def calculate_attack_troops(self, action: dict, state: dict) -> int:
        troops_ratio = action.get("ratio", 0.5)
        my_troops = state.get("me", {}).get("population", 0)

        try:
            normalized_ratio = max(0.0, min(1.0, float(troops_ratio)))
            return int(normalized_ratio * int(my_troops))
        except (ValueError, TypeError):
            return 0

    async def process_message(self, message: str) -> None:
        try:
            state = json.loads(message)
        except Exception:
            return

        msg_type = state.get("type")
        if msg_type == "created":
            self.current_game_id = state.get("gameID")
            self.has_spawned = False
            self.previous_owned_count = 0
            print(f"Started new game {self.current_game_id}")
            self.agent.total_reward = 0
            if self.metrics:
                self.metrics.start_game()
            return
        if msg_type != "state":
            return

        self.debug_print_state(state)

        me = state.get("me", {})
        owned_count = me.get("ownedCount", 0)
        population = me.get("population", 0)
        conquest_pct = me.get("conquestPercent", 0)

        self.env.update_state(state)
        self.previous_owned_count = owned_count

        self.handle_fail_spawn(state)
        self.handle_victory(state)
        self.handle_game_over(owned_count, population, conquest_pct)

    def handle_fail_spawn(self, state: dict) -> None:
        tick = state.get("tick", 0)
        in_spawn_phase = state.get("inSpawnPhase", True)
        me = state.get("me", {})
        owned_count = me.get("ownedCount", 0)

        if (
            not in_spawn_phase
            and owned_count == 0
            and self.previous_owned_count == 0
            and tick > SPAWN_PHASE_DURATION + 50
        ):
            print(f"\nSpawn failed - never acquired any tiles (tick: {tick})")
            raise RuntimeError("Spawn failed - ending game")

    def handle_victory(self, state: dict) -> None:
        conquest_pct = state.get("me", {}).get("conquestPercent", 0)
        if conquest_pct >= CONQUEST_WIN_THRESHOLD:
            print(
                f"\nVictory! Conquest: {conquest_pct}% (threshold: {CONQUEST_WIN_THRESHOLD}%)"
            )
            if self.metrics:
                self.metrics.game_wins.append(1)
            raise RuntimeError("Victory achieved - ending game")

    def handle_game_over(
        self, owned_count: int, population: int, conquest_pct: int
    ) -> None:
        if self.previous_owned_count > 0 and owned_count == 0:
            print(
                f"\nPlayer eliminated (conquest: {conquest_pct}%, owned: {owned_count}, pop: {population})"
            )
            raise RuntimeError("Player eliminated - ending game")

    async def on_agent_action(self, _action: dict) -> None:
        if self.metrics:
            self.metrics.add_reward(self.agent.reward)

    async def autosave_loop(self):
        try:
            while self.running:
                await asyncio.sleep(AUTOSAVE_INTERVAL)
                await self.agent.save()
        except asyncio.CancelledError:
            pass

    async def start_ping_loop(self, ws):
        try:
            while self.running:
                await asyncio.sleep(10)
                ping_msg = {
                    "type": "ping",
                    "clientID": self.client_id,
                    "gameID": self.current_game_id,
                }
                try:
                    await self.send_intent(ws, ping_msg, "PING")
                except ConnectionClosed:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Ping loop error: {e}")

    async def run(self) -> None:
        backoff = 0.5

        try:
            while True:
                self.player_id = None
                self.current_game_id = None
                self.has_spawned = False
                self.previous_owned_count = 0
                self.running = True

                print(f"\n=== Starting Game #{self.game_count} ===")
                print(
                    f"Connecting to server {SERVER_WS} with clientID={self.client_id}"
                )
                self.game_count += 1

                try:
                    async with websockets.connect(SERVER_WS) as ws:
                        hello = {
                            "type": "hello",
                            "clientID": self.client_id,
                            "persistentID": self.persistent_id,
                            "username": self.username,
                        }
                        await self.send_intent(ws, hello, "HELLO")
                        ping_task = asyncio.create_task(self.start_ping_loop(ws))
                        autosave_task = asyncio.create_task(self.autosave_loop())

                        await self.run_connection(ws, [autosave_task, ping_task])

                except Exception as e:
                    print("Connection error or game ended:", repr(e))
                    backoff = min(backoff * 2, 10.0)
                finally:
                    print("\nGame ended, saving qtable...")
                    await self.cleanup()

                    if self.metrics:
                        final_tick = (self.env.current_state or {}).get("tick", 0)
                        me = (self.env.current_state or {}).get("me", {})
                        conquest_pct = me.get("conquestPercent", 0) or 0
                        win = conquest_pct >= CONQUEST_WIN_THRESHOLD
                        try:
                            qtable_size = await self.agent.qtable.get_size()
                        except Exception:
                            qtable_size = 0

                        self.metrics.end_game(
                            final_tick,
                            win=win,
                            qtable_size=qtable_size,
                            epsilon=self.agent.epsilon,
                        )
                        graph_path = self.metrics.generate_graphs()
                        if graph_path:
                            print(f"Graph saved to: {graph_path}")
                        summary = self.metrics.get_summary()
                        if summary:
                            print(
                                "Total games: "
                                f"{summary['total_games']}, "
                                f"Avg score: {summary['avg_score']:.2f}, "
                                f"Avg duration: {summary['avg_duration']:.1f} ticks, "
                                f"Win rate: {summary['win_rate'] * 100:.1f}%, "
                                f"Avg Q-table size: {summary['avg_qtable_size']:.0f}, "
                                f"Last epsilon: {summary['last_epsilon']:.4f}"
                            )

                    print(f"Reconnecting in {backoff:.1f}s...")
                    await asyncio.sleep(backoff)

        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            self.running = False
            await self.cleanup()
            print("Q-table saved. Exiting.")
