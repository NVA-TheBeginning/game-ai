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
    PRINT_INTERVAL,
    PRUNE_MAX_TARGETS,
    SERVER_WS,
)
from lib.metrics import GameMetrics
from lib.utils import Action, format_number, get_action_key


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
        self.last_tick: int | None = None
        self.last_in_spawn = False
        self.autosave_timer = 0
        self.total_score = 0.0
        self.prev_players_map: dict[int, dict[str, Any]] = {}
        self.metrics = GameMetrics() if GRAPH_ENABLED else None

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

    def _safe_update_env(self, env_obj: Any, state: dict[str, Any]) -> None:
        try:
            if hasattr(env_obj, "update_state") and callable(env_obj.update_state):
                env_obj.update_state(state)
                return
        except Exception:
            pass

        try:
            if isinstance(env_obj, dict):
                env_obj["previous_state"] = env_obj.get("current_state")
                env_obj["current_state"] = state
                return
        except Exception:
            pass

        try:
            prev = getattr(env_obj, "current_state", None)
            try:
                env_obj.previous_state = prev
                env_obj.current_state = state
                return
            except Exception:
                pass
        except Exception:
            pass

    async def _ensure_player_id(self) -> str:
        async with self.agent._state_lock:
            if self.player_id is None:
                self.player_id = make_id(8)
        return self.player_id

    def choose_auto_spawn(self, state: dict[str, Any]) -> dict[str, Any] | None:
        empty_neighbors = (state.get("candidates") or {}).get("emptyNeighbors") or []
        if empty_neighbors:
            choice = random.choice(empty_neighbors[:PRUNE_MAX_TARGETS])
            return {
                "type": Action.SPAWN.value,
                "x": choice.get("x"),
                "y": choice.get("y"),
                "used_server": True,
            }

        mw = (state.get("map") or {}).get("width") or 0
        mh = (state.get("map") or {}).get("height") or 0
        if mw and mh:
            tx = random.randrange(0, mw)
            ty = random.randrange(0, mh)
            return {"type": Action.SPAWN.value, "x": tx, "y": ty, "used_server": False}

        return None

    async def send_action_intent(
        self, ws, action: dict[str, Any], state: dict[str, Any]
    ) -> None:
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

    def print_attack_events(self, state: dict[str, Any], tick: int) -> None:
        try:
            players_list = state.get("players") or []
            players_map = {p.get("smallID"): p for p in players_list if p is not None}

            me = state.get("me", {})
            small_id = me.get("smallID")

            if small_id is not None:
                my_curr = players_map.get(small_id) or {}
                my_prev = self.prev_players_map.get(small_id) or {}

                curr_incoming = my_curr.get("incomingAttacks") or []
                prev_incoming = my_prev.get("incomingAttacks") or []
                curr_outgoing = my_curr.get("outgoingAttacks") or []
                prev_outgoing = my_prev.get("outgoingAttacks") or []

                def _normalize(a: dict[str, Any]) -> tuple | None:
                    try:
                        attacker = (
                            a.get("attackerID")
                            if a.get("attackerID") is not None
                            else a.get("attackerSmallID")
                        )
                        target = (
                            a.get("targetID")
                            if a.get("targetID") is not None
                            else a.get("targetSmallID")
                        )
                        troops = a.get("troops")
                        return (
                            int(attacker) if attacker is not None else None,
                            int(target) if target is not None else None,
                            int(troops) if troops is not None else None,
                        )
                    except Exception:
                        return None

                prev_in_keys = set(filter(None, (_normalize(a) for a in prev_incoming)))
                for a in curr_incoming:
                    key = _normalize(a)
                    if key and key not in prev_in_keys:
                        attacker_id = key[0]
                        troops_n = key[2]
                        attacker = players_map.get(attacker_id) or {}
                        attacker_name = (
                            attacker.get("displayName")
                            or attacker.get("name")
                            or "<unknown>"
                        )
                        print(
                            f"[tick {tick}] INCOMING attack from {attacker_name}: {troops_n} troops"
                        )

                prev_out_keys = set(
                    filter(None, (_normalize(a) for a in prev_outgoing))
                )
                for a in curr_outgoing:
                    key = _normalize(a)
                    if key and key not in prev_out_keys:
                        target_id = key[1]
                        troops_n = key[2]
                        target = players_map.get(target_id) or {}
                        if target_id not in players_map:
                            target_name = "empty cell"
                        else:
                            target_name = (
                                target.get("displayName")
                                or target.get("name")
                                or f"player {target_id}"
                            )
                        print(
                            f"[tick {tick}] OUTGOING attack to {target_name}: {troops_n} troops"
                        )

            self.prev_players_map = players_map
        except Exception:
            pass

    async def _handle_auto_spawn(self, ws, state: dict[str, Any]) -> bool:
        in_spawn = bool(state.get("inSpawnPhase", False))
        if not in_spawn or self.last_in_spawn:
            return False

        auto = self.choose_auto_spawn(state)
        if not auto:
            self.last_in_spawn = True
            return True

        player_id = await self._ensure_player_id()
        spawn_intent = {
            "type": "intent",
            "clientID": self.client_id,
            "gameID": self.current_game_id,
            "intent": {
                "type": "spawn",
                "clientID": self.client_id,
                "playerID": player_id,
                "flag": None,
                "name": self.username,
                "playerType": "BOT",
                "x": auto.get("x"),
                "y": auto.get("y"),
            },
        }
        await self.send_intent(ws, spawn_intent, "AUTO-SPAWN")
        spawn_action = {"type": Action.SPAWN.value, "x": auto.get("x"), "y": auto.get("y")}
        await self.agent.set_previous_state_action(state, spawn_action)
        self.last_in_spawn = True
        return True

    def _update_env(self, state: dict[str, Any]) -> None:
        try:
            self._safe_update_env(self.env, state)
        except Exception:
            try:
                self.env.previous_state = getattr(self.env, "current_state", None)
                self.env.current_state = state
            except Exception:
                pass

    async def _process_prev_transition(self, state: dict[str, Any], tick: int) -> None:
        prev_state_action = await self.agent.get_previous_state_action_if_ready()
        if prev_state_action is None:
            return

        prev_state, prev_action_copy = prev_state_action
        reward = self.env.calculate_reward(prev_state, state, prev_action_copy)
        await self.agent.update(state, reward)
        self.total_score += reward

        if self.metrics is not None:
            self.metrics.add_reward(reward)

        if isinstance(tick, int) and (tick % PRINT_INTERVAL == 0):
            prev_action_str = get_action_key(prev_action_copy)
            qtable_size = await self.agent.qtable.get_size()
            print(
                f"Tick {tick}: Action={prev_action_str}, Rewards={format_number(self.total_score)}, QStates={qtable_size}"
            )

    async def _choose_and_send_action(self, ws, state: dict[str, Any]) -> None:
        possible_actions = self.env.get_possible_actions(state)
        action = await self.agent.best_action(state, possible_actions)
        self.agent.epsilon = max(EPSILON_MIN, self.agent.epsilon * EPSILON_DECAY)

        if action is not None and action.get("type") != Action.NONE.value:
            await self.send_action_intent(ws, action, state)
            await self.agent.set_previous_state_action(state, action)

    async def _maybe_autosave(self) -> None:
        self.autosave_timer += 1
        if self.autosave_timer >= AUTOSAVE_INTERVAL:
            await self.agent.save()
            self.autosave_timer = 0

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
            self.last_tick = None
            self.last_in_spawn = False
            self.autosave_timer = 0
            self.total_score = 0.0

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

                    async for message in ws:
                        try:
                            state = json.loads(message)
                        except Exception:
                            continue

                        msg_type = state.get("type")
                        if msg_type == "created":
                            self.current_game_id = state.get("gameID")
                            print(f"Started new game {self.current_game_id}")
                            if self.metrics is not None:
                                self.metrics.start_game()
                            continue
                        if msg_type == "start":
                            print(
                                f"Received start message for game {self.current_game_id}"
                            )
                            continue
                        if msg_type != "state":
                            continue

                        try:
                            await self.process_state_tick(ws, state)
                        except Exception as e:
                            print("Error processing state tick:", e)
                            continue

            except Exception as e:
                print("Connection error or game ended:", repr(e))
                backoff = min(backoff * 2, 10.0)
            finally:
                print("Game ended, saving qtable...\n")
                await self.agent.save()

                if self.metrics is not None and self.last_tick is not None:
                    self.metrics.end_game(final_tick=self.last_tick)

                    try:
                        graph_path = self.metrics.generate_graphs()
                        if graph_path:
                            print(f"Training metrics graph saved to: {graph_path}")
                            summary = self.metrics.get_summary()
                            print(
                                f"Summary: {summary['total_games']} games, "
                                f"avg score: {summary['avg_score']:.2f}, "
                                f"avg duration: {summary['avg_duration']:.0f} ticks"
                            )
                    except Exception as graph_error:
                        print(f"Failed to generate graph: {graph_error}")

                print(f"Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
