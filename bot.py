import asyncio
import json
import random
import os
import string
from typing import Any, Dict, Optional, List
import websockets

# Configuration (defined directly as constants)
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 3000
BOT_WS_PATH = "/bot"
SERVER_WS = f"ws://{SERVER_HOST}:{SERVER_PORT}{BOT_WS_PATH}"

QTABLE_FILE = "qtable.json"
PRUNE_MAX_TARGETS = 50
ATTACK_RATIOS = [0.20, 0.40, 0.60, 0.80]
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9995
AUTOSAVE_INTERVAL = 300
PRINT_INTERVAL = 20

# Reward constants
REWARD_CAPTURE_ENEMY = 0.0
REWARD_CAPTURE_EMPTY = 0.0
REWARD_FAILED_ATTACK = 0.0
REWARD_TERRITORY_GAIN = 10.0
REWARD_TERRITORY_LOSS = -10.0
REWARD_SPAWN_SUCCESS = 20.0
REWARD_MISSED_SPAWN = -50.0
REWARD_SMALL_STEP = -0.1


def make_id(length: int = 8) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def safe_num(v: Optional[float], fmt: str = "{:.2f}") -> str:
    if v is None:
        return "N/A"
    try:
        return fmt.format(v)
    except Exception:
        try:
            return str(float(v))
        except Exception:
            return str(v)


def get_state_key(state: Dict[str, Any]) -> str:
    in_spawn = bool(state.get("inSpawnPhase", False))
    candidates = state.get("candidates") or {}
    population = state.get("me", {}).get("population", 0.0) or 0.0
    empty_count = len(candidates.get("emptyNeighbors") or [])
    enemy_count = len(candidates.get("enemyNeighbors") or [])
    rank = state.get("me", {}).get("rank", 0) or 0
    conquest = int(state.get("me", {}).get("conquestPercent", 0) or 0)
    return f"spawn:{in_spawn}|empty:{empty_count}|enemy:{enemy_count}|pop:{population:.2f}|rank:{rank}|conq:{conquest}"


def get_action_key(action: Dict[str, Any]) -> str:
    action_type = action.get("type")
    if action_type == "spawn":
        return f"spawn:{action.get('x')},{action.get('y')}"
    elif action_type == "attack":
        return f"attack:{action.get('x')},{action.get('y')}|ratio:{action.get('ratio')}"
    return "none"


class Agent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable: Dict[str, Dict[str, float]] = {}
        self.previous_state: Optional[Dict[str, Any]] = None
        self.previous_action: Optional[Dict[str, Any]] = None

    def get_q_value(self, state_key: str, action_key: str) -> float:
        if state_key not in self.qtable:
            return 0.0
        return self.qtable[state_key].get(action_key, 0.0)

    def best_action(self, state: Dict[str, Any], possible_actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not possible_actions:
            return None
        state_key = get_state_key(state)
        if state_key not in self.qtable:
            self.qtable[state_key] = {}
            for action in possible_actions:
                action_key = get_action_key(action)
                self.qtable[state_key][action_key] = 0.0
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        possible_actions_keys = {get_action_key(action): action for action in possible_actions}
        for action_key in possible_actions_keys:
            if action_key not in self.qtable[state_key]:
                self.qtable[state_key][action_key] = 0.0
        valid_q_actions = [
            (action_key, q_value)
            for action_key, q_value in self.qtable[state_key].items()
            if action_key in possible_actions_keys
        ]
        if not valid_q_actions:
            return random.choice(possible_actions)
        best_action_key = max(valid_q_actions, key=lambda item: item[1])[0]
        return possible_actions_keys.get(best_action_key, random.choice(possible_actions))

    def update(self, state: Dict[str, Any], reward: float) -> None:
        if self.previous_state is None or self.previous_action is None:
            return
        prev_state_key = get_state_key(self.previous_state)
        prev_action_key = get_action_key(self.previous_action)
        current_state_key = get_state_key(state)
        current_q = self.get_q_value(prev_state_key, prev_action_key)
        max_next_q = 0.0
        if current_state_key in self.qtable and self.qtable[current_state_key]:
            max_next_q = max(self.qtable[current_state_key].values())
        delta = self.alpha * (reward + self.gamma * max_next_q - current_q)
        if prev_state_key not in self.qtable:
            self.qtable[prev_state_key] = {}
        self.qtable[prev_state_key][prev_action_key] = current_q + delta

    def save_qtable(self, file_path: str) -> None:
        try:
            with open(file_path, "w") as f:
                json.dump(self.qtable, f, indent=2)
        except Exception as e:
            print("Failed saving qtable:", e)

    def load_qtable(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            return
        try:
            with open(file_path, "r") as f:
                self.qtable = json.load(f)
        except Exception as e:
            print("Failed loading qtable:", e)


agent = Agent()
agent.load_qtable(QTABLE_FILE)


def calculate_reward(old_state: Dict[str, Any], new_state: Dict[str, Any], action: Optional[Dict[str, Any]]) -> float:
    """Compute reward from old_state -> new_state based on the previous action.

    Adds:
    - positive reward when a spawn action resulted in smallID being assigned
    - large negative penalty when spawn was possible but agent didn't spawn
    """
    reward = REWARD_SMALL_STEP

    # Spawn success detection: previous action was spawn and now we have a smallID
    try:
        prev_small = old_state.get("me", {}).get("smallID")
        new_small = new_state.get("me", {}).get("smallID")
        if action and action.get("type") == "spawn" and (not prev_small) and new_small:
            reward += REWARD_SPAWN_SUCCESS
            print(f"Reward: spawn success detected -> +{REWARD_SPAWN_SUCCESS}")
    except Exception:
        pass

    # Missed-spawn penalty: if previous state was spawn-phase and spawn was possible,
    # but previous action was not a spawn -> penalize
    try:
        prev_in_spawn = bool(old_state.get("inSpawnPhase", False))
        prev_candidates = (old_state.get("candidates") or {}).get("emptyNeighbors") or []
        if prev_in_spawn and len(prev_candidates) > 0:
            if not action or action.get("type") != "spawn":
                reward += REWARD_MISSED_SPAWN
                print(f"Penalty: missed spawn in spawn phase -> {REWARD_MISSED_SPAWN}")
    except Exception:
        pass

    # Territory/attack rewards (kept from prior logic)
    try:
        old_territory = len(old_state.get("owned", []))
        new_territory = len(new_state.get("owned", []))
        territory_diff = new_territory - old_territory
        if territory_diff > 0:
            reward += REWARD_TERRITORY_GAIN * territory_diff
            if action and action.get("type") == "attack":
                old_candidates = old_state.get("candidates", {})
                enemy_neighbors = old_candidates.get("enemyNeighbors", [])
                action_coords = (action.get("x"), action.get("y"))
                if any((n.get("x"), n.get("y")) == action_coords for n in enemy_neighbors):
                    reward += REWARD_CAPTURE_ENEMY
                else:
                    reward += REWARD_CAPTURE_EMPTY
        elif territory_diff < 0:
            reward += REWARD_TERRITORY_LOSS * abs(territory_diff)
        if action and action.get("type") == "attack" and territory_diff == 0:
            reward += REWARD_FAILED_ATTACK
    except Exception:
        pass

    return reward


def get_possible_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of possible actions for the agent.

    During spawn phase, if emptyNeighbors exist, return only spawn actions (no 'none' or attack).
    If no emptyNeighbors, return a single 'none' action (can't spawn).
    """
    candidates = state.get("candidates") or {}
    empty = candidates.get("emptyNeighbors") or []
    enemy = candidates.get("enemyNeighbors") or []

    if state.get("inSpawnPhase"):
        if empty:
            actions: List[Dict[str, Any]] = []
            for cell in empty[:PRUNE_MAX_TARGETS]:
                actions.append({"type": "spawn", "x": cell["x"], "y": cell["y"]})
            return actions
        else:
            return [{"type": "none"}]

    # Not spawn phase: build attack/none actions
    targets = (enemy or []) + (empty or [])
    if not targets:
        return [{"type": "none"}]
    actions = [{"type": "none"}]
    prioritized = (enemy or []) + (empty or [])
    for cell in prioritized[:PRUNE_MAX_TARGETS]:
        for ratio in ATTACK_RATIOS:
            actions.append({"type": "attack", "x": cell["x"], "y": cell["y"], "ratio": ratio})
    return actions


async def send_intent(ws, intent: Dict[str, Any], log_prefix: str = "INTENT"):
    """Serialize and send intent over websocket; failures are printed but not raised."""
    try:
        payload = json.dumps(intent, separators=(",", ":"))
    except Exception:
        payload = str(intent)
    try:
        await ws.send(payload)
    except Exception as e:
        print(f"Failed to send {log_prefix}:", e)


def choose_auto_spawn(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    empty_neighbors = (state.get("candidates") or {}).get("emptyNeighbors") or []
    if empty_neighbors:
        choice = random.choice(empty_neighbors[:PRUNE_MAX_TARGETS])
        return {"type": "spawn", "x": choice.get("x"), "y": choice.get("y"), "used_server": True}

    mw = (state.get("map") or {}).get("width") or 0
    mh = (state.get("map") or {}).get("height") or 0
    if mw and mh:
        tx = random.randrange(0, mw)
        ty = random.randrange(0, mh)
        return {"type": "spawn", "x": tx, "y": ty, "used_server": False}

    return None


async def send_action_intent(ws, client_id: str, game_id: str, username: str, action: Dict[str, Any], player_id: Optional[str], state: Dict[str, Any]):
    if action.get("type") == "spawn":
        if player_id is None:
            player_id = make_id(8)
        intent_msg = {
            "type": "intent",
            "clientID": client_id,
            "gameID": game_id,
            "intent": {
                "type": "spawn",
                "clientID": client_id,
                "playerID": player_id,
                "flag": None,
                "name": username,
                "playerType": "BOT",
                "x": action.get("x"),
                "y": action.get("y"),
            },
        }
        print(f"SENDING SPAWN INTENT: {intent_msg}")
        await send_intent(ws, intent_msg, "SPAWN")
        return player_id

    if action.get("type") == "attack":
        players_map = {p.get("smallID"): p.get("playerID") for p in state.get("players", [])}
        candidates = state.get("candidates", {})
        target = None
        for e in candidates.get("enemyNeighbors", []) + candidates.get("emptyNeighbors", []):
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
            "clientID": client_id,
            "gameID": game_id,
            "intent": {
                "type": "attack",
                "clientID": client_id,
                "attackerID": player_id,
                "targetID": target_player_id,
                "troops": troops,
            },
        }
        print(f"SENDING ATTACK INTENT: {intent_attack}")
        await send_intent(ws, intent_attack, "ATTACK")
        return player_id

    return player_id


async def process_state_tick(ws, state: Dict[str, Any], context: Dict[str, Any]):
    """Process a single 'state' message. Context stores mutable loop state like previous_state, player_id, etc."""
    tick = state.get("tick")
    if tick == context["last_tick"]:
        return
    context["last_tick"] = tick

    me = state.get("me", {})
    small_id = me.get("smallID")
    troops = me.get("troops")
    owned = me.get("owned") or []

    if (isinstance(tick, int) and (tick % PRINT_INTERVAL == 0 or tick < PRINT_INTERVAL)):
        print(
            f"Tick {tick}: smallID={small_id} troops={troops if troops is not None else 'N/A'} owned={len(owned)}",
        )

    in_spawn = bool(state.get("inSpawnPhase", False))
    if in_spawn and not context["last_in_spawn"]:
        print(f"==> ENTER spawn phase (tick={tick})")
    if not in_spawn and context["last_in_spawn"]:
        print(f"<== EXIT spawn phase (tick={tick})")

    # Auto-send spawn intent on the frame we enter spawn phase
    if in_spawn and not context["last_in_spawn"]:
        auto = choose_auto_spawn(state)
        if auto:
            if context["player_id"] is None:
                context["player_id"] = make_id(8)
            spawn_intent = {
                "type": "intent",
                "clientID": context["client_id"],
                "gameID": context["current_game_id"],
                "intent": {
                    "type": "spawn",
                    "clientID": context["client_id"],
                    "playerID": context["player_id"],
                    "flag": None,
                    "name": context["username"],
                    "playerType": "BOT",
                    "x": auto.get("x"),
                    "y": auto.get("y"),
                },
            }
            print(f"AUTO-SENDING SPAWN INTENT: {spawn_intent}")
            await send_intent(ws, spawn_intent, "AUTO-SPAWN")
            # record for learning
            agent.previous_state = state
            agent.previous_action = {"type": "spawn", "x": auto.get("x"), "y": auto.get("y")}
            context["previous_state"] = state
            context["last_in_spawn"] = True
            return

    context["last_in_spawn"] = in_spawn

    # Learning update
    if context["previous_state"] is not None and agent.previous_action is not None:
        reward = calculate_reward(context["previous_state"], state, agent.previous_action)
        try:
            reward_val = float(reward)
        except Exception:
            reward_val = 0.0
        agent.update(state, reward_val)
        if (isinstance(tick, int) and (tick % PRINT_INTERVAL == 0 or tick < PRINT_INTERVAL)):
            prev_action_str = get_action_key(agent.previous_action)
            print(f"Tick {tick}: Action={prev_action_str}, Reward={safe_num(reward_val)} , QStates={len(agent.qtable)}")

    # Decide and send action for this tick
    possible_actions = get_possible_actions(state)
    action = agent.best_action(state, possible_actions)
    agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

    if action is not None and action.get("type") != "none":
        context["player_id"] = await send_action_intent(
            ws, context["client_id"], context["current_game_id"], context["username"], action, context["player_id"], state
        )
        agent.previous_state = state
        agent.previous_action = action

    context["previous_state"] = state

    # autosave
    context["autosave_timer"] += 1
    if context["autosave_timer"] >= AUTOSAVE_INTERVAL:
        agent.save_qtable(QTABLE_FILE)
        context["autosave_timer"] = 0


async def run_bot_loop():
    game_count = 0
    backoff = 0.5

    while True:
        client_id = make_id(8)
        persistent_id = make_id(8)
        username = f"rl-bot-{client_id[:4]}"

        context = {
            "client_id": client_id,
            "current_game_id": None,
            "player_id": None,
            "previous_state": None,
            "last_tick": None,
            "last_in_spawn": False,
            "autosave_timer": 0,
            "username": username,
        }

        print(f"Connecting to server {SERVER_WS} with clientID={client_id}")
        try:
            async with websockets.connect(SERVER_WS) as ws:
                # register
                hello = {"type": "hello", "clientID": client_id, "persistentID": persistent_id, "username": username}
                await send_intent(ws, hello, "HELLO")

                async for message in ws:
                    try:
                        state = json.loads(message)
                    except Exception:
                        # ignore malformed messages
                        continue

                    msg_type = state.get("type")
                    if msg_type == "created":
                        context["current_game_id"] = state.get("gameID")
                        game_count += 1
                        print(f"Started new game #{game_count} id={context['current_game_id']}")
                        continue
                    if msg_type == "start":
                        print(f"Received start message for game {context['current_game_id']}")
                        continue
                    if msg_type != "state":
                        continue

                    try:
                        await process_state_tick(ws, state, context)
                    except Exception as e:
                        # isolate per-message failures so connection stays alive
                        print("Error processing state tick:", e)
                        continue

        except Exception as e:
            print("Connection error or game ended:", repr(e))
            backoff = min(backoff * 2, 10.0)
        finally:
            print(f"Game #{game_count} ended (connection closed). Saving qtable...")
            agent.save_qtable(QTABLE_FILE)
            print(f"Reconnecting in {backoff:.1f}s...")
            await asyncio.sleep(backoff)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot_loop())
    except KeyboardInterrupt:
        print("Interrupted, saving qtable...")
        agent.save_qtable(QTABLE_FILE)
