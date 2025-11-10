import asyncio
import json
import os
import pickle
import random
import string
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from websockets.asyncio.server import ServerConnection, serve
import websockets




MODE = "bot"  




SERVER_WS = "ws://localhost:3000/bot"
INTERFACE_PORT = 8765
QTABLE_FILE = "qtable.pkl"


PRUNE_MAX_TARGETS = 50
ATTACK_RATIOS = [0.20, 0.40, 0.60, 0.80]
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9995


AUTOSAVE_INTERVAL = 300
PRINT_INTERVAL = 20


REWARD_CAPTURE_ENEMY = 0.0
REWARD_CAPTURE_EMPTY = 0.0
REWARD_FAILED_ATTACK = 0.0
REWARD_TERRITORY_GAIN = 10.0
REWARD_TERRITORY_LOSS = -10.0
REWARD_SPAWN_SUCCESS = 20.0
REWARD_MISSED_SPAWN = -50.0
REWARD_SMALL_STEP = -0.1





class Action(Enum):
    """Represents the possible action types in the game."""
    SPAWN = "spawn"
    ATTACK = "attack"
    NONE = "none"





def make_id(length: int = 8) -> str:
    """Generate a random alphanumeric ID."""
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def safe_num(v: Optional[float], fmt: str = "{:.2f}") -> str:
    """Safely format a number for display."""
    if v is None:
        return "N/A"
    try:
        return fmt.format(v)
    except Exception:
        try:
            return str(float(v))
        except Exception:
            return str(v)


def arg_max(table: Dict[str, float]) -> str:
    """Return the key with the maximum value in the table."""
    return max(table, key=table.get)


def get_state_key(state: Dict[str, Any]) -> str:
    """Convert a state dictionary to a unique string key for Q-table."""
    in_spawn = bool(state.get("inSpawnPhase", False))
    candidates = state.get("candidates") or {}
    population = state.get("me", {}).get("population", 0.0) or 0.0
    empty_count = len(candidates.get("emptyNeighbors") or [])
    enemy_count = len(candidates.get("enemyNeighbors") or [])
    rank = state.get("me", {}).get("rank", 0) or 0
    conquest = int(state.get("me", {}).get("conquestPercent", 0) or 0)
    return f"spawn:{in_spawn}|empty:{empty_count}|enemy:{enemy_count}|pop:{population:.2f}|rank:{rank}|conq:{conquest}"


def get_action_key(action: Dict[str, Any]) -> str:
    """Convert an action dictionary to a unique string key for Q-table."""
    action_type = action.get("type")
    if action_type == "spawn":
        return f"spawn:{action.get('x')},{action.get('y')}"
    elif action_type == "attack":
        return f"attack:{action.get('x')},{action.get('y')}|ratio:{action.get('ratio')}"
    return "none"





class Environment:
    """Represents the game environment and handles state transitions."""
    
    def __init__(self):
        """Initialize the environment."""
        self.current_state: Optional[Dict[str, Any]] = None
        self.previous_state: Optional[Dict[str, Any]] = None
    
    def update_state(self, state: Dict[str, Any]) -> None:
        """Update the environment with a new state."""
        self.previous_state = self.current_state
        self.current_state = state
    
    def get_state(self) -> Optional[Dict[str, Any]]:
        """Get the current state."""
        return self.current_state
    
    def do(self, action: Dict[str, Any]) -> float:
        """
        Execute an action and return the reward.
        This calculates the reward based on state transitions.
        """
        if self.previous_state is None or self.current_state is None:
            return 0.0
        
        return self.calculate_reward(self.previous_state, self.current_state, action)
    
    def calculate_reward(
        self, old_state: Dict[str, Any], new_state: Dict[str, Any], action: Optional[Dict[str, Any]]
    ) -> float:
        """
        Compute reward from old_state -> new_state based on the action taken.
        """
        reward = REWARD_SMALL_STEP
        
        
        try:
            prev_small = old_state.get("me", {}).get("smallID")
            new_small = new_state.get("me", {}).get("smallID")
            if action and action.get("type") == "spawn" and (not prev_small) and new_small:
                reward += REWARD_SPAWN_SUCCESS
                print(f"Reward: spawn success detected -> +{REWARD_SPAWN_SUCCESS}")
        except Exception:
            pass
        
        
        try:
            prev_in_spawn = bool(old_state.get("inSpawnPhase", False))
            prev_candidates = (old_state.get("candidates") or {}).get("emptyNeighbors") or []
            if prev_in_spawn and len(prev_candidates) > 0:
                if not action or action.get("type") != "spawn":
                    reward += REWARD_MISSED_SPAWN
                    print(f"Penalty: missed spawn in spawn phase -> {REWARD_MISSED_SPAWN}")
        except Exception:
            pass
        
        
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
    
    def get_possible_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Return list of possible actions for the agent.
        During spawn phase, only spawn actions are available.
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
        
        
        targets = (enemy or []) + (empty or [])
        if not targets:
            return [{"type": "none"}]
        
        actions = [{"type": "none"}]
        prioritized = (enemy or []) + (empty or [])
        for cell in prioritized[:PRUNE_MAX_TARGETS]:
            for ratio in ATTACK_RATIOS:
                actions.append({"type": "attack", "x": cell["x"], "y": cell["y"], "ratio": ratio})
        
        return actions





class Agent:
    """Q-learning agent that learns to play the game."""
    
    def __init__(self, env: Environment, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.2):
        """
        Initialize the agent.
        
        Args:
            env: The environment the agent operates in
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable: Dict[str, Dict[str, float]] = {}
        self.previous_state: Optional[Dict[str, Any]] = None
        self.previous_action: Optional[Dict[str, Any]] = None
        self.score = 0.0
        self.history: List[float] = []
    
    def reset(self) -> None:
        """Reset the agent for a new episode."""
        if self.score != 0:
            self.history.append(self.score)
        self.score = 0.0
        self.previous_state = None
        self.previous_action = None
    
    def get_q_value(self, state_key: str, action_key: str) -> float:
        """Get Q-value for a state-action pair."""
        if state_key not in self.qtable:
            return 0.0
        return self.qtable[state_key].get(action_key, 0.0)
    
    def best_action(self, state: Dict[str, Any], possible_actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Select the best action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            possible_actions: List of valid actions
            
        Returns:
            Selected action or None
        """
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
    
    def do(self, action: Dict[str, Any], state: Dict[str, Any]) -> None:
        """
        Execute an action and update Q-values.
        
        Args:
            action: The action to execute
            state: The new state after action
        """
        if self.previous_state is not None and self.previous_action is not None:
            reward = self.env.calculate_reward(self.previous_state, state, self.previous_action)
            self.update(state, reward)
            self.score += reward
        
        self.previous_state = state
        self.previous_action = action
    
    def update(self, state: Dict[str, Any], reward: float) -> None:
        """
        Update Q-values using Q-learning formula.
        
        Args:
            state: Current state
            reward: Reward received
        """
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
    
    def save(self, filename: str = QTABLE_FILE) -> None:
        """Save Q-table and history to file."""
        try:
            with open(filename, "wb") as f:
                pickle.dump((self.qtable, self.history), f)
            print(f"Q-table saved to {filename} ({len(self.qtable)} states)")
        except Exception as e:
            print(f"Error saving Q-table: {e}")
    
    def load(self, filename: str = QTABLE_FILE) -> None:
        """Load Q-table and history from file."""
        if not Path(filename).exists():
            print(f"No saved Q-table found at {filename}, starting fresh")
            return
        
        try:
            with open(filename, "rb") as f:
                self.qtable, self.history = pickle.load(f)
            print(f"Q-table loaded from {filename} ({len(self.qtable)} states)")
        except Exception as e:
            print(f"Error loading Q-table: {e}, starting fresh")
            self.qtable = {}
            self.history = []





class BotConnection:
    """Handles connection to game server in bot mode."""
    
    def __init__(self, agent: Agent, env: Environment):
        self.agent = agent
        self.env = env
        self.client_id = make_id(8)
        self.persistent_id = make_id(8)
        self.username = f"rl-bot-{self.client_id[:4]}"
        self.player_id: Optional[str] = None
        self.current_game_id: Optional[str] = None
        self.last_tick: Optional[int] = None
        self.last_in_spawn = False
        self.autosave_timer = 0
        self.total_score = 0.0
        self.prev_players_map: Dict[int, Dict[str, Any]] = {}
    
    async def send_intent(self, ws, intent: Dict[str, Any], log_prefix: str = "INTENT") -> None:
        """Send an intent message to the server."""
        try:
            payload = json.dumps(intent, separators=(",", ":"))
        except Exception:
            payload = str(intent)
        try:
            await ws.send(payload)
        except Exception as e:
            print(f"Failed to send {log_prefix}:", e)
    
    def choose_auto_spawn(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Choose a random spawn location for auto-spawning."""
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
    
    async def send_action_intent(self, ws, action: Dict[str, Any], state: Dict[str, Any]) -> None:
        """Send an action intent to the server."""
        if action.get("type") == "spawn":
            if self.player_id is None:
                self.player_id = make_id(8)
            intent_msg = {
                "type": "intent",
                "clientID": self.client_id,
                "gameID": self.current_game_id,
                "intent": {
                    "type": "spawn",
                    "clientID": self.client_id,
                    "playerID": self.player_id,
                    "flag": None,
                    "name": self.username,
                    "playerType": "BOT",
                    "x": action.get("x"),
                    "y": action.get("y"),
                },
            }
            await self.send_intent(ws, intent_msg, "SPAWN")
        
        elif action.get("type") == "attack":
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
                "clientID": self.client_id,
                "gameID": self.current_game_id,
                "intent": {
                    "type": "attack",
                    "clientID": self.client_id,
                    "attackerID": self.player_id,
                    "targetID": target_player_id,
                    "troops": troops,
                },
            }
            await self.send_intent(ws, intent_attack, "ATTACK")
    
    def print_attack_events(self, state: Dict[str, Any], tick: int) -> None:
        """Detect and print incoming/outgoing attacks."""
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
                
                def _normalize(a: Dict[str, Any]) -> Optional[tuple]:
                    try:
                        attacker = a.get("attackerID") if a.get("attackerID") is not None else a.get("attackerSmallID")
                        target = a.get("targetID") if a.get("targetID") is not None else a.get("targetSmallID")
                        troops = a.get("troops")
                        return (int(attacker) if attacker is not None else None, 
                                int(target) if target is not None else None, 
                                int(troops) if troops is not None else None)
                    except Exception:
                        return None
                
                prev_in_keys = set(filter(None, (_normalize(a) for a in prev_incoming)))
                for a in curr_incoming:
                    key = _normalize(a)
                    if key and key not in prev_in_keys:
                        attacker_id = key[0]
                        troops_n = key[2]
                        attacker = players_map.get(attacker_id) or {}
                        attacker_name = attacker.get("displayName") or attacker.get("name") or "<unknown>"
                        print(f"[tick {tick}] INCOMING attack from {attacker_name}: {troops_n} troops")
                
                prev_out_keys = set(filter(None, (_normalize(a) for a in prev_outgoing)))
                for a in curr_outgoing:
                    key = _normalize(a)
                    if key and key not in prev_out_keys:
                        target_id = key[1]
                        troops_n = key[2]
                        target = players_map.get(target_id) or {}
                        target_name = target.get("displayName") or target.get("name") or "<unknown>"
                        print(f"[tick {tick}] OUTGOING attack to {target_name}: {troops_n} troops")
            
            self.prev_players_map = players_map
        except Exception:
            pass
    
    async def process_state_tick(self, ws, state: Dict[str, Any]) -> None:
        """Process a single state message from the server."""
        tick = state.get("tick")
        if tick == self.last_tick:
            return
        self.last_tick = tick
        
        me = state.get("me", {})
        in_spawn = bool(state.get("inSpawnPhase", False))
        
        
        if isinstance(tick, int) and (tick % PRINT_INTERVAL == 0):
            print(f"Tick {tick}: me={me}")
        
        
        self.print_attack_events(state, tick)
        
        
        if in_spawn and not self.last_in_spawn:
            print(f"==> ENTER spawn phase (tick={tick})")
        if not in_spawn and self.last_in_spawn:
            print(f"<== EXIT spawn phase (tick={tick})")
        
        
        if in_spawn and not self.last_in_spawn:
            auto = self.choose_auto_spawn(state)
            if auto:
                if self.player_id is None:
                    self.player_id = make_id(8)
                spawn_intent = {
                    "type": "intent",
                    "clientID": self.client_id,
                    "gameID": self.current_game_id,
                    "intent": {
                        "type": "spawn",
                        "clientID": self.client_id,
                        "playerID": self.player_id,
                        "flag": None,
                        "name": self.username,
                        "playerType": "BOT",
                        "x": auto.get("x"),
                        "y": auto.get("y"),
                    },
                }
                await self.send_intent(ws, spawn_intent, "AUTO-SPAWN")
                self.agent.previous_state = state
                self.agent.previous_action = {"type": "spawn", "x": auto.get("x"), "y": auto.get("y")}
                self.last_in_spawn = True
                return
        
        self.last_in_spawn = in_spawn
        
        
        self.env.update_state(state)
        if self.agent.previous_state is not None and self.agent.previous_action is not None:
            reward = self.env.calculate_reward(self.agent.previous_state, state, self.agent.previous_action)
            self.agent.update(state, reward)
            self.total_score += reward
            if isinstance(tick, int) and (tick % PRINT_INTERVAL == 0 or tick < PRINT_INTERVAL):
                prev_action_str = get_action_key(self.agent.previous_action)
                print(f"Tick {tick}: Action={prev_action_str}, TotalScore={safe_num(self.total_score)}, QStates={len(self.agent.qtable)}")
        
        
        possible_actions = self.env.get_possible_actions(state)
        action = self.agent.best_action(state, possible_actions)
        self.agent.epsilon = max(EPSILON_MIN, self.agent.epsilon * EPSILON_DECAY)
        
        if action is not None and action.get("type") != "none":
            await self.send_action_intent(ws, action, state)
            self.agent.previous_state = state
            self.agent.previous_action = action
        
        
        self.autosave_timer += 1
        if self.autosave_timer >= AUTOSAVE_INTERVAL:
            self.agent.save()
            self.autosave_timer = 0
    
    async def run(self) -> None:
        """Main bot connection loop."""
        game_count = 0
        backoff = 0.5
        
        while True:
            self.client_id = make_id(8)
            self.persistent_id = make_id(8)
            self.username = f"rl-bot-{self.client_id[:4]}"
            self.player_id = None
            self.current_game_id = None
            self.last_tick = None
            self.last_in_spawn = False
            self.autosave_timer = 0
            self.total_score = 0.0
            
            print(f"Connecting to server {SERVER_WS} with clientID={self.client_id}")
            try:
                async with websockets.connect(SERVER_WS) as ws:
                    
                    hello = {"type": "hello", "clientID": self.client_id, "persistentID": self.persistent_id, "username": self.username}
                    await self.send_intent(ws, hello, "HELLO")
                    
                    async for message in ws:
                        try:
                            state = json.loads(message)
                        except Exception:
                            continue
                        
                        msg_type = state.get("type")
                        if msg_type == "created":
                            self.current_game_id = state.get("gameID")
                            game_count += 1
                            print(f"Started new game {self.current_game_id}")
                            continue
                        if msg_type == "start":
                            print(f"Received start message for game {self.current_game_id}")
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
                print("Game ended, saving qtable...")
                self.agent.save()
                print(f"Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)


class ServerInterface:
    """Handles receiving connections from bots in interface mode."""
    
    def __init__(self, agent: Agent, env: Environment):
        self.agent = agent
        self.env = env
    
    async def handle_connection(self, ws: ServerConnection) -> None:
        """Handle a single bot connection."""
        hello = await ws.recv()
        try:
            msg = json.loads(hello)
            print("Bot connected:", msg)
        except Exception:
            print("Bot connected; failed reading hello message")
        
        last_tick = None
        
        try:
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
                
                
                self.env.update_state(state)
                
                
                if self.agent.previous_state is not None and self.agent.previous_action is not None:
                    reward = self.env.calculate_reward(self.agent.previous_state, state, self.agent.previous_action)
                    self.agent.update(state, reward)
                    prev_action_str = get_action_key(self.agent.previous_action)
                    print(f"Tick {tick}: Action={prev_action_str}, Reward={reward:.2f}, QTable size={len(self.agent.qtable)}")
                
                
                possible_actions = self.env.get_possible_actions(state)
                action = self.agent.best_action(state, possible_actions)
                
                if action is not None:
                    if action.get("type") != "none":
                        await ws.send(json.dumps(action))
                    else:
                        print(f"Tick {tick}: Agent chose to do nothing")
                    
                    
                    self.agent.previous_state = state
                    self.agent.previous_action = action

                    
                    
                    try:
                        self.agent.save()
                    except Exception:
                        pass
        finally:
            print("\nConnection closed, saving Q-table...")
            try:
                self.agent.save()
            except Exception:
                pass


async def run_main() -> None:
    """Entry point selecting bot or interface mode and running the appropriate loop."""
    env = Environment()
    agent = Agent(env)
    
    try:
        agent.load(QTABLE_FILE)
    except Exception:
        pass

    if MODE == "bot":
        bot = BotConnection(agent, env)
        try:
            await bot.run()
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            print("Interrupted, saving qtable...")
            agent.save()
    else:
        
        server_iface = ServerInterface(agent, env)
        print(f"Starting interface websocket server on 0.0.0.0:{INTERFACE_PORT}")
        async with serve(server_iface.handle_connection, "0.0.0.0", INTERFACE_PORT):
            try:
                await asyncio.Future()  
            except KeyboardInterrupt:
                print("Interface interrupted, saving qtable...")
                agent.save()


if __name__ == "__main__":
    try:
        asyncio.run(run_main())
    except KeyboardInterrupt:
        print("Shutting down, saving qtable...")
        
        try:
            
            a = Agent(Environment())
            a.load(QTABLE_FILE)
            a.save(QTABLE_FILE)
        except Exception:
            pass