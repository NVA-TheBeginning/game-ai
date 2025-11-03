import asyncio
import json
import random
import os
from typing import Any, Dict
from websockets.asyncio.server import ServerConnection, serve


PORT = 8765
QTABLE_FILE = "qtable.json"

GAME_END_CONQUEST_THRESHOLD = 80.0
PRUNE_MAX_TARGETS = 50
ATTACK_RATIOS = [0.20, 0.40, 0.60, 0.80]
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9995
AUTOSAVE_INTERVAL = 300


# Reward constants
REWARD_CAPTURE_ENEMY = 0.0
REWARD_CAPTURE_EMPTY = 0.0
REWARD_FAILED_ATTACK = 0.0
REWARD_TERRITORY_GAIN = 10.0
REWARD_TERRITORY_LOSS = -10.0
REWARD_POPULATION_TOO_LOW = -5.0
REWARD_POPULATION_WAY_TOO_LOW = -50.0
REWARD_POPULATION_TOO_HIGH = -2.5
REWARD_SMALL_STEP = -0.1



def arg_max(table: Dict[str, float]) -> str:
    return max(table, key=table.get)


def get_state_key(state: Dict[str, Any]) -> str:
    in_spawn = state.get("inSpawnPhase", False)
    candidates = state.get("candidates") or {}
    population = state.get("me", {}).get("population")
    try:
        population = float(population) if population is not None else 0.0
    except Exception:
        population = 0.0
    empty_count = len(candidates.get("emptyNeighbors") or [])
    enemy_count = len(candidates.get("enemyNeighbors") or [])
    return f"spawn:{in_spawn}|empty:{empty_count}|enemy:{enemy_count}|population:{population:.2f}"


def get_action_key(action: Dict[str, Any]) -> str:
    action_type = action.get("type")
    if action_type == "spawn":
        return f"spawn:{action.get('x')},{action.get('y')}"
    elif action_type == "attack":
        return f"attack:{action.get('x')},{action.get('y')}|ratio:{action.get('ratio')}"
    return "none"


class Agent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.qtable: Dict[str, Dict[str, float]] = {}
        self.previous_state: Dict[str, Any] | None = None
        self.previous_action: Dict[str, Any] | None = None

    def get_q_value(self, state_key: str, action_key: str) -> float:
        """Get Q-value for a state-action pair."""
        if state_key not in self.qtable:
            return 0.0
        return self.qtable[state_key].get(action_key, 0.0)

    def best_action(
        self, state: Dict[str, Any], possible_actions: list[Dict[str, Any]]
    ) -> Dict[str, Any] | None:
        if not possible_actions:
            return None

        state_key = get_state_key(state)

        # Initialize Q-values for this state if not seen before
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

        q_actions = self.qtable[state_key].items()

        valid_q_actions = [
            (action_key, q_value)
            for action_key, q_value in q_actions
            if action_key in possible_actions_keys
        ]

        if not valid_q_actions:
            print("No valid Q-actions found; selecting random action.")
            return random.choice(possible_actions)

        best_action_key = max(valid_q_actions, key=lambda item: item[1])[0]

        for action in possible_actions:
            if get_action_key(action) == best_action_key:
                return action

        return random.choice(possible_actions)

    def update(self, state: Dict[str, Any], reward: float) -> None:
        """Update Q-values using Q-learning formula."""
        if self.previous_state is None or self.previous_action is None:
            return

        prev_state_key = get_state_key(self.previous_state)
        prev_action_key = get_action_key(self.previous_action)
        current_state_key = get_state_key(state)

        current_q = self.get_q_value(prev_state_key, prev_action_key)

        max_next_q = 0.0
        if current_state_key in self.qtable and self.qtable[current_state_key]:
            max_next_q = max(self.qtable[current_state_key].values())

        # Q(s,a) += alpha * (reward + gamma * max Q(s') - Q(s,a))
        delta = self.alpha * (reward + self.gamma * max_next_q - current_q)

        if prev_state_key not in self.qtable:
            self.qtable[prev_state_key] = {}
        self.qtable[prev_state_key][prev_action_key] = current_q + delta

    def save_qtable(self, file_path: str) -> None:
        print(f"Saving Q-table with {len(self.qtable)} states to {file_path}...")
        try:
            with open(file_path, "w") as f:
                json.dump(self.qtable, f, indent=4)
            print("Saved Q-table successfully.")
        except (IOError, PermissionError) as e:
            print(f"Error saving Q-table: {e}")

    def load_qtable(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            print(f"No Q-table file found at {file_path}. Creating new table.")
            return

        print(f"Loading Q-table from {file_path}...")
        try:
            with open(file_path, "r") as f:
                self.qtable = json.load(f)
            print(f"Loaded {len(self.qtable)} states.")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading Q-table: {e}. Creating new table.")
            self.qtable = {}


agent = Agent()
agent.load_qtable(QTABLE_FILE)


def calculate_reward(
    old_state: Dict[str, Any], new_state: Dict[str, Any], action: Dict[str, Any]
) -> float:
    reward = REWARD_SMALL_STEP

    old_territory = len(old_state.get("owned", []))
    new_territory = len(new_state.get("owned", []))

    territory_diff = new_territory - old_territory
    if territory_diff > 0:
        reward += REWARD_TERRITORY_GAIN * territory_diff
        if action.get("type") == "attack":
            old_candidates = old_state.get("candidates", {})
            enemy_neighbors = old_candidates.get("enemyNeighbors", [])
            action_coords = (action.get("x"), action.get("y"))
            if any((n.get("x"), n.get("y")) == action_coords for n in enemy_neighbors):
                reward += REWARD_CAPTURE_ENEMY
            else:
                reward += REWARD_CAPTURE_EMPTY
    elif territory_diff < 0:
        reward += REWARD_TERRITORY_LOSS * abs(territory_diff)

    if action.get("type") == "attack" and territory_diff == 0:
        reward += REWARD_FAILED_ATTACK

    # population = new_state.get("me", {}).get("population", 0)
    # if population < 0.30 * new_state.get("me", {}).get("maxPopulation", 1):
    #     reward += REWARD_POPULATION_TOO_LOW
    # elif population > 0.80 * new_state.get("me", {}).get("maxPopulation", 1):
    #     reward += REWARD_POPULATION_TOO_HIGH
    # if population < 0.10 * new_state.get("me", {}).get("maxPopulation", 1):
    #     reward += REWARD_POPULATION_WAY_TOO_LOW

    return reward


def get_possible_actions(state: Dict[str, Any]) -> list[Dict[str, Any]]:
    actions = []
    in_spawn = state.get("inSpawnPhase")
    candidates = state.get("candidates") or {}
    empty = candidates.get("emptyNeighbors") or []
    enemy = candidates.get("enemyNeighbors") or []

    if in_spawn:
        for cell in empty[:PRUNE_MAX_TARGETS]:
            actions.append({"type": "spawn", "x": cell["x"], "y": cell["y"]})
        if not actions:
            actions.append({"type": "none"})
        return actions

    targets = enemy + empty
    if not targets:
        return [{"type": "none"}]

    prioritized = enemy + empty
    pruned = prioritized[:PRUNE_MAX_TARGETS]

    actions.append({"type": "none"})

    for cell in pruned:
        for ratio in ATTACK_RATIOS:
            actions.append(
                {"type": "attack", "x": cell["x"], "y": cell["y"], "ratio": ratio}
            )
    return actions

def agent_action_with_decay(agent: Agent, state, possible_actions):
    action = agent.best_action(state, possible_actions)
    agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
    return action

def decide_action(state: Dict[str, Any]) -> Dict[str, Any] | None:
    possible_actions = get_possible_actions(state)

    if not possible_actions:
        return None

    return agent.best_action(state, possible_actions)


async def handle(ws: ServerConnection):
    hello = await ws.recv()
    try:
        msg = json.loads(hello)
        print("Bot connected:", msg)
    except Exception:
        print("Bot connected; failed reading hello message")

    last_tick = None
    previous_state = None

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

        # Calculate reward and update Q-table if we have a previous state
        if previous_state is not None and agent.previous_action is not None:
            reward = calculate_reward(previous_state, state, agent.previous_action)
            agent.update(state, reward)
            prev_action_str = get_action_key(agent.previous_action)
            print(
                f"Tick {tick}: Action={prev_action_str}, Reward={reward:.2f}, QTable size={len(agent.qtable)}"
            )

        action = agent_action_with_decay(agent, state, possible_actions=get_possible_actions(state))
        if action is not None:
            if action.get("type") != "none":
                await ws.send(json.dumps(action))
            else:
                print(f"Tick {tick}: Agent chose to do nothing")

            agent.previous_state = state
            agent.previous_action = action

        previous_state = state

    agent.save_qtable(QTABLE_FILE)


async def main():
    print(f"Starting bot server on ws://127.0.0.1:{PORT}")
    async with serve(handle, "127.0.0.1", PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down server...")
        print("Saving Q-table before exit...")
        agent.save_qtable(QTABLE_FILE)
        pass
