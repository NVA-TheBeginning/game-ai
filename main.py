import asyncio
import json
import random
from typing import Any, Dict
from websockets.asyncio.server import ServerConnection, serve


PORT = 8765

# Reward constants
REWARD_CAPTURE_ENEMY = 10.0
REWARD_CAPTURE_EMPTY = 5.0
REWARD_FAILED_ATTACK = -5.0
REWARD_TERRITORY_GAIN = 2.0
REWARD_TERRITORY_LOSS = -2.0
REWARD_SMALL_STEP = -0.1


def arg_max(table: Dict[str, float]) -> str:
    return max(table, key=table.get)


def get_state_key(state: Dict[str, Any]) -> str:
    in_spawn = state.get("inSpawnPhase", False)
    candidates = state.get("candidates") or {}
    empty_count = len(candidates.get("emptyNeighbors") or [])
    enemy_count = len(candidates.get("enemyNeighbors") or [])
    return f"spawn:{in_spawn}|empty:{empty_count}|enemy:{enemy_count}"


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

        best_action_key = arg_max(self.qtable[state_key])
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


agent = Agent()


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

    return reward


def get_possible_actions(state: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Generate all possible actions for the current state.
    """
    actions = []
    in_spawn = state.get("inSpawnPhase")
    candidates = state.get("candidates") or {}
    empty = candidates.get("emptyNeighbors") or []
    enemy = candidates.get("enemyNeighbors") or []

    if in_spawn:
        for cell in empty:
            actions.append({"type": "spawn", "x": cell["x"], "y": cell["y"]})
    else:
        actions.append({"type": "none"})

        attack_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        targets = enemy + empty

        for cell in targets:
            for ratio in attack_ratios:
                actions.append(
                    {
                        "type": "attack",
                        "x": cell["x"],
                        "y": cell["y"],
                        "ratio": ratio,
                    }
                )

    return actions


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

        action = decide_action(state)
        if action is not None:
            if action.get("type") != "none":
                await ws.send(json.dumps(action))
            else:
                print(f"Tick {tick}: Agent chose to do nothing")

            agent.previous_state = state
            agent.previous_action = action

        previous_state = state


async def main():
    print(f"Starting bot server on ws://127.0.0.1:{PORT}")
    async with serve(handle, "127.0.0.1", PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
