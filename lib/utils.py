import math
import random
import re
from typing import Any, Dict, List, Optional

# A small, self-contained set of helper functions that implement the heavy
# behaviour previously defined inside `Agent` in `main.py`. These functions
# operate on the agent instance (which must provide attributes like `qtable`,
# `alpha`, `gamma`, etc.). Keeping them here avoids circular imports.


def _state_to_vector(state: Dict[str, Any]) -> Optional[List[float]]:
    try:
        if not state:
            return [0.0] * 9
        candidates = state.get("candidates") or {}
        empty_count = len(candidates.get("emptyNeighbors") or [])
        enemy_count = len(candidates.get("enemyNeighbors") or [])
        me = state.get("me", {}) or {}
        population = float(me.get("population") or 0.0)
        troops = float(me.get("troops") or 0.0)
        maxpop = float(me.get("maxPopulation") or 0.0)
        conq = float(me.get("conquestPercent") or 0.0) * 100.0
        owned = float(me.get("ownedCount") or len(state.get("owned", []) or []))
        in_spawn = 1.0 if bool(state.get("inSpawnPhase", False)) else 0.0
        rank = float(me.get("rank") or 0)
        return [
            in_spawn,
            float(empty_count),
            float(enemy_count),
            population,
            troops,
            maxpop,
            conq,
            owned,
            rank,
        ]
    except Exception:
        return None


def _key_to_vector(key: str) -> Optional[List[float]]:
    try:
        parts = key.split("|")
        data = {}
        for p in parts:
            if ":" in p:
                k, v = p.split(":", 1)
                data[k] = v

        in_spawn = (
            1.0 if data.get("spawn", "False") in ("True", "1", "true", "1.0") else 0.0
        )
        empty_count = float(re.sub(r"[^0-9.-]", "", data.get("empty", "0")) or 0)
        enemy_count = float(re.sub(r"[^0-9.-]", "", data.get("enemy", "0")) or 0)
        pop_raw = data.get("pop", "0")
        pop = (
            float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", pop_raw)[0])
            if re.findall(r"[-+]?[0-9]*\.?[0-9]+", pop_raw)
            else 0.0
        )
        rank = float(re.sub(r"[^0-9.-]", "", data.get("rank", "0")) or 0)
        conq_raw = data.get("conq", "0")
        conq = (
            float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", conq_raw)[0])
            if re.findall(r"[-+]?[0-9]*\.?[0-9]+", conq_raw)
            else 0.0
        )
        owned_raw = data.get("owned", data.get("ownedCount", "0"))
        owned = (
            float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", owned_raw)[0])
            if re.findall(r"[-+]?[0-9]*\.?[0-9]+", owned_raw)
            else 0.0
        )
        troops = (
            float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", data.get("troops", "0"))[0])
            if re.findall(r"[-+]?[0-9]*\.?[0-9]+", data.get("troops", "0"))
            else 0.0
        )
        maxpop = (
            float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", data.get("maxpop", "0"))[0])
            if re.findall(r"[-+]?[0-9]*\.?[0-9]+", data.get("maxpop", "0"))
            else 0.0
        )

        return [
            in_spawn,
            empty_count,
            enemy_count,
            pop,
            troops,
            maxpop,
            conq,
            owned,
            rank,
        ]
    except Exception:
        return None


def get_q_value(agent, state_key: str, action_key: str) -> float:
    if state_key in agent.qtable:
        return agent.qtable[state_key].get(action_key, 0.0)

    closest = find_closest_state_by_key(agent, state_key)
    if closest is not None and closest in agent.qtable:
        return agent.qtable[closest].get(action_key, 0.0)

    return 0.0


def best_action(agent, state: Dict[str, Any], possible_actions: List[Dict[str, Any]]):
    state_key = (
        agent._get_state_key(state) if hasattr(agent, "_get_state_key") else None
    )
    # fallback to building a key-like string if helper not present
    if state_key is None:
        # naive fallback: use str(state)
        state_key = str(state)

    if state_key not in agent.qtable:
        closest = find_closest_state_key_by_state(agent, state)
        if closest is not None and closest in agent.qtable:
            agent.qtable[state_key] = dict(agent.qtable[closest])
        else:
            agent.qtable[state_key] = {}

        for action in possible_actions:
            action_key = (
                agent._get_action_key(action)
                if hasattr(agent, "_get_action_key")
                else str(action)
            )
            if action_key not in agent.qtable[state_key]:
                agent.qtable[state_key][action_key] = 0.0

    if random.random() < getattr(agent, "epsilon", 0.2):
        return random.choice(possible_actions)

    possible_actions_keys = {
        (agent._get_action_key(a) if hasattr(agent, "_get_action_key") else str(a)): a
        for a in possible_actions
    }

    for action_key in possible_actions_keys:
        if action_key not in agent.qtable[state_key]:
            agent.qtable[state_key][action_key] = 0.0

    valid_q_actions = [
        (action_key, q_value)
        for action_key, q_value in agent.qtable[state_key].items()
        if action_key in possible_actions_keys
    ]

    if not valid_q_actions:
        return random.choice(possible_actions)

    best_action_key = max(valid_q_actions, key=lambda item: item[1])[0]
    return possible_actions_keys.get(best_action_key, random.choice(possible_actions))


def update(agent, state: Dict[str, Any], reward: float) -> None:
    prev_state_key = (
        agent._get_state_key(agent.previous_state)
        if hasattr(agent, "_get_state_key")
        else str(agent.previous_state)
    )
    prev_action_key = (
        agent._get_action_key(agent.previous_action)
        if hasattr(agent, "_get_action_key")
        else str(agent.previous_action)
    )
    current_state_key = (
        agent._get_state_key(state) if hasattr(agent, "_get_state_key") else str(state)
    )

    current_q = get_q_value(agent, prev_state_key, prev_action_key)

    max_next_q = 0.0
    if current_state_key in agent.qtable and agent.qtable[current_state_key]:
        max_next_q = max(agent.qtable[current_state_key].values())
    else:
        closest = find_closest_state_by_key(agent, current_state_key)
        if closest is not None and closest in agent.qtable and agent.qtable[closest]:
            max_next_q = max(agent.qtable[closest].values())

    delta = getattr(agent, "alpha", 0.1) * (
        reward + getattr(agent, "gamma", 0.95) * max_next_q - current_q
    )

    if prev_state_key not in agent.qtable:
        closest = find_closest_state_by_key(agent, prev_state_key)
        if closest is not None and closest in agent.qtable:
            agent.qtable[prev_state_key] = dict(agent.qtable[closest])
        else:
            agent.qtable[prev_state_key] = {}
    agent.qtable[prev_state_key][prev_action_key] = current_q + delta


def find_closest_state_key_by_state(agent, state: Dict[str, Any]) -> Optional[str]:
    if not agent.qtable:
        return None
    target = _state_to_vector(state)
    if target is None:
        return None
    best_key = None
    best_dist = float("inf")
    for k in agent.qtable.keys():
        vec = _key_to_vector(k)
        if vec is None:
            continue
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, vec)))
        if d < best_dist:
            best_dist = d
            best_key = k
    return best_key


def find_closest_state_by_key(agent, state_key: str) -> Optional[str]:
    if state_key in agent.qtable:
        return state_key
    target = _key_to_vector(state_key)
    if target is None:
        return None
    best_key = None
    best_dist = float("inf")
    for k in agent.qtable.keys():
        vec = _key_to_vector(k)
        if vec is None:
            continue
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, vec)))
        if d < best_dist:
            best_dist = d
            best_key = k
    return best_key
