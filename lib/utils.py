from enum import Enum
from typing import Any

from lib.constants import (
    CITY_BASE_COST,
    CITY_MAX_COST,
    DEFENSE_POST_BASE_COST,
    DEFENSE_POST_MAX_COST,
    MISSILE_SILO_COST,
    PORT_BASE_COST,
    PORT_MAX_COST,
)

FLOAT_TOLERANCE = 1e-9


class Action(Enum):
    SPAWN = "spawn"
    ATTACK = "attack"
    BUILD = "build_unit"
    NONE = "none"


def format_number(v: float | None, fmt: str = "{:.2f}") -> str:
    if v is None:
        return "N/A"
    try:
        return fmt.format(v)
    except Exception:
        try:
            return str(float(v))
        except Exception:
            return str(v)


def normalize_number(v: Any) -> int | float | None:
    if v is None:
        return None
    try:
        if isinstance(v, int):
            return v
        f = float(v)
        if abs(f - int(f)) < FLOAT_TOLERANCE:
            return int(f)
        return f
    except Exception:
        return None


def get_action_key(action: dict[str, Any]) -> str:
    action_type = action.get("type")
    if action_type == Action.SPAWN.value:
        return f"spawn:{action.get('x')},{action.get('y')}"
    if action_type == Action.ATTACK.value:
        return f"attack:idx{action.get('neighbor_index')}|ratio:{action.get('ratio')}"
    if action_type == Action.BUILD.value:
        return f"build:{action.get('unit')}"
    return Action.NONE.value


def calculate_building_cost(building_type: str, current_count: int) -> int:
    if building_type == "City":
        return min(CITY_MAX_COST, 2**current_count * CITY_BASE_COST)
    if building_type == "Port":
        return min(PORT_MAX_COST, 2**current_count * PORT_BASE_COST)
    if building_type == "Defense Post":
        return min(DEFENSE_POST_MAX_COST, (current_count + 1) * DEFENSE_POST_BASE_COST)
    if building_type == "Missile Silo":
        return MISSILE_SILO_COST
    return 0
