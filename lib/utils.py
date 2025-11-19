from enum import Enum
from typing import Any

FLOAT_TOLERANCE = 1e-9


class Action(Enum):
    SPAWN = "spawn"
    ATTACK = "attack"
    NONE = "none"


def format_number(v: float | None, fmt: str = "{:.2f}") -> str:
    """
    Format a number for display purposes with fallback handling.
    """
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
    """
    Normalize a value to int or float for state processing.
    """
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
        return f"attack:{action.get('x')},{action.get('y')}|ratio:{action.get('ratio')}"
    return Action.NONE.value
