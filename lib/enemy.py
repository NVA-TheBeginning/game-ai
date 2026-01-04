from typing import Any


class Enemy:
    """Represents an enemy tile that can be attacked or interacted with."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def troops(self) -> int:
        return self._data.get("troops", 0)

    @property
    def x(self) -> int:
        return self._data.get("x", 0)

    @property
    def y(self) -> int:
        return self._data.get("y", 0)

    @property
    def owner_player_id(self) -> str | None:
        return self._data.get("ownerPlayerID")
