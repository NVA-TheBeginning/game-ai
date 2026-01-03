import asyncio
import contextlib
import fcntl
import pickle
import threading
from pathlib import Path
from typing import Any

from lib.constants import QTABLE_FILE

_initialization_lock = threading.Lock()


class QTable:
    """Thread-safe Q-table manager for multiple bot instances."""

    _instance: QTable | None = None
    _lock: asyncio.Lock | None = None

    def __init__(self, filename: str = QTABLE_FILE):
        self.filename = Path(filename)
        self._local_qtable: dict[Any, dict[str, float]] = {}
        self._dirty = False

    @classmethod
    def get_instance(cls, filename: str = QTABLE_FILE) -> QTable:
        if cls._instance is None:
            with _initialization_lock:
                if cls._instance is None:
                    cls._instance = cls(filename)
        assert cls._instance is not None
        return cls._instance

    @classmethod
    async def get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            with _initialization_lock:
                if cls._lock is None:
                    cls._lock = asyncio.Lock()
        assert cls._lock is not None
        return cls._lock

    def _acquire_file_lock(self, file_handle) -> None:
        try:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
        except (OSError, AttributeError) as e:
            print(f"Warning: Could not acquire file lock: {e}")

    def _release_file_lock(self, file_handle) -> None:
        with contextlib.suppress(OSError, AttributeError):
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)

    async def load(self) -> None:
        if not self.filename.exists():
            print(f"No saved Q-table found at {self.filename}, starting fresh")
            self._local_qtable = {}
            return

        async with await self.get_lock():
            try:
                with self.filename.open("rb") as f:
                    self._acquire_file_lock(f)
                    try:
                        data = pickle.load(f)
                        self._local_qtable = data
                        print(
                            f"Q-table loaded from {self.filename} "
                            f"({len(self._local_qtable)} states)"
                        )
                        self._dirty = False
                    finally:
                        self._release_file_lock(f)
            except Exception as e:
                print(f"Error loading Q-table: {e}, starting fresh")
                self._local_qtable = {}

    def _merge_qtable(self, other_qtable: dict[Any, dict[str, float]]) -> None:
        for state_key, actions in other_qtable.items():
            if state_key not in self._local_qtable:
                self._local_qtable[state_key] = {}
            for action_key, q_value in actions.items():
                current_value = self._local_qtable[state_key].get(action_key, 0.0)
                self._local_qtable[state_key][action_key] = max(current_value, q_value)

    def _load_and_merge(self) -> None:
        if not self.filename.exists():
            return

        with self.filename.open("rb") as f:
            self._acquire_file_lock(f)
            try:
                try:
                    data = pickle.load(f)
                    other_qtable = data[0] if isinstance(data, tuple) else data
                    self._merge_qtable(other_qtable)
                except Exception:
                    pass
            finally:
                self._release_file_lock(f)

    async def save(self) -> None:
        if len(self._local_qtable) == 0 and not self._dirty:
            return

        async with await self.get_lock():
            try:
                self._load_and_merge()

                with self.filename.open("wb") as f:
                    self._acquire_file_lock(f)
                    try:
                        pickle.dump(self._local_qtable, f)
                        print(
                            f"Q-table saved to {self.filename} "
                            f"({len(self._local_qtable)} states)"
                        )
                        self._dirty = False
                    finally:
                        self._release_file_lock(f)
            except Exception as e:
                print(f"Error saving Q-table: {e}")

    async def get_q_value(self, state_key: Any, action_key: str) -> float:
        async with await self.get_lock():
            if state_key not in self._local_qtable:
                return 0.0
            return self._local_qtable[state_key].get(action_key, 0.0)

    async def get_state_actions(
        self, state_key: Any, action_keys: list[str]
    ) -> dict[str, float]:
        async with await self.get_lock():
            if state_key not in self._local_qtable:
                return dict.fromkeys(action_keys, 0.0)

            return {
                action_key: self._local_qtable[state_key].get(action_key, 0.0)
                for action_key in action_keys
            }

    async def set_q_value(
        self, state_key: Any, action_key: str, q_value: float
    ) -> None:
        async with await self.get_lock():
            if state_key not in self._local_qtable:
                self._local_qtable[state_key] = {}
            self._local_qtable[state_key][action_key] = q_value
            self._dirty = True

    async def ensure_state_exists(self, state_key: Any, action_keys: list[str]) -> None:
        async with await self.get_lock():
            if state_key not in self._local_qtable:
                self._local_qtable[state_key] = {}
                print(
                    f"\rNew State: {state_key} | Total states: {len(self._local_qtable)}",
                    end="",
                    flush=True,
                )
            for action_key in action_keys:
                if action_key not in self._local_qtable[state_key]:
                    self._local_qtable[state_key][action_key] = 0.0

    async def get_max_q_value(self, state_key: Any) -> float:
        async with await self.get_lock():
            if state_key not in self._local_qtable or not self._local_qtable[state_key]:
                return 0.0
            return max(self._local_qtable[state_key].values())

    async def get_size(self) -> int:
        async with await self.get_lock():
            return len(self._local_qtable)
