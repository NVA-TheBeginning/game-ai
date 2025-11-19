import contextlib
import json
from typing import TYPE_CHECKING

from lib.constants import DEBUG_MODE
from lib.utils import Action, get_action_key

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection


class ServerInterface:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    async def handle_connection(self, ws: ServerConnection) -> None:
        hello = await ws.recv()
        try:
            msg = json.loads(hello)
            print("Bot connected:", msg)
        except Exception:
            print("Bot connected; failed reading hello message")

        last_tick = None
        spawn_sent = False
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
                try:
                    if hasattr(self.env, "update_state") and callable(
                        self.env.update_state
                    ):
                        self.env.update_state(state)
                    elif isinstance(self.env, dict):
                        self.env["previous_state"] = self.env.get("current_state")
                        self.env["current_state"] = state
                    else:
                        try:
                            self.env.previous_state = getattr(
                                self.env, "current_state", None
                            )
                            self.env.current_state = state
                        except Exception as e:
                            print(f"Failed to set previous/current state: {e}")
                except Exception:
                    pass
                if (
                    self.agent.previous_state is not None
                    and self.agent.previous_action is not None
                ):
                    reward = self.env.calculate_reward(
                        self.agent.previous_state, state, self.agent.previous_action
                    )
                    self.agent.update(state, reward)
                    prev_action_str = get_action_key(self.agent.previous_action)
                    if DEBUG_MODE:
                        print(
                            f"Tick {tick}: Action={prev_action_str}, Reward={reward:.2f}, QTable size={len(self.agent.qtable)}"
                        )

                possible_actions = self.env.get_possible_actions(state)
                action = self.agent.best_action(state, possible_actions)
                if action is not None:
                    if action.get("type") != Action.NONE.value:
                        await ws.send(json.dumps(action))
                    elif DEBUG_MODE:
                        print(f"Tick {tick}: Agent chose to do nothing")

                    self.agent.previous_state = state
                    self.agent.previous_action = action

                    with contextlib.suppress(Exception):
                        self.agent.save()

                try:
                    if (not spawn_sent) and tick == 1:
                        spawn_intent = {"type": Action.SPAWN.value, "x": -1, "y": -1}
                        await ws.send(json.dumps(spawn_intent))
                        print("Sent spawn intent to bot:", spawn_intent)
                        spawn_sent = True
                except Exception:
                    print("Failed sending spawn intent to bot")
        finally:
            print("\nConnection closed, saving Q-table...")
            try:
                self.agent.save()
            except Exception as e:
                print(f"Exception occurred while saving agent in finally block: {e}")
