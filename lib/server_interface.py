import json
from typing import Any, Dict
from websockets.asyncio.server import ServerConnection


def get_action_key(action: Dict[str, Any]) -> str:
    action_type = action.get("type")
    if action_type == "spawn":
        return f"spawn:{action.get('x')},{action.get('y')}"
    elif action_type == "attack":
        return f"attack:{action.get('x')},{action.get('y')}|ratio:{action.get('ratio')}"
    return "none"


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
                if (
                    self.agent.previous_state is not None
                    and self.agent.previous_action is not None
                ):
                    reward = self.env.calculate_reward(
                        self.agent.previous_state, state, self.agent.previous_action
                    )
                    self.agent.update(state, reward)
                    prev_action_str = get_action_key(self.agent.previous_action)
                    print(
                        f"Tick {tick}: Action={prev_action_str}, Reward={reward:.2f}, QTable size={len(self.agent.qtable)}"
                    )

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
