import asyncio
import json
from typing import TYPE_CHECKING

from lib.utils import Action

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection


class ServerInterface:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.running = False

    async def action_sender_loop(self, ws):
        """Reads actions from environment queue and sends them to client."""
        try:
            while self.running:
                action = await self.env._action_queue.get()
                if action.get("type") != Action.NONE.value:
                    await ws.send(json.dumps(action))
                self.env._action_queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Action sender loop error: {e}")

    async def agent_loop(self):
        """Main agent loop."""
        try:
            await self.env._state_event.wait()
            while self.running:
                action = await self.agent.best_action()
                await self.agent.do(action)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Agent loop error: {e}")

    async def handle_connection(self, ws: ServerConnection) -> None:
        hello = await ws.recv()
        try:
            msg = json.loads(hello)
            print("Bot connected:", msg)
        except Exception:
            print("Bot connected; failed reading hello message")

        self.running = True
        sender_task = asyncio.create_task(self.action_sender_loop(ws))
        agent_task = asyncio.create_task(self.agent_loop())

        try:
            async for message in ws:
                try:
                    state = json.loads(message)
                except Exception:
                    continue

                if state.get("type") != "state":
                    continue

                self.env.update_state(state)

        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            self.running = False
            sender_task.cancel()
            agent_task.cancel()
            await asyncio.gather(sender_task, agent_task, return_exceptions=True)

            print("\nConnection closed, saving Q-table...")
            await self.agent.save()
