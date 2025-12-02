import json
from typing import TYPE_CHECKING

from lib.connection_handler import ConnectionHandler

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection


class ServerInterface(ConnectionHandler):
    def __init__(self, agent, env):
        super().__init__(agent, env)

    async def send_action(self, ws, action: dict) -> None:
        await ws.send(json.dumps(action))

    async def process_message(self, message: str) -> None:
        try:
            state = json.loads(message)
        except Exception:
            return

        if state.get("type") != "state":
            return

        self.env.update_state(state)

    async def on_agent_action(self, _action: dict) -> None:
        pass

    async def handle_connection(self, ws: ServerConnection) -> None:
        hello = await ws.recv()
        try:
            msg = json.loads(hello)
            print("Bot connected:", msg)
        except Exception:
            print("Bot connected; failed reading hello message")

        self.agent.total_reward = 0
        self.running = True
        await self.run_connection(ws)

        print("\nConnection closed, saving Q-table...")
        await self.cleanup()
