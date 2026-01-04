import asyncio
from abc import ABC, abstractmethod

from lib.utils import Action


class ConnectionHandler(ABC):
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.running = False

    async def agent_loop(self):
        try:
            await self.env._state_event.wait()
            while self.running:
                action = await self.agent.best_action()
                await self.agent.do(action)
                await self.on_agent_action()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Agent loop error: {e}")

    async def action_sender_loop(self, ws):
        try:
            while self.running:
                action = await self.env._action_queue.get()
                if action.get("type") != Action.NONE.value:
                    await self.send_action(ws, action)
                self.env._action_queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Action sender loop error: {e}")

    async def receive_loop(self, ws):
        try:
            async for message in ws:
                await self.process_message(message)
        except Exception as e:
            print(f"Receive loop error: {e}")
        finally:
            self.running = False

    async def run_connection(self, ws, additional_tasks=None):
        receive_task = asyncio.create_task(self.receive_loop(ws))
        sender_task = asyncio.create_task(self.action_sender_loop(ws))
        agent_task = asyncio.create_task(self.agent_loop())

        tasks = [receive_task, sender_task, agent_task]
        if additional_tasks:
            tasks.extend(additional_tasks)

        await receive_task

        self.running = False
        for task in tasks[1:]:
            task.cancel()

        await asyncio.gather(*tasks[1:], return_exceptions=True)

    async def cleanup(self):
        await self.agent.save()

    @abstractmethod
    async def send_action(self, ws, action: dict) -> None:
        pass

    @abstractmethod
    async def process_message(self, message: str) -> None:
        pass

    @abstractmethod
    async def on_agent_action(self) -> None:
        pass
