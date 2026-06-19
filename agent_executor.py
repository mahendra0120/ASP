from a2a.helpers import new_task_from_user_message, new_text_artifact, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types.a2a_pb2 import TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent

class HelloWorldAgent:
    """An agent that greets 'Hello World!"""

    async def invoke(self) -> str:
        """Invoke the agent to generate the greeting"""
        return "Hello World!"


class HelloWorldAgentExecutor(AgentExecutor):
    """Test ProxyAgent Implementation"""
    
    def __init__(self) -> None:
        self.agent = HelloWorldAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the agent process and enqueue the final response."""
        task = context.current_task or new_task_from_user_message(context.message)

        await event_queue.enqueue_event(task)

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id = context.task_id,
                context_id = context.context_id,
                status = TaskStatus(
                    state = TaskState.TASK_STATE_WORKING,
                    message = new_text_message("Processing Request....")
                )
        )
        )

        result = await self.agent.invoke()

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id = context.task_id,
                context_id = context.context_id,
                artifact = new_text_artifact(name = 'result', text = result)
            )
        )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id = context.task_id,
                context_id = context.context_id,
                status = TaskStatus(
                    state = TaskState.TASK_STATE_COMPLETED
                )
            )
        )


    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Raise exception as cancel is not supported"""
        raise Exception("cancel not supported!")