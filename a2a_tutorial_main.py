import asyncio
import uvicorn
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from agent_executor import HelloWorldAgentExecutor
from starlette.applications import Starlette
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
#import nest_asyncio
#nest_asyncio.apply()

if __name__ == '__main__':
    skill = AgentSkill(
        id='hello_world',
        name='Returns hello world',
        description='just returns hello world',
        tags=['hello world'],
        examples=['hi', 'hello world'],
    )

    extended_skill = AgentSkill(
        id='super_hello_world',
        name='Returns a SUPER Hello World',
        description='A more enthusiastic greeting, only for authenticated users.',
        tags=['hello world', 'super', 'extended'],
        examples=['super hi', 'give me a super hello'],
    )

    public_agent_card = AgentCard(
        name='Hello World Agent',
        description='Just a hello world agent',
        version='0.0.1',
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        capabilities=AgentCapabilities(
            streaming=True, extended_agent_card=True
        ),
        supported_interfaces=[
            AgentInterface(
                protocol_binding='JSONRPC',
                url='http://127.0.0.1:9999',
            )
        ],
        skills=[skill],
    )

    extended_agent_card = AgentCard(
        name='Hello World Agent - Extended Edition',
        description='The full-featured hello world agent for authenticated users.',
        version='0.0.2',
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        capabilities=AgentCapabilities(
            streaming=True, extended_agent_card=True
        ),
        supported_interfaces=[
            AgentInterface(
                protocol_binding='JSONRPC',
                url='http://127.0.0.1:9999',
            )
        ],
        skills=[skill, extended_skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor = HelloWorldAgentExecutor(),
        task_store  = InMemoryTaskStore(),
        agent_card = public_agent_card,
        extended_agent_card = extended_agent_card
    )

    routes = []
    routes.extend(create_agent_card_routes(public_agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, '/'))

    app = Starlette(routes = routes)

    uvicorn.run(app, host='127.0.0.1', port=9999)