5. Starting the Server¶
Now that we have an Agent Card and an Agent Executor, we can set up and start the A2A server.

The A2A Python SDK provides an A2AStarletteApplication class that simplifies running an A2A-compliant HTTP server. It uses Starlette for the web framework and is typically run with an ASGI server like Uvicorn.

Server Setup in Helloworld¶
Let's look at __main__.py again to see how the server is initialized and started.


import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent_executor import (
    HelloWorldAgentExecutor,  # type: ignore[import-untyped]
)


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

    # This will be the public-facing agent card
    public_agent_card = AgentCard(
        name='Hello World Agent',
        description='Just a hello world agent',
        url='http://localhost:9999/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],  # Only the basic skill for the public card
        supports_authenticated_extended_card=True,
    )

    # This will be the authenticated extended agent card
    # It includes the additional 'extended_skill'
    specific_extended_agent_card = public_agent_card.model_copy(
        update={
            'name': 'Hello World Agent - Extended Edition',  # Different name for clarity
            'description': 'The full-featured hello world agent for authenticated users.',
            'version': '1.0.1',  # Could even be a different version
            # Capabilities and other fields like url, default_input_modes, default_output_modes,
            # supports_authenticated_extended_card are inherited from public_agent_card unless specified here.
            'skills': [
                skill,
                extended_skill,
            ],  # Both skills for the extended card
        }
    )

    request_handler = DefaultRequestHandler(
        agent_executor=HelloWorldAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
        extended_agent_card=specific_extended_agent_card,
    )

    uvicorn.run(server.build(), host='0.0.0.0', port=9999)
Let's break this down:

DefaultRequestHandler:

The SDK provides DefaultRequestHandler. This handler takes your AgentExecutor implementation (here, HelloWorldAgentExecutor) and a TaskStore (here, InMemoryTaskStore).
It routes incoming A2A RPC calls to the appropriate methods on your executor (like execute or cancel).
The TaskStore is used by the DefaultRequestHandler to manage the lifecycle of tasks, especially for stateful interactions, streaming, and resubscription. Even if your agent executor is simple, the handler needs a task store.
A2AStarletteApplication:

The A2AStarletteApplication class is instantiated with the agent_card and the request_handler (referred to as http_handler in its constructor).
The agent_card is crucial because the server will expose it at the /.well-known/agent-card.json endpoint (by default).
The request_handler is responsible for processing all incoming A2A method calls by interacting with your AgentExecutor.
uvicorn.run(server_app_builder.build(), ...):

The A2AStarletteApplication has a build() method that constructs the actual Starlette application.
This application is then run using uvicorn.run(), making your agent accessible over HTTP.
host='0.0.0.0' makes the server accessible on all network interfaces on your machine.
port=9999 specifies the port to listen on. This matches the url in the AgentCard.
Running the Helloworld Server¶
Navigate to the a2a-samples directory in your terminal (if you're not already there) and ensure your virtual environment is activated.

To run the Helloworld server:


# from the a2a-samples directory
python samples/python/agents/helloworld/__main__.py
You should see output similar to this, indicating the server is running:


INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9999 (Press CTRL+C to quit)
Your A2A Helloworld agent is now live and listening for requests! In the next step, we'll interact with it.