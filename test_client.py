import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import display_agent_card, new_text_message
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from a2a.types.a2a_pb2 import GetExtendedAgentCardRequest, Role, SendMessageRequest

async def main()-> None:
    base_url = 'http://127.0.0.1:9999'
    
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(
            httpx_client = httpx_client,
            base_url = base_url
            )

        print(f'Attempting to fetch public agent card from {base_url}{AGENT_CARD_WELL_KNOWN_PATH}')

        public_card = await resolver.get_agent_card()
        print("\nSuccessfully fetched the public agent card:")
        display_agent_card(public_card)

        #print("\n-----Non-Streaming Call------")
        #config = ClientConfig(streaming = False)
        #client = await create_client(agent = public_card, client_config = config)
        #print("\n------Non-Streaming Initialized------")

        #message = new_text_message("Say hello.", role = Role.ROLE_USER)
        #request = SendMessageRequest(message = message)

        #print("Response:")
        #async for chunk in client.send_message(request):
        #    print(chunk)


        print("\n-----Streaming Call-------")
        streaming_config = ClientConfig(streaming = True)
        streaming_client = await create_client(agent = public_card, client_config = streaming_config)
        print("\n--------Streaming Call Initialized-------")

        message = new_text_message("Say hello.", role = Role.ROLE_USER)
        request = SendMessageRequest(message = message)

        streaming_response = streaming_client.send_message(request)

        async for chunk in streaming_response:
            print("Response:", chunk) 

        print("\n-----Extended Call------")
        extended_card = await streaming_client.get_extended_agent_card(GetExtendedAgentCardRequest())
        print('\nSuccessfully fetched authenticated extended agent card:')
        display_agent_card(extended_card)

        await streaming_client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())