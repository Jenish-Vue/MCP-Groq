import asyncio
import json
import os
import sys
from typing import Optional, Dict, List
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import openai

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)


class MCPClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()

    async def connect_to_servers(self, server_script_paths: List[str]):
        """Starts multiple local server processes and connects via MCP."""
        for server_script_path in server_script_paths:
            is_python = server_script_path.endswith('.py')
            is_js = server_script_path.endswith('.js')

            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command,
                args=[server_script_path],
                env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

            await session.initialize()
            self.sessions[server_script_path] = session

            response = await session.list_tools()
            tools = [tool.name for tool in response.tools]
            print(f"\nConnected to {server_script_path} with tools: {tools}")

    async def process_query(self, query: str) -> str:
        """Handles user input, queries OpenAI Groq, and executes tool calls if needed."""
        messages = [{"role": "user", "content": query}]
        available_tools = []
        tool_to_server = {}

        for server_script_path, session in self.sessions.items():
            response = await session.list_tools()
            for tool in response.tools:
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                })
                tool_to_server[tool.name] = server_script_path

        groq_response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            tools=available_tools
        )

        final_text = []
        assistant_message = groq_response.choices[0].message

        if assistant_message.content:
            final_text.append(assistant_message.content)

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                server_script_path = tool_to_server.get(tool_name)
                if not server_script_path:
                    final_text.append(f"Tool {tool_name} not found on any connected server.")
                    continue

                print(f"\nCalling tool '{tool_name}' on {server_script_path} with arguments: {tool_args}")

                tool_args_dict = json.loads(tool_args)
                result = await self.sessions[server_script_path].call_tool(tool_name, tool_args_dict)
                tool_response_text = str(result.content)

                final_text.append(f"[Called {tool_name} on {server_script_path} with args {tool_args}]")
                messages.append({"role": "user", "content": tool_response_text})

                groq_response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=messages
                )
                final_text.append(groq_response.choices[0].message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Interactive chat loop for user queries."""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Cleans up async resources."""
        await self.exit_stack.aclose()


async def main():
    """Main entry point to start MCP client."""
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script_1> <path_to_server_script_2> ...")
        sys.exit(1)

    server_script_paths = sys.argv[1:]

    client = MCPClient()
    try:
        await client.connect_to_servers(server_script_paths)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
