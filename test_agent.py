import json
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent import build_agent


class ScriptedModel:
    def bind_tools(self, tools):
        async def response(messages):
            tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
            if not tool_messages:
                return AIMessage(content="", tool_calls=[{"name": "buscar_cliente", "args": {"nombre": "Ana"}, "id": "consulta-1", "type": "tool_call"}])
            if len(tool_messages) == 1:
                return AIMessage(content="", tool_calls=[{"name": "buscar_pedidos", "args": {"cliente_id": 102}, "id": "suma-1", "type": "tool_call"}])
            return AIMessage(content="El cliente 102 tiene 3 pedidos por un total de 14500.")
        return RunnableLambda(response)


@pytest.mark.asyncio
async def test_cycle_and_persistent_state(tmp_path):
    db = str(tmp_path / "memory.sqlite")
    config = {"configurable": {"thread_id": "prueba"}, "recursion_limit": 10}
    async with AsyncSqliteSaver.from_conn_string(db) as memory:
        app = build_agent(memory, ScriptedModel())
        result = await app.ainvoke({"messages": [HumanMessage(content="Pedidos y total del cliente 102")]}, config)
        assert len([m for m in result["messages"] if isinstance(m, ToolMessage)]) == 2
    async with AsyncSqliteSaver.from_conn_string(db) as memory:
        app = build_agent(memory, ScriptedModel())
        state = await app.aget_state(config)
        assert state.values["messages"][0].content == "Pedidos y total del cliente 102"
        assert "14500" in state.values["messages"][-1].content
