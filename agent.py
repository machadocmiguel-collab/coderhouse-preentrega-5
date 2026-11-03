import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
CLIENTS = {'ana': 102, 'bruno': 103}
ORDERS = {
    102: [{'id': 501, 'total': 4500}, {'id': 502, 'total': 6000}, {'id': 503, 'total': 4000}],
    103: [{'id': 504, 'total': 2100}],
}


class ClientQuery(BaseModel):
    nombre: str = Field(min_length=1, description='Nombre del cliente a buscar')


@tool(args_schema=ClientQuery)
async def buscar_cliente(nombre: str) -> dict:
    """Busca por nombre el identificador de un cliente de la base de demostración.

    Usar antes de consultar pedidos cuando el usuario conoce el nombre pero
    no el identificador. Devuelve cliente_id o un error si no existe.
    """
    identifier = CLIENTS.get(nombre.strip().casefold())
    if identifier is None:
        return {'error': 'Cliente no encontrado. Pedir al usuario que aclare el nombre.'}
    return {'cliente_id': identifier, 'nombre': nombre}


class OrderQuery(BaseModel):
    cliente_id: int = Field(gt=0, description='Identificador obtenido de buscar_cliente')


@tool(args_schema=OrderQuery)
async def buscar_pedidos(cliente_id: int) -> dict:
    """Consulta los pedidos de un cliente por su identificador.

    Usar después de buscar_cliente para conocer pedidos, cantidad, total o
    último pedido. Devuelve datos verificados de la base de demostración.
    Si no existe el cliente, devuelve un error para pedir una aclaración.
    """
    orders = ORDERS.get(cliente_id)
    if orders is None:
        return {'error': 'Identificador inexistente. Buscar el cliente o pedir aclaración.'}
    return {'cliente_id': cliente_id, 'pedidos': orders, 'cantidad': len(orders),
            'total': sum(order['total'] for order in orders), 'ultimo': orders[-1]}


TOOLS = [buscar_cliente, buscar_pedidos]


class AgentState(MessagesState):
    pass


def build_agent(checkpointer, model=None):
    model = model or ChatOpenAI(model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'), temperature=0, max_tokens=512, timeout=180)
    bound = model.bind_tools(TOOLS)
    rules = SystemMessage(content='Respondé en español usando exclusivamente datos devueltos por las herramientas. '
        'Si recibís un nombre, primero buscá su identificador con buscar_cliente. '
        'Después consultá buscar_pedidos con ese identificador. No inventes identificadores ni importes. '
        'Si una herramienta falla, corregí los argumentos y volvé a consultar o pedí aclaración. '
        'Usá el historial para entender preguntas de seguimiento.')

    async def call_model(state: AgentState):
        response = await bound.ainvoke([rules, *state['messages']])
        return {'messages': [response]}

    graph = StateGraph(AgentState)
    graph.add_node('agent', call_model)
    graph.add_node('tools', ToolNode(TOOLS, handle_tool_errors=True))
    graph.add_edge(START, 'agent')
    graph.add_conditional_edges('agent', tools_condition)
    graph.add_edge('tools', 'agent')
    return graph.compile(checkpointer=checkpointer)


async def run_question(question: str, thread_id: str, database=ROOT / 'checkpoints.sqlite') -> dict:
    config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 10}
    async with AsyncSqliteSaver.from_conn_string(str(database)) as memory:
        app = build_agent(memory)
        return await app.ainvoke({'messages': [HumanMessage(content=question)]}, config)


def save_trace(result: dict) -> None:
    trace = [{'tipo': message.type, 'contenido': message.content,
              **({'herramientas': message.tool_calls} if getattr(message, 'tool_calls', None) else {})}
             for message in result['messages']]
    (ROOT / 'traza.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')


async def main():
    thread = f'pedidos-{uuid4().hex}'
    first = await run_question('¿Cuántos pedidos tuvo Ana y cuál fue el total?', thread)
    save_trace(first)
    print(first['messages'][-1].content, flush=True)
    result = await run_question('¿Y cuál fue el último?', thread)
    save_trace(result)
    print(result['messages'][-1].content)


if __name__ == '__main__':
    asyncio.run(main())
