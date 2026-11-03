# Preentrega 5

Agente asíncrono con StateGraph, MessagesState, herramientas propias y rutas condicionales por tool calling. SQLite conserva los mensajes por thread_id mediante AsyncSqliteSaver, la variante asíncrona de SqliteSaver. Límite de recursión: 10.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.1
python agent.py
pytest -q
```

La demostración consulta pedidos y calcula el total mediante dos herramientas. La segunda interacción recupera el cliente anterior después de reabrir SQLite. `traza.json` registra mensajes y llamadas a herramientas, sin razonamiento interno del modelo.

Mantener Ollama iniciado. La configuración gratuita usa su interfaz compatible con ChatOpenAI. Para OpenAI, cambiar clave y modelo y quitar OPENAI_BASE_URL.

Las pruebas automáticas comprueban dos llamadas a herramientas y la recuperación de checkpoints con un modelo de prueba.
