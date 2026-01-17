# Agent Framework Comparison

Compare four AI agent frameworks on the same course generation task.

## Frameworks

| Framework | Pattern | Key Feature |
|-----------|---------|-------------|
| **OpenAI SDK** | Primitives | Minimal abstractions, code orchestration |
| **LangGraph** | Graph-based | Conditional edges, TypedDict state |
| **Google ADK** | Hierarchical | SequentialAgent, LoopAgent |
| **Orchestral** | Synchronous | Provider-agnostic, cost tracking |

## Quick Start

```bash
pip install -r requirements.txt
cp .env.template .env
# Add OPENROUTER_API_KEY to .env
```

### Run from CLI

```bash
python cli_runner.py "Python basics for beginners"
python cli_runner.py --frameworks langgraph,openai "Machine Learning"
python cli_runner.py --load results/latest
```

### Run Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

## Project Structure

```
├── cli_runner.py              # Main CLI tool
├── langgraph_impl/agent.py    # LangGraph implementation
├── openai_sdk_impl/agent.py   # OpenAI SDK implementation
├── google_adk_impl/agent.py   # Google ADK implementation
├── orchestral_impl/agent.py   # Orchestral implementation
├── common/models.py           # Shared Pydantic models
├── streamlit_app/app.py       # Web UI
└── FRAMEWORK_WALKTHROUGH.md   # Detailed architecture docs
```

## APIs Used

- **OpenRouter** - LLM access (configurable model)
- **Jina** - Web search and URL-to-markdown

## Framework Comparison

| Aspect | OpenAI SDK | LangGraph | Google ADK | Orchestral |
|--------|------------|-----------|------------|------------|
| Loop | Python `for` | Conditional edges | `LoopAgent` | Python `for` |
| State | Dict | TypedDict | SessionState | Context |
| Abstraction | Low | Medium | High | Low |

See [FRAMEWORK_WALKTHROUGH.md](FRAMEWORK_WALKTHROUGH.md) for details.
