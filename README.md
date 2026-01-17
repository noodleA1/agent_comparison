# Agent Framework Comparison

Compare four AI agent frameworks on the same course generation task.

**[Read the detailed Framework Walkthrough](FRAMEWORK_WALKTHROUGH.md)** for architecture diagrams, code patterns, and decision guidance.

## Frameworks

| Framework | Pattern | Lines of Code | Key Feature |
|-----------|---------|---------------|-------------|
| **OpenAI SDK** | Primitives | 358 | Code orchestration, minimal abstractions |
| **LangGraph** | Graph-based | 400 | Conditional edges, TypedDict state |
| **Google ADK** | Hierarchical | 428 | SequentialAgent, LoopAgent, template variables |
| **Orchestral** | Synchronous | 432 | Provider-agnostic, built-in cost tracking |

## Architecture

Each framework implementation is **self-contained** with its own:
- LLM client setup (OpenRouter via OpenAI SDK)
- Tool definitions (Jina Search/Reader)
- State management patterns
- Workflow orchestration

This allows direct comparison of how each framework handles the same requirements.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.template .env
# Edit .env and add your keys:
# - OPENROUTER_API_KEY (required)
# - JINA_API_KEY (optional, for research)
```

### 3. Run Smoke Test

```bash
python run.py
```

### 4. Start Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

## Project Structure

```
course-generator-comparison/
├── common/
│   └── models.py              # Shared Pydantic data models
├── langgraph_impl/
│   └── agent.py               # Self-contained LangGraph (400 LOC)
├── openai_sdk_impl/
│   └── agent.py               # Self-contained OpenAI SDK (358 LOC)
├── google_adk_impl/
│   └── agent.py               # Self-contained Google ADK (428 LOC)
├── orchestral_impl/
│   └── agent.py               # Self-contained Orchestral (432 LOC)
├── streamlit_app/
│   └── app.py                 # Comparison UI
├── FRAMEWORK_WALKTHROUGH.md   # Detailed architecture comparison
├── run.py                     # Smoke test runner
├── requirements.txt
└── .env.template
```

Each `agent.py` contains:
- Framework-specific LLM setup patterns
- Idiomatic tool definitions (`@tool`, `@function_tool`, `@define_tool`, etc.)
- State management (TypedDict, SessionState, Context, etc.)
- Complete workflow implementation

## Workflow

Each framework implements the same workflow:

```
Prompt → Understand → Research (Jina) → Syllabus (10 lessons)
      → Loop: Research lesson → Generate lesson plan
      → Output: Complete course package
```

## API Usage

All frameworks use the same external APIs (each with their own integration):
- **OpenRouter** for LLM access (any model via unified API)
- **Jina Reader** (r.jina.ai) for URL-to-markdown
- **Jina Search** (s.jina.ai) for web search

## Metrics Tracked

- Execution time
- Token usage
- API calls
- Jina research calls
- Errors

## Key Comparisons

| Aspect | OpenAI SDK | LangGraph | Google ADK | Orchestral |
|--------|------------|-----------|------------|------------|
| **Loop Pattern** | Python `for` | Conditional edges | `LoopAgent` | Python `for` |
| **State** | Dict | TypedDict | SessionState | Context |
| **Tool Decorator** | `@function_tool` | `@tool` | `FunctionTool` | `@define_tool` |
| **Abstraction Level** | Low | Medium | High | Low |

See [FRAMEWORK_WALKTHROUGH.md](FRAMEWORK_WALKTHROUGH.md) for detailed analysis.
