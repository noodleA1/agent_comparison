# Agent Framework Walkthrough: How Each Achieves the Same Result

This document provides a detailed walkthrough of how four different agent frameworks implement the same course generation workflow. Each framework takes a different architectural approach to achieve identical results.

## The Task

Generate a 10-lesson course by:
1. Extracting the topic from user prompt
2. Researching the topic (via Jina Search API)
3. Creating a syllabus with 10 lessons
4. Looping through each lesson to research and generate detailed plans
5. Compiling the final course package

---

## Lines of Code Comparison

| Framework | Total Lines | Code Lines* | Complexity |
|-----------|-------------|-------------|------------|
| **OpenAI SDK** | 528 | 358 | Simplest |
| **LangGraph** | 565 | 400 | Medium |
| **Google ADK** | 625 | 428 | Medium-High |
| **Orchestral** | 639 | 432 | Medium |

*Code lines = excluding comments and blank lines

---

## 1. OpenAI Agents SDK (Simplest - 358 code lines)

### Architecture: Primitives + Code Orchestration

```
┌─────────────────────────────────────────────────────┐
│                    YOUR CODE                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │ Agent 1 │  │ Agent 2 │  │ Agent 3 │  ...        │
│  └────┬────┘  └────┬────┘  └────┬────┘             │
│       │            │            │                   │
│       └────────────┴────────────┘                   │
│                    │                                 │
│            Python for loop                          │
│         (you control everything)                    │
└─────────────────────────────────────────────────────┘
```

### LLM Setup

```python
from openai import OpenAI

def get_openai_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", "")
    )
```

**Simple:** Just configure the OpenAI client with a different base URL.

### Tool Definition

```python
@function_tool
def jina_search(query: str) -> str:
    """Search the web using Jina."""
    response = httpx.get(f"https://s.jina.ai/{query}")
    return response.text
```

**Simple:** Decorator marks it as a tool, docstring becomes description.

### Workflow Execution

```python
# Define specialized agents
topic_agent = Agent(name="TopicExtractor", instructions="...")
syllabus_agent = Agent(name="SyllabusCreator", instructions="...")
lesson_agent = Agent(name="LessonGenerator", instructions="...")

# Execute with explicit Python code
topic = Runner.run(topic_agent, prompt)           # Step 1
research = jina_search(topic)                      # Step 2 (tool call)
syllabus = Runner.run(syllabus_agent, research)   # Step 3

# YOUR Python loop controls iteration
for lesson in syllabus.lessons:                    # Step 4
    lesson_research = jina_search(lesson.title)
    plan = Runner.run(lesson_agent, lesson_research)
```

### Key Insight

**You write normal Python.** The framework provides `Agent` and `Runner.run()` but you control the flow with regular code. No graphs, no special loop constructs, no state machines.

### Pros
- Minimal learning curve
- Full control over execution
- Easy to debug (normal Python)
- Smallest codebase

### Cons
- Less structure for complex workflows
- No built-in checkpointing
- Manual state management

---

## 2. LangGraph (Medium - 400 code lines)

### Architecture: Graph-Based with State

```
┌─────────────────────────────────────────────────────┐
│                   StateGraph                         │
│                                                      │
│   START                                              │
│     │                                                │
│     ▼                                                │
│  ┌──────────────┐                                   │
│  │ understand   │ ◄── Node (function)               │
│  └──────┬───────┘                                   │
│         │ edge                                       │
│         ▼                                            │
│  ┌──────────────┐                                   │
│  │  research    │ ◄── Uses @tool                    │
│  └──────┬───────┘                                   │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐     ┌──────────────┐             │
│  │  syllabus    │────►│ lesson_loop  │◄─┐          │
│  └──────────────┘     └──────┬───────┘  │          │
│                              │          │           │
│                     conditional edge    │           │
│                       (should_continue?)│           │
│                              └──────────┘           │
│                                    │                │
│                                   END               │
└─────────────────────────────────────────────────────┘
```

### LLM Setup

```python
class ChatOpenAI:
    def __init__(self, model, base_url, api_key):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def invoke(self, messages, **kwargs) -> dict:
        response = self.client.chat.completions.create(...)
        return {"content": response.choices[0].message.content}
```

**LangChain pattern:** Wrapper class with `.invoke()` method.

### Tool Definition

```python
@tool
def jina_search(query: str) -> dict:
    """Search the web using Jina Search API."""
    response = httpx.get(f"https://s.jina.ai/{query}")
    return {"success": True, "results": response.text}
```

**Same decorator pattern**, but tools integrate with `ToolNode` for automatic invocation.

### State Definition (Extra Concept)

```python
class CourseState(TypedDict):
    """State flows through the entire graph."""
    prompt: str
    topic: str
    research: str
    syllabus_json: str
    current_lesson: int      # Loop counter
    lessons: List[dict]      # Accumulated results
    citations: List[str]
```

**New concept:** You must define a TypedDict that holds ALL state. Every node receives and returns this state.

### Workflow Execution

```python
# Each node is a function that takes state and returns state
def understand_node(state: CourseState) -> CourseState:
    result = llm.invoke([...])
    state["topic"] = result["content"]
    return state

def research_node(state: CourseState) -> CourseState:
    result = jina_search(state["topic"])
    state["research"] = result["results"]
    return state

# Conditional edge function
def should_continue(state: CourseState) -> Literal["continue", "end"]:
    if state["current_lesson"] < 10:
        return "continue"
    return "end"

# Graph definition (in production):
# graph = StateGraph(CourseState)
# graph.add_node("understand", understand_node)
# graph.add_edge("understand", "research")
# graph.add_conditional_edges("generate_lesson", should_continue, {...})
# app = graph.compile()

# Execution
while should_continue(state) == "continue":
    state = research_lesson_node(state)
    state = generate_lesson_node(state)
```

### Key Insight

**Control flow is declarative (edges), not imperative (code).** You define nodes and connect them with edges. The graph engine handles execution. Conditional edges replace if/else statements.

### Pros
- Visual workflow representation
- Built-in checkpointing (pause/resume)
- LangSmith integration for observability
- Good for complex branching logic

### Cons
- Steeper learning curve
- Must learn graph concepts
- State management overhead
- More boilerplate

---

## 3. Google ADK (Medium-High - 428 code lines)

### Architecture: Hierarchical Agents

```
┌─────────────────────────────────────────────────────┐
│              SequentialAgent (root)                  │
│                                                      │
│   ┌─────────────────────────────────────────────┐   │
│   │  LlmAgent: TopicAgent                        │   │
│   │    output_key="topic"  ──────────────────┐  │   │
│   └─────────────────────────────────────────────┘   │
│                      │                          │    │
│                      ▼                          │    │
│   ┌─────────────────────────────────────────────┐   │
│   │  LlmAgent: SyllabusAgent                    │   │
│   │    instruction="... {{topic}} ..."  ◄───────┘  │
│   │    output_key="syllabus"                    │   │
│   └─────────────────────────────────────────────┘   │
│                      │                               │
│                      ▼                               │
│   ┌─────────────────────────────────────────────┐   │
│   │  LoopAgent (max_iterations=10)              │   │
│   │    ┌─────────────────────────────────────┐  │   │
│   │    │ FunctionTool: jina_search           │  │   │
│   │    └─────────────────────────────────────┘  │   │
│   │    ┌─────────────────────────────────────┐  │   │
│   │    │ LlmAgent: LessonWriter              │  │   │
│   │    │   instruction="... {{research}} ..."│  │   │
│   │    └─────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### LLM Setup

```python
class GeminiModel:
    def __init__(self, model):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", ...)

    def generate_content(self, prompt, system_instruction=None):
        response = self.client.chat.completions.create(...)
        return {"text": response.choices[0].message.content}
```

**Google pattern:** `generate_content()` method instead of `invoke()`.

### Tool Definition

```python
class FunctionTool:
    def __init__(self, func, description):
        self.func = func
        self.name = func.__name__

@create_function_tool("Search the web")
def jina_search(query: str) -> str:
    response = httpx.get(f"https://s.jina.ai/{query}")
    return response.text
```

**Similar decorator**, but creates a `FunctionTool` wrapper class.

### Session State (Different from LangGraph)

```python
class SessionState:
    def __init__(self):
        self._state = {}

    def get(self, key, default=None):
        return self._state.get(key, default)

    def set(self, key, value):
        self._state[key] = value
```

**Shared state between agents** via `session.get()` and `session.set()`.

### Agent Definition (More Concepts)

```python
class LlmAgent:
    def __init__(self, name, instruction, output_key=None, tools=None):
        self.instruction = instruction  # Can contain {{template}} vars
        self.output_key = output_key    # Where to store result

    def run(self, session, context=""):
        # Template variable substitution
        prompt = self.instruction
        for key, value in session._state.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", value)

        result = model.generate_content(prompt=context, system_instruction=prompt)

        # Store in session state
        if self.output_key:
            session.set(self.output_key, result["text"])
```

**Key patterns:**
- `output_key`: Automatically stores result in session state
- `{{template}}`: Variables are substituted from session state

### Workflow Execution

```python
session = SessionState()

# Agent 1: Extract topic, store in session["topic"]
topic_agent = LlmAgent(
    name="TopicAgent",
    instruction="Extract the topic...",
    output_key="topic"  # Result stored here
)
topic_agent.run(session, prompt)

# Agent 2: Uses {{topic}} from session state
syllabus_agent = LlmAgent(
    name="SyllabusAgent",
    instruction="Create syllabus for {{topic}}...",  # Substituted!
    output_key="syllabus"
)
syllabus_agent.run(session, "")

# LoopAgent handles iteration
for i in range(10):
    session.set("loop_iteration", i)
    jina_search(...)  # FunctionTool
    lesson_writer.run(session, "")  # Uses {{lesson_research}}
```

### Key Insight

**Agents are composed hierarchically.** `SequentialAgent` runs agents in order. `LoopAgent` repeats agents. `ParallelAgent` runs agents concurrently. Data flows through `SessionState` with `output_key` pattern.

### Pros
- Clean hierarchical composition
- Built-in loop/parallel abstractions
- Template variable substitution
- Good Google Cloud integration

### Cons
- Most concepts to learn
- Gemini-optimized (less flexible)
- Session state can be confusing
- Verbose for simple tasks

---

## 4. Orchestral AI (Medium - 432 code lines)

### Architecture: Single Agent + Explicit Control

```
┌─────────────────────────────────────────────────────┐
│                 Single Agent                         │
│                                                      │
│   ┌─────────────────────────────────────────────┐   │
│   │  llm = Claude(model="claude-sonnet-4-0")    │   │
│   │       ▲                                      │   │
│   │       │ One-line provider switch:            │   │
│   │       │ llm = GPT(model="gpt-4")            │   │
│   └───────┼─────────────────────────────────────┘   │
│           │                                          │
│   ┌───────┴─────────────────────────────────────┐   │
│   │  tools = [jina_search, jina_read]           │   │
│   │  (@define_tool auto-generates schema)        │   │
│   └─────────────────────────────────────────────┘   │
│                                                      │
│   ┌─────────────────────────────────────────────┐   │
│   │  context = Context()                         │   │
│   │    .total_tokens = 15432                     │   │
│   │    .total_cost = $0.0234                     │   │
│   │    .messages = [...]                         │   │
│   └─────────────────────────────────────────────┘   │
│                                                      │
│   Execution: 100% synchronous Python                │
│   (Full stack traces, deterministic)                │
└─────────────────────────────────────────────────────┘
```

### LLM Setup (Provider-Agnostic - Key Feature)

```python
class Claude:
    def __init__(self, model="claude-sonnet-4-0"):
        model_map = {
            "claude-sonnet-4-0": "anthropic/claude-3.5-sonnet",
            "claude-opus-4-0": "anthropic/claude-opus-4-0",
        }
        self.model = model_map.get(model)
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", ...)

    def chat(self, messages, max_tokens=2000):
        response = self.client.chat.completions.create(...)
        return {"content": response.choices[0].message.content, "usage": {...}}

class GPT:  # Same interface!
    def __init__(self, model="gpt-4"):
        self.model = f"openai/{model}"
        ...
```

**Provider-agnostic:** Switch from Claude to GPT with one line: `agent.llm = GPT("gpt-4")`

### Tool Definition

```python
@define_tool("Search the web")
def jina_search(query: str) -> dict:
    """Search using Jina. Schema auto-generated from type hints."""
    response = httpx.get(f"https://s.jina.ai/{query}")
    return {"success": True, "results": response.text}
```

**Auto-schema generation:** No manual JSON schema needed. Type hints are used to generate the schema.

### Context (Cost Tracking - Key Feature)

```python
class Context:
    def __init__(self):
        self.messages = []
        self.total_cost = 0.0
        self.total_tokens = 0

    def add_cost(self, usage, cost_per_1k_input=0.003, cost_per_1k_output=0.015):
        self.total_tokens += usage["total_tokens"]
        self.total_cost += (usage["prompt_tokens"] / 1000) * cost_per_1k_input
        self.total_cost += (usage["completion_tokens"] / 1000) * cost_per_1k_output
```

**Built-in cost tracking:** Every API call updates the context. Check `agent.context.total_cost` anytime.

### Workflow Execution

```python
agent = Agent(
    llm=Claude(model="claude-sonnet-4-0"),
    tools=[jina_search, jina_read],
    system_prompt="You are a course designer..."
)

# Synchronous execution - YOUR Python code controls everything
topic = agent.run("Extract topic from: " + prompt)

# Explicit tool invocation (not implicit)
research = agent.use_tool("jina_search", query=topic)

syllabus = agent.run(f"Create syllabus for {topic}...")

# Plain Python loop
for lesson in syllabus.lessons:
    lesson_research = agent.use_tool("jina_search", query=lesson.title)
    plan = agent.run(f"Generate lesson plan: {lesson_research}")

# Check costs anytime
print(f"Total cost: ${agent.context.total_cost:.4f}")
```

### Key Insight

**Synchronous, explicit, provider-agnostic.** No async, no implicit tool calls, no graph abstractions. Tools are called explicitly via `agent.use_tool()`. You get full stack traces on errors. Switch LLM providers with one line.

### Pros
- Provider-agnostic (Claude, GPT, Gemini)
- Built-in cost tracking
- 100% synchronous (easy debugging)
- Explicit tool invocation
- Full stack traces

### Cons
- Newer framework (less ecosystem)
- No async support
- No built-in parallelism
- Requires Python 3.13+

---

## Side-by-Side Comparison

| Aspect | OpenAI SDK | LangGraph | Google ADK | Orchestral |
|--------|------------|-----------|------------|------------|
| **Mental Model** | Functions + loops | Graph + state machine | Agent hierarchy | Single orchestrator |
| **Loop Handling** | `for` loop | Conditional edge | `LoopAgent` | `for` loop |
| **State** | Local variables | `TypedDict` | `SessionState` | `Context` |
| **Tool Calls** | Direct call | `ToolNode` auto | `FunctionTool` | `agent.use_tool()` |
| **LLM Call** | `Runner.run()` | `llm.invoke()` | `model.generate_content()` | `agent.run()` |
| **Data Flow** | Return values | State mutations | `output_key` → session | Return values |
| **Execution** | Sync/Async | Async with checkpoints | Event-driven | 100% Sync |
| **Provider Lock** | OpenAI-focused | LangChain ecosystem | Gemini-optimized | Provider-agnostic |
| **Cost Tracking** | Manual | Via LangSmith | Manual | Built-in |
| **Debugging** | Normal Python | Graph traces | Event logs | Full stack traces |

---

## Decision Guide

### Choose OpenAI SDK if you want:
- **Minimal abstractions** - just Python with some helpers
- **Quick start** - lowest learning curve
- **Full control** - you decide everything
- **OpenAI ecosystem** - designed for OpenAI models

### Choose LangGraph if you want:
- **Visual workflows** - graphs are easier to reason about
- **Checkpointing** - pause and resume long-running workflows
- **Complex branching** - conditional edges handle complex logic
- **Observability** - LangSmith integration

### Choose Google ADK if you want:
- **Hierarchical composition** - agents containing agents
- **Google Cloud integration** - Vertex AI, Cloud Run
- **Built-in patterns** - LoopAgent, ParallelAgent, SequentialAgent
- **Gemini optimization** - best performance with Gemini models

### Choose Orchestral if you want:
- **Provider switching** - easily swap Claude/GPT/Gemini
- **Cost awareness** - built-in tracking for research/production
- **Reproducibility** - synchronous, deterministic execution
- **Simple debugging** - full Python stack traces

---

## Gap-Driven Refinement: Framework Pattern Showcase

The most revealing comparison is how each framework implements **gap-driven refinement** - the same goal achieved through different architectural patterns. After generating a course, a "student simulation" agent identifies gaps in the content. If issues are found, the course is refined.

### The Challenge

After generating lessons, the student simulation might find:
- 5 content gaps that need addressing
- `ready_for_publication: False`

Each framework must:
1. Detect that refinement is needed
2. Incorporate gap context into the refinement
3. Re-generate affected lessons
4. Track costs separately

---

### LangGraph: TIME-TRAVEL Pattern

```
┌─────────────────────────────────────────────────────────┐
│                 TIME-TRAVEL CHECKPOINTING                │
│                                                          │
│   1. Save checkpoint BEFORE lesson generation           │
│      ┌──────────────────────────────────────────┐       │
│      │  checkpointer.save("pre_lessons")         │       │
│      └──────────────────────────────────────────┘       │
│                        │                                 │
│                        ▼                                 │
│   2. Generate lessons normally                          │
│                        │                                 │
│                        ▼                                 │
│   3. Gap assessment finds issues                        │
│                        │                                 │
│   4. ═══════════ CONDITIONAL EDGE ═══════════           │
│      │                                                   │
│      │  def should_refine(state) -> "refine" | "end"    │
│      │      if gaps >= 2 and not ready:                 │
│      │          return "refine"  ←── TRIGGERS REWIND    │
│      │                                                   │
│   5. REWIND: Load pre_lessons checkpoint                │
│      ┌──────────────────────────────────────────┐       │
│      │  state = checkpointer.load("pre_lessons") │       │
│      │  state["gap_context"] = gap_data          │       │
│      └──────────────────────────────────────────┘       │
│                        │                                 │
│   6. REPLAY with gap awareness                          │
│      for lesson in syllabus:                            │
│          refined = llm.invoke(lesson + gap_context)     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key Mechanism:** LangGraph's `MemorySaver` checkpointer saves named snapshots. When refinement is needed, we **time-travel** back to the pre-lesson state, inject gap context, and replay the lesson generation with this new awareness.

**Code Pattern:**
```python
# Save checkpoint before lesson loop
self.checkpointer.save(thread_id, state, "pre_lessons")

# After gap assessment - conditional edge
def should_refine_after_gaps(state):
    if len(state["gap_assessment"]["gaps_found"]) >= 2:
        return "refine"
    return "end"

# Refinement node
def gap_driven_refinement_node(state):
    # TIME-TRAVEL: Load pre-lesson state
    pre_state = self.checkpointer.load(thread_id, "pre_lessons")
    pre_state["gap_context"] = state["gap_assessment"]
    # Replay with awareness
    for lesson in pre_state["syllabus"]["lessons"]:
        refined = llm.invoke(f"Given gaps: {pre_state['gap_context']}, improve: {lesson}")
```

---

### OpenAI SDK: HANDOFF Pattern

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT HANDOFF                         │
│                                                          │
│   ┌─────────────────────┐                               │
│   │ StudentSimulator    │                               │
│   │ Agent               │                               │
│   │   "Review course"   │                               │
│   └──────────┬──────────┘                               │
│              │                                           │
│              │ Gap data in structured output            │
│              │                                           │
│              ▼                                           │
│   ┌─────────────────────────────────────────────┐       │
│   │  if should_refine(gap_data):                │       │
│   │      HANDOFF ──────────────────────────►    │       │
│   └─────────────────────────────────────────────┘       │
│                                          │               │
│                                          ▼               │
│                        ┌─────────────────────────┐      │
│                        │ LessonRefiner           │      │
│                        │ Agent                   │      │
│                        │                         │      │
│                        │ Instructions:           │      │
│                        │ "Given gap context,     │      │
│                        │  improve each lesson"   │      │
│                        │                         │      │
│                        │ Context passed via      │      │
│                        │ structured message      │      │
│                        └─────────────────────────┘      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key Mechanism:** OpenAI SDK uses **agent-to-agent handoffs** where one agent passes control (and context) to another specialized agent. The gap assessment context is structured and passed explicitly.

**Code Pattern:**
```python
# Define specialized refiner agent
lesson_refiner_agent = Agent(
    name="LessonRefiner",
    instructions="You receive gap context from StudentSimulator. Improve lessons accordingly.",
    model=model
)

# After student simulation
if should_refine(gap_data):
    gap_context = f"""
    HANDOFF from StudentSimulator:
    GAPS: {gap_data['gaps_found']}
    MISSING: {gap_data['missing_prerequisites']}
    """

    for lesson in lessons:
        # Handoff with structured context
        refined = Runner.run(lesson_refiner_agent, f"{gap_context}\n\nLESSON:\n{lesson}")
```

---

### Google ADK: ESCALATION Pattern

```
┌─────────────────────────────────────────────────────────┐
│                  LOOPAGENT ESCALATION                    │
│                                                          │
│   ┌─────────────────────────────────────────────┐       │
│   │  Gap Assessment Agent                        │       │
│   │    ↓                                         │       │
│   │  session.set("gaps_found", 5)                │       │
│   │  session.set("ready_for_publication", False) │       │
│   └──────────────────────────────────────────────┘       │
│                        │                                 │
│                        ▼                                 │
│   ┌─────────────────────────────────────────────┐       │
│   │  ESCALATION CHECK                            │       │
│   │                                              │       │
│   │  if session.get("gaps_found") >= 2:          │       │
│   │      session.set("escalation_key", True)  ←──│       │
│   │                                 │            │       │
│   └─────────────────────────────────│────────────┘       │
│                                     │                    │
│                                     ▼                    │
│   ┌─────────────────────────────────────────────┐       │
│   │  LoopAgent (triggered by escalation)        │       │
│   │  ┌───────────────────────────────────────┐  │       │
│   │  │  max_iterations = 10                  │  │       │
│   │  │  sub_agents = [LessonRefiner]         │  │       │
│   │  │                                       │  │       │
│   │  │  for each lesson:                     │  │       │
│   │  │    gap_context = session.get("gaps")  │  │       │
│   │  │    refined = refiner.run(lesson)      │  │       │
│   │  └───────────────────────────────────────┘  │       │
│   └─────────────────────────────────────────────┘       │
│                        │                                 │
│   session.set("escalation_key", False)  ←── Clear       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key Mechanism:** Google ADK's `LoopAgent` can be triggered by an **escalation signal** in session state. Setting `escalation_key=True` activates the refinement loop, which iterates through lessons using the `LoopAgent` primitive.

**Code Pattern:**
```python
# After gap assessment
if session.get("gaps_found", 0) >= 2:
    # Set escalation signal
    session.set("refinement_escalation", True)

    # LoopAgent for refinement
    for i, lesson in enumerate(lessons):
        session.set("current_lesson_index", i)
        gap_context = f"GAPS: {session.get('gaps_found')}"

        # LessonRefiner agent runs within escalation context
        lesson_refiner.run(session, f"{gap_context}\n{lesson}")

    # Clear escalation
    session.set("refinement_escalation", False)
```

---

### Orchestral: HOOK + SUBAGENT Pattern

```
┌─────────────────────────────────────────────────────────┐
│                   HOOK + SUBAGENT                        │
│                                                          │
│   ┌─────────────────────────────────────────────┐       │
│   │  Gap Assessment (via Student Subagent)       │       │
│   │    → gaps_found: 5                           │       │
│   │    → ready: False                            │       │
│   └──────────────────────────────────────────────┘       │
│                        │                                 │
│                        ▼                                 │
│   ╔═════════════════════════════════════════════╗       │
│   ║  REFINEMENT HOOK                            ║       │
│   ║  (intercepts workflow execution)            ║       │
│   ║                                             ║       │
│   ║  class RefinementHook:                      ║       │
│   ║      threshold = 2                          ║       │
│   ║                                             ║       │
│   ║      def should_refine(gap_data):           ║       │
│   ║          issues = len(gaps) + len(missing)  ║       │
│   ║          if issues >= self.threshold:       ║       │
│   ║              return INTERRUPT  ←────────────║       │
│   ║                           │                 ║       │
│   ╚═══════════════════════════│═════════════════╝       │
│                               │                          │
│                               ▼                          │
│   ┌─────────────────────────────────────────────┐       │
│   │  LESSON REFINER SUBAGENT                    │       │
│   │  (spawned by hook interrupt)                │       │
│   │                                             │       │
│   │  class LessonRefinerSubagent:               │       │
│   │      def __call__(lesson, gap_context):     │       │
│   │          refiner = Agent(llm=self.llm, ...) │       │
│   │          return refiner.run(prompt)         │       │
│   │                                             │       │
│   │  for lesson in lessons:                     │       │
│   │      refined = subagent(lesson, gaps)       │       │
│   └─────────────────────────────────────────────┘       │
│                                                          │
│   costs_by_phase["gap_refinement"] = subagent.cost      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key Mechanism:** Orchestral uses a **hook** to intercept the workflow when certain conditions are met, then delegates to a **subagent** for the actual refinement work. This keeps concerns separated and costs tracked per phase.

**Code Pattern:**
```python
class RefinementHook:
    def __init__(self, threshold=2):
        self.threshold = threshold

    def should_refine(self, gap_data) -> bool:
        issues = len(gap_data.get("gaps_found", [])) + len(gap_data.get("missing", []))
        return issues >= self.threshold

class LessonRefinerSubagent:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, lesson, gap_context, context):
        refiner = Agent(llm=self.llm, system_prompt="Improve this lesson...")
        return refiner.run(f"Gap context:\n{gap_context}\n\nLesson:\n{lesson}")

# In main workflow
if refinement_hook.should_refine(gap_data):
    for lesson in lessons:
        refined = refiner_subagent(lesson, gap_context, context)
    context.costs_by_phase["gap_refinement"] = subagent_cost
```

---

### Pattern Comparison Summary

| Framework | Pattern | Trigger | Mechanism | Best For |
|-----------|---------|---------|-----------|----------|
| **LangGraph** | TIME-TRAVEL | Conditional edge | Checkpointer rewind | Complex stateful workflows |
| **OpenAI SDK** | HANDOFF | Code check | Agent-to-agent | Agent composition |
| **Google ADK** | ESCALATION | Session signal | LoopAgent activation | Hierarchical teams |
| **Orchestral** | HOOK+SUBAGENT | Hook intercept | Subagent delegation | Cost-tracked pipelines |

---

## Final Thoughts

All four frameworks can accomplish the same task. The difference is in:

1. **How you think about the problem** (imperative vs declarative)
2. **How much control you want** (explicit vs implicit)
3. **What ecosystem you're in** (OpenAI, LangChain, Google, provider-agnostic)
4. **What features matter** (checkpointing, cost tracking, debugging)

For this course generation task:
- **OpenAI SDK** is the most straightforward
- **LangGraph** adds value if you need checkpointing or complex branching
- **Google ADK** shines with hierarchical agent teams
- **Orchestral** is best for cost-conscious, provider-flexible applications
