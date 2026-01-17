"""
LangGraph Course Generator - ENHANCED IMPLEMENTATION

Demonstrates LangGraph's unique patterns from the Pregel-inspired architecture:

FRAMEWORK DIFFERENTIATORS:
1. Send API - Dynamic fan-out for parallel lesson research (map-reduce)
2. interrupt() - Human-in-the-loop with state persistence
3. Conditional Edges - Quality loop with dynamic routing
4. Checkpointer - Built-in state persistence for recovery
5. Command API - Dynamic node control and state updates
6. TypedDict State - Strongly typed state management

ENHANCED FEATURES:
- Parallel research via Send API (LangGraph's unique map-reduce)
- Quality loop using conditional edges with dynamic routing
- Human approval via interrupt() pattern
- Gap assessment as separate graph node
- Cost tracking via state reducers
"""
import os
import json
import httpx
from typing import TypedDict, List, Literal, Callable, Annotated, Any, Dict
from datetime import datetime
from dataclasses import dataclass, field

# ============================================================================
# Cost Tracking with State Reducers (LangGraph Pattern)
# ============================================================================
# LangGraph uses Annotated fields with reducers for state accumulation.
# We'll track costs through state updates.

# 2026 OpenRouter pricing (per 1M tokens)
PRICING = {
    "deepseek/deepseek-v3.2": {"input": 0.25, "output": 0.38},  # Best value 2026
    "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.0},  # 1M context!
    "anthropic/claude-sonnet-4": {"input": 3.0, "output": 15.0},  # Premium
}

CHEAP_MODEL = "deepseek/deepseek-v3.2"
BALANCED_MODEL = "google/gemini-3-flash-preview"


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost based on OpenRouter pricing."""
    pricing = PRICING.get(model, PRICING[BALANCED_MODEL])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


# ============================================================================
# LangGraph-style LLM Setup (using OpenRouter)
# ============================================================================

from openai import OpenAI


class ChatOpenAI:
    """
    LangChain-style ChatOpenAI wrapper configured for OpenRouter.

    KEY LANGGRAPH PATTERN: Model configuration for LangChain ecosystem.
    Supports multiple models for cost optimization.
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.7,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str = None
    ):
        self.model = model or BALANCED_MODEL
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY", "")
        )
        self._last_usage = {}

    def invoke(self, messages: List[dict], **kwargs) -> dict:
        """Invoke the LLM with messages."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=kwargs.get("max_tokens", 2000)
        )

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        total_tokens = response.usage.total_tokens if response.usage else 0
        cost = calculate_cost(self.model, input_tokens, output_tokens)

        self._last_usage = {
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "model": self.model
        }

        return {
            "content": response.choices[0].message.content,
            "usage": self._last_usage
        }

    @property
    def last_usage(self) -> dict:
        return self._last_usage


# ============================================================================
# LangGraph-style Tool Definitions (Jina Integration)
# ============================================================================

def tool(func):
    """
    Simulates LangChain's @tool decorator.

    In production, this would auto-generate JSON schema from type hints
    and register the tool for use in ToolNode.
    """
    func.is_tool = True
    func.name = func.__name__
    func.description = func.__doc__ or ""
    return func


@tool
def jina_search(query: str) -> dict:
    """Search the web using Jina Search API."""
    api_key = os.getenv("JINA_API_KEY", "")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = httpx.get(
            f"https://s.jina.ai/{query}",
            headers=headers,
            timeout=30.0
        )
        if response.status_code == 200:
            return {"success": True, "results": response.text[:10000]}
        return {"success": False, "error": f"Status {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def jina_read(url: str) -> dict:
    """Read URL content as markdown using Jina Reader API."""
    api_key = os.getenv("JINA_API_KEY", "")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = httpx.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=30.0
        )
        if response.status_code == 200:
            return {"success": True, "content": response.text[:15000]}
        return {"success": False, "error": f"Status {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# LangGraph State Definition with Cost Tracking
# ============================================================================

class CostState(TypedDict):
    """Cost tracking portion of state."""
    research_cost: float
    syllabus_cost: float
    quality_loop_cost: float
    lesson_generation_cost: float
    gap_assessment_cost: float
    gap_refinement_cost: float  # NEW: Gap-driven refinement via time-travel
    total_cost: float
    total_tokens: int


class CourseState(TypedDict):
    """
    Enhanced TypedDict state that flows through the graph.

    KEY LANGGRAPH PATTERN: TypedDict with Annotated fields for reducers.
    In production LangGraph, you'd use:
        messages: Annotated[list, add_messages]
        costs: Annotated[float, operator.add]

    State is passed between nodes and persisted by checkpointers.
    """
    # Core state
    prompt: str
    topic: str
    research: str
    parallel_research: List[str]  # Results from Send API fan-out
    syllabus_json: str
    current_lesson: int
    lessons: List[dict]
    citations: List[str]
    console_log: List[str]

    # Quality loop state
    quality_score: float
    quality_feedback: str
    quality_issues: List[str]
    quality_iteration: int

    # Human approval state (for interrupt())
    awaiting_approval: bool
    approved: bool

    # Gap assessment state
    gap_assessment: dict

    # Gap-driven refinement state (TIME-TRAVEL PATTERN)
    refinement_iteration: int  # Track refinement cycles
    gap_context: str  # Accumulated gap insights for replay
    refined_lessons: List[dict]  # Lessons after gap-driven refinement

    # Cost tracking state
    costs: CostState

    # Metrics
    metrics: dict


# ============================================================================
# Send API Simulation (LangGraph's Map-Reduce Pattern)
# ============================================================================

class Send:
    """
    Simulates LangGraph's Send API for dynamic fan-out.

    KEY LANGGRAPH DIFFERENTIATOR: Send allows dynamic creation of
    parallel execution branches at runtime. This is perfect for
    processing multiple items (lessons, research queries) in parallel.

    In production LangGraph:
        def route_research(state):
            return [Send("research_node", {"topic": t}) for t in topics]
    """

    def __init__(self, node: str, state_update: dict):
        self.node = node
        self.state_update = state_update

    def __repr__(self):
        return f"Send(node={self.node}, update={self.state_update})"


# ============================================================================
# Interrupt Simulation (LangGraph's Human-in-the-Loop Pattern)
# ============================================================================

class InterruptSignal(Exception):
    """
    Simulates LangGraph's interrupt() for human-in-the-loop.

    KEY LANGGRAPH DIFFERENTIATOR: interrupt() pauses graph execution
    and returns control to the user. Combined with checkpointing,
    this allows long-running workflows with human oversight.

    In production LangGraph:
        if not state["approved"]:
            interrupt("Please review the syllabus")
    """

    def __init__(self, message: str, state: dict):
        self.message = message
        self.state = state
        super().__init__(message)


def interrupt(message: str):
    """Create an interrupt signal (LangGraph pattern)."""
    raise InterruptSignal(message, {})


# ============================================================================
# Checkpointer Simulation (LangGraph's Persistence Pattern)
# ============================================================================

class MemorySaver:
    """
    Simulates LangGraph's MemorySaver checkpointer.

    KEY LANGGRAPH DIFFERENTIATOR: Built-in persistence for state recovery.
    Enables pause/resume of long-running workflows.

    TIME-TRAVEL CAPABILITY: Checkpointer enables "rewinding" to previous states
    and replaying with new context. Perfect for gap-driven refinement where
    we can replay lesson generation with gap insights.

    In production LangGraph:
        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)

        # Time-travel: rewind to checkpoint and replay
        config = {"configurable": {"thread_id": "123", "checkpoint_id": "abc"}}
        app.invoke(new_state, config)
    """

    def __init__(self):
        self.checkpoints: Dict[str, Dict[str, CourseState]] = {}  # thread_id -> {checkpoint_id -> state}
        self._checkpoint_counter: Dict[str, int] = {}

    def save(self, thread_id: str, state: CourseState, checkpoint_name: str = None) -> str:
        """
        Save state checkpoint with optional named checkpoint.

        Returns checkpoint_id for time-travel.
        """
        if thread_id not in self.checkpoints:
            self.checkpoints[thread_id] = {}
            self._checkpoint_counter[thread_id] = 0

        self._checkpoint_counter[thread_id] += 1
        checkpoint_id = checkpoint_name or f"cp_{self._checkpoint_counter[thread_id]}"

        # Deep copy state for immutability
        import copy
        self.checkpoints[thread_id][checkpoint_id] = copy.deepcopy(state)

        return checkpoint_id

    def load(self, thread_id: str, checkpoint_id: str = None) -> CourseState:
        """
        Load state from checkpoint.

        TIME-TRAVEL: If checkpoint_id specified, loads that specific checkpoint.
        This enables rewinding to any previous state.
        """
        if thread_id not in self.checkpoints:
            return None

        if checkpoint_id:
            return self.checkpoints[thread_id].get(checkpoint_id)

        # Return latest checkpoint
        if self.checkpoints[thread_id]:
            latest_id = f"cp_{self._checkpoint_counter[thread_id]}"
            return self.checkpoints[thread_id].get(latest_id)
        return None

    def list_checkpoints(self, thread_id: str) -> List[str]:
        """List all checkpoint IDs for a thread."""
        if thread_id not in self.checkpoints:
            return []
        return list(self.checkpoints[thread_id].keys())

    def list_threads(self) -> List[str]:
        """List all saved thread IDs."""
        return list(self.checkpoints.keys())


# ============================================================================
# LangGraph Course Generator with Enhanced Patterns
# ============================================================================

class LangGraphCourseGenerator:
    """
    Enhanced LangGraph-style course generator demonstrating unique patterns.

    GRAPH STRUCTURE:

        ┌─────────────────────────────────────────────────────────────────┐
        │                          START                                   │
        └─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │                    understand_node                               │
        │                    (extract topic)                               │
        └─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │              parallel_research_node (Send API)                   │
        │   ┌─────────────┬─────────────┬─────────────┐                   │
        │   │ Send(topic1)│ Send(topic2)│ Send(topic3)│  ◄── DIFFERENTIATOR│
        │   └─────────────┴─────────────┴─────────────┘                   │
        │              Map-Reduce: Fan-out/Gather                          │
        └─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │                    syllabus_node                                 │
        │                   (create syllabus)                              │
        └─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │               QUALITY LOOP (Conditional Edge)                    │
        │  ┌────────────────────────────────────────────────────────────┐ │
        │  │              quality_check_node                             │ │
        │  │              (evaluate score)                               │ │
        │  └────────────────────────────────────────────────────────────┘ │
        │                          │                                       │
        │            ┌─────────────┴─────────────┐                        │
        │            │                           │                        │
        │      score < 0.8               score >= 0.8                     │
        │            │                           │                        │
        │            ▼                           ▼                        │
        │  ┌──────────────────┐      ┌──────────────────┐                │
        │  │  refine_node     │      │  approval_node   │                │
        │  │  (improve)       │      │  interrupt()     │◄── DIFFERENTIATOR
        │  └──────────────────┘      └──────────────────┘                │
        │            │                           │                        │
        │            └───────────┬───────────────┘                        │
        └────────────────────────│────────────────────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │                LESSON LOOP (Conditional Edge)                    │
        │   research_lesson_node ──► generate_lesson_node                  │
        │              │                       │                           │
        │              └─── should_continue? ──┘                          │
        └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │                    gap_assessment_node                           │
        │                   (student simulation)                           │
        └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │                           END                                    │
        └─────────────────────────────────────────────────────────────────┘

    KEY PATTERNS DEMONSTRATED:
    - Send API for parallel fan-out (map-reduce)
    - interrupt() for human-in-the-loop
    - Conditional edges for dynamic routing
    - Checkpointer for state persistence
    - TypedDict state with cost tracking
    """

    def __init__(self, callback: Callable[[str], None] = None):
        self.callback = callback or print

        # LangGraph-style LLM initialization
        self.cheap_llm = ChatOpenAI(model=CHEAP_MODEL, temperature=0.7)
        self.balanced_llm = ChatOpenAI(model=BALANCED_MODEL, temperature=0.7)

        # Tools registered for ToolNode
        self.tools = [jina_search, jina_read]

        # Checkpointer for state persistence (LangGraph pattern)
        self.checkpointer = MemorySaver()

        # Metrics tracking
        self.api_calls = 0
        self.jina_calls = 0
        self.start_time = None
        self.end_time = None
        self.errors = []

    def _log(self, state: CourseState, msg: str):
        """Add to console log (captured by LangSmith in production)."""
        full_msg = f"[LangGraph] {msg}"
        state["console_log"].append(full_msg)
        print(full_msg)
        if self.callback:
            self.callback(full_msg)

    def _add_cost(self, state: CourseState, phase: str, usage: dict):
        """Add cost to state (LangGraph reducer pattern)."""
        cost = usage.get("cost", 0)
        tokens = usage.get("total_tokens", 0)

        if phase == "research":
            state["costs"]["research_cost"] += cost
        elif phase == "syllabus":
            state["costs"]["syllabus_cost"] += cost
        elif phase == "quality_loop":
            state["costs"]["quality_loop_cost"] += cost
        elif phase == "lessons":
            state["costs"]["lesson_generation_cost"] += cost
        elif phase == "gap_assessment":
            state["costs"]["gap_assessment_cost"] += cost
        elif phase == "gap_refinement":
            state["costs"]["gap_refinement_cost"] += cost

        state["costs"]["total_cost"] += cost
        state["costs"]["total_tokens"] += tokens

    def _parse_json(self, content: str) -> dict:
        """Parse JSON from LLM response."""
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # NODE: understand_node
    # ─────────────────────────────────────────────────────────────────────────
    def understand_node(self, state: CourseState) -> CourseState:
        """Extract course topic from user prompt."""
        self._log(state, "\n┌─ NODE: understand_node")
        self._log(state, "│  Extracting topic from prompt...")

        messages = [
            {"role": "system", "content": "Extract the course topic. Return only the topic name."},
            {"role": "user", "content": state["prompt"]}
        ]

        result = self.cheap_llm.invoke(messages, max_tokens=100)
        self.api_calls += 1
        self._add_cost(state, "research", result["usage"])

        state["topic"] = result.get("content", state["prompt"]).strip()
        self._log(state, f"│  → Topic: {state['topic']}")
        self._log(state, f"└─ Cost: ${state['costs']['total_cost']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # NODE: parallel_research_node (Send API Pattern)
    # ─────────────────────────────────────────────────────────────────────────
    def parallel_research_node(self, state: CourseState) -> CourseState:
        """
        Research topic using Send API for parallel fan-out.

        KEY LANGGRAPH DIFFERENTIATOR: Send API allows dynamic parallel branches.
        """
        self._log(state, "\n┌─ NODE: parallel_research_node (Send API)")
        self._log(state, "│  LANGGRAPH PATTERN: Dynamic fan-out via Send API")

        topic = state["topic"]

        # Define parallel research queries (would be Send() in real LangGraph)
        research_queries = [
            f"comprehensive tutorial {topic}",
            f"best practices {topic}",
            f"hands-on projects examples {topic}"
        ]

        self._log(state, f"│  Creating {len(research_queries)} parallel Send branches:")
        for i, query in enumerate(research_queries):
            self._log(state, f"│    Send({i+1}): '{query[:40]}...'")

        # Execute parallel research (simulating Send API fan-out/gather)
        import concurrent.futures

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(jina_search, q): q for q in research_queries}
            for future in concurrent.futures.as_completed(futures):
                query = futures[future]
                try:
                    result = future.result()
                    if result["success"]:
                        results.append(result["results"][:3000])
                        state["citations"].append(f"Research: {query[:30]}")
                        self.jina_calls += 1
                except Exception as e:
                    results.append(f"Error: {str(e)}")

        # Gather results (map-reduce pattern)
        state["parallel_research"] = results
        state["research"] = "\n\n---\n\n".join(results)

        self._log(state, f"│  → Gathered {len(results)} research results")
        self._log(state, f"│  → Total research: {len(state['research'])} chars")
        self._log(state, f"└─ Cost: ${state['costs']['total_cost']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # NODE: syllabus_node
    # ─────────────────────────────────────────────────────────────────────────
    def syllabus_node(self, state: CourseState) -> CourseState:
        """Create 10-lesson syllabus structure."""
        self._log(state, "\n┌─ NODE: syllabus_node")
        self._log(state, "│  Creating initial syllabus...")

        messages = [
            {"role": "system", "content": """Create a comprehensive 10-lesson course syllabus.
Return ONLY valid JSON:
{
  "course_title": "...",
  "course_objective": "...",
  "target_audience": "...",
  "difficulty_level": "...",
  "lessons": [
    {"lesson_number": 1, "title": "...", "objectives": ["..."], "topics": ["..."]}
  ]
}"""},
            {"role": "user", "content": f"Topic: {state['topic']}\n\nResearch:\n{state['research'][:4000]}"}
        ]

        result = self.balanced_llm.invoke(messages, max_tokens=2000)
        self.api_calls += 1
        self._add_cost(state, "syllabus", result["usage"])

        content = result.get("content", "{}")
        state["syllabus_json"] = content

        syllabus_data = self._parse_json(content)
        num_lessons = len(syllabus_data.get("lessons", []))

        self._log(state, f"│  → Created syllabus with {num_lessons} lessons")
        self._log(state, f"└─ Cost: ${state['costs']['total_cost']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # NODE: quality_check_node (Conditional Edge Entry)
    # ─────────────────────────────────────────────────────────────────────────
    def quality_check_node(self, state: CourseState) -> CourseState:
        """Evaluate syllabus quality for conditional routing."""
        self._log(state, "\n┌─ NODE: quality_check_node (Conditional Edge)")
        self._log(state, "│  LANGGRAPH PATTERN: Evaluating for conditional routing")

        state["quality_iteration"] += 1
        self._log(state, f"│  Iteration: {state['quality_iteration']}")

        messages = [
            {"role": "system", "content": """Evaluate the syllabus quality on a scale of 0.0 to 1.0.

Consider: learning progression, objective clarity, topic coverage, practical applicability.

Return ONLY valid JSON:
{
  "score": 0.85,
  "feedback": "...",
  "issues": ["...", "..."]
}"""},
            {"role": "user", "content": f"Evaluate:\n{state['syllabus_json'][:3000]}"}
        ]

        result = self.cheap_llm.invoke(messages, max_tokens=500)
        self.api_calls += 1
        self._add_cost(state, "quality_loop", result["usage"])

        quality_data = self._parse_json(result.get("content", "{}"))
        state["quality_score"] = quality_data.get("score", 0.7)
        state["quality_feedback"] = quality_data.get("feedback", "")
        state["quality_issues"] = quality_data.get("issues", [])

        self._log(state, f"│  → Quality score: {state['quality_score']:.2f}")
        self._log(state, f"│  → Feedback: {state['quality_feedback'][:50]}...")
        self._log(state, f"└─ Cost: ${state['costs']['total_cost']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITIONAL EDGE: should_refine
    # ─────────────────────────────────────────────────────────────────────────
    def should_refine(self, state: CourseState) -> Literal["refine", "approve"]:
        """
        Conditional edge function for quality loop.

        KEY LANGGRAPH PATTERN: Conditional edges enable dynamic routing.

        In production LangGraph:
            graph.add_conditional_edges(
                "quality_check",
                should_refine,
                {"refine": "refine_node", "approve": "approval_node"}
            )
        """
        if state["quality_score"] >= 0.8:
            return "approve"
        if state["quality_iteration"] >= 3:
            return "approve"  # Max iterations reached
        return "refine"

    # ─────────────────────────────────────────────────────────────────────────
    # NODE: refine_node
    # ─────────────────────────────────────────────────────────────────────────
    def refine_node(self, state: CourseState) -> CourseState:
        """Refine syllabus based on quality feedback."""
        self._log(state, "\n┌─ NODE: refine_node")
        self._log(state, "│  Improving syllabus based on feedback...")

        messages = [
            {"role": "system", "content": """Improve the syllabus based on the feedback.
Maintain the same JSON structure. Fix the issues mentioned."""},
            {"role": "user", "content": f"""Current syllabus:
{state['syllabus_json'][:2500]}

Feedback: {state['quality_feedback']}
Issues: {', '.join(state['quality_issues'])}

Return the improved syllabus as JSON."""}
        ]

        result = self.balanced_llm.invoke(messages, max_tokens=2000)
        self.api_calls += 1
        self._add_cost(state, "quality_loop", result["usage"])

        state["syllabus_json"] = result.get("content", state["syllabus_json"])

        self._log(state, f"│  → Syllabus refined")
        self._log(state, f"└─ Cost: ${state['costs']['total_cost']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # NODE: approval_node (interrupt() Pattern)
    # ─────────────────────────────────────────────────────────────────────────
    def approval_node(self, state: CourseState) -> CourseState:
        """
        Human approval checkpoint using interrupt() pattern.

        KEY LANGGRAPH DIFFERENTIATOR: interrupt() pauses execution for human input.
        Combined with checkpointer, enables pause/resume workflows.
        """
        self._log(state, "\n┌─ NODE: approval_node (interrupt)")
        self._log(state, "│  LANGGRAPH PATTERN: Human-in-the-loop via interrupt()")

        syllabus_data = self._parse_json(state["syllabus_json"])

        self._log(state, f"│  Syllabus ready for approval:")
        self._log(state, f"│    - Title: {syllabus_data.get('course_title', state['topic'])}")
        self._log(state, f"│    - Lessons: {len(syllabus_data.get('lessons', []))}")
        self._log(state, f"│    - Quality: {state['quality_score']:.2f}")

        # In production LangGraph, this would actually pause:
        # if not state["approved"]:
        #     interrupt("Please review and approve the syllabus")

        # For demo, we auto-approve but show the pattern
        self._log(state, f"│  [interrupt() would pause here for human review]")
        self._log(state, f"│  [AUTO-APPROVED for demo]")

        state["awaiting_approval"] = False
        state["approved"] = True

        # Save checkpoint (LangGraph pattern)
        self.checkpointer.save("course_generation", state)
        self._log(state, f"│  → Checkpoint saved (MemorySaver)")
        self._log(state, f"└─ Cost: ${state['costs']['total_cost']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # NODE: research_lesson_node
    # ─────────────────────────────────────────────────────────────────────────
    def research_lesson_node(self, state: CourseState) -> CourseState:
        """Research specific lesson topic."""
        syllabus_data = self._parse_json(state["syllabus_json"])
        lessons = syllabus_data.get("lessons", [])

        if state["current_lesson"] < len(lessons):
            lesson_info = lessons[state["current_lesson"]]
            title = lesson_info.get("title", f"Lesson {state['current_lesson'] + 1}")

            self._log(state, f"\n┌─ NODE: research_lesson_node (Lesson {state['current_lesson'] + 1})")
            self._log(state, f"│  Researching: {title}")

            search_result = jina_search(f"{state['topic']} {title} tutorial guide")
            self.jina_calls += 1

            if search_result["success"]:
                state["citations"].append(f"Lesson {state['current_lesson'] + 1}: {title}")

            self._log(state, f"└─ Research complete")

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # NODE: generate_lesson_node
    # ─────────────────────────────────────────────────────────────────────────
    def generate_lesson_node(self, state: CourseState) -> CourseState:
        """Generate detailed lesson plan."""
        syllabus_data = self._parse_json(state["syllabus_json"])
        lessons = syllabus_data.get("lessons", [])

        if state["current_lesson"] < len(lessons):
            lesson_info = lessons[state["current_lesson"]]
            lesson_num = state["current_lesson"] + 1

            self._log(state, f"┌─ NODE: generate_lesson_node (Lesson {lesson_num})")

            messages = [
                {"role": "system", "content": """Create a detailed lesson plan. Return JSON:
{
  "lesson_number": 1,
  "title": "...",
  "objectives": ["..."],
  "content_outline": ["..."],
  "activities": ["..."],
  "resources": ["..."],
  "citations": ["..."]
}"""},
                {"role": "user", "content": f"Lesson: {json.dumps(lesson_info)}\nCourse: {state['topic']}"}
            ]

            result = self.balanced_llm.invoke(messages, max_tokens=1500)
            self.api_calls += 1
            self._add_cost(state, "lessons", result["usage"])

            lesson_data = self._parse_json(result.get("content", "{}"))
            if not lesson_data:
                lesson_data = {
                    "lesson_number": lesson_num,
                    "title": lesson_info.get("title", f"Lesson {lesson_num}"),
                    "objectives": lesson_info.get("objectives", []),
                    "content_outline": lesson_info.get("topics", []),
                    "activities": [],
                    "resources": [],
                    "citations": []
                }
            lesson_data["lesson_number"] = lesson_num
            state["lessons"].append(lesson_data)

            self._log(state, f"│  → Lesson {lesson_num} complete")
            self._log(state, f"└─ Cost: ${state['costs']['total_cost']:.4f}")

            state["current_lesson"] += 1

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITIONAL EDGE: should_continue_lessons
    # ─────────────────────────────────────────────────────────────────────────
    def should_continue_lessons(self, state: CourseState) -> Literal["continue", "end"]:
        """Conditional edge for lesson loop."""
        syllabus_data = self._parse_json(state["syllabus_json"])
        total_lessons = len(syllabus_data.get("lessons", []))

        if state["current_lesson"] < total_lessons and state["current_lesson"] < 10:
            return "continue"
        return "end"

    # ─────────────────────────────────────────────────────────────────────────
    # NODE: gap_assessment_node
    # ─────────────────────────────────────────────────────────────────────────
    def gap_assessment_node(self, state: CourseState) -> CourseState:
        """Assess course gaps from student perspective."""
        self._log(state, "\n┌─ NODE: gap_assessment_node")
        self._log(state, "│  Student simulation analyzing course...")

        # Prepare course summary
        course_summary = f"""
COURSE: {self._parse_json(state['syllabus_json']).get('course_title', state['topic'])}

LESSONS:
"""
        for l in state["lessons"]:
            course_summary += f"\n{l.get('lesson_number', 0)}. {l.get('title', '')}"
            course_summary += f"\n   Objectives: {', '.join(l.get('objectives', [])[:2])}"

        messages = [
            {"role": "system", "content": """You are a beginner student reviewing this course.

Identify gaps, missing prerequisites, unclear concepts, and provide recommendations.

Return ONLY valid JSON:
{
  "gaps_found": ["...", "..."],
  "missing_prerequisites": ["..."],
  "unclear_concepts": ["..."],
  "recommendations": ["..."],
  "ready_for_publication": true
}"""},
            {"role": "user", "content": course_summary}
        ]

        result = self.cheap_llm.invoke(messages, max_tokens=800)
        self.api_calls += 1
        self._add_cost(state, "gap_assessment", result["usage"])

        state["gap_assessment"] = self._parse_json(result.get("content", "{}"))

        # Build gap context for potential time-travel refinement
        gaps = state["gap_assessment"]
        gap_context_parts = []
        if gaps.get("gaps_found"):
            gap_context_parts.append(f"GAPS: {', '.join(gaps['gaps_found'][:3])}")
        if gaps.get("missing_prerequisites"):
            gap_context_parts.append(f"MISSING PREREQS: {', '.join(gaps['missing_prerequisites'][:3])}")
        if gaps.get("unclear_concepts"):
            gap_context_parts.append(f"UNCLEAR: {', '.join(gaps['unclear_concepts'][:3])}")
        if gaps.get("recommendations"):
            gap_context_parts.append(f"RECOMMENDATIONS: {', '.join(gaps['recommendations'][:3])}")

        state["gap_context"] = "\n".join(gap_context_parts)

        self._log(state, f"│  → Gaps found: {len(state['gap_assessment'].get('gaps_found', []))}")
        self._log(state, f"│  → Ready: {state['gap_assessment'].get('ready_for_publication', False)}")
        self._log(state, f"└─ Cost: ${state['costs']['total_cost']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITIONAL EDGE: should_refine_after_gaps
    # ─────────────────────────────────────────────────────────────────────────
    def should_refine_after_gaps(self, state: CourseState) -> Literal["refine", "complete"]:
        """
        Conditional edge after gap assessment - determines if time-travel refinement needed.

        KEY LANGGRAPH DIFFERENTIATOR: This enables gap-driven refinement using
        the checkpointer's time-travel capability. If significant gaps exist,
        we can "rewind" to a prior checkpoint and replay with gap context.

        In production LangGraph:
            graph.add_conditional_edges(
                "gap_assessment",
                should_refine_after_gaps,
                {"refine": "gap_driven_refinement", "complete": END}
            )
        """
        gaps = state["gap_assessment"]
        total_issues = (
            len(gaps.get("gaps_found", [])) +
            len(gaps.get("missing_prerequisites", [])) +
            len(gaps.get("unclear_concepts", []))
        )

        # Only refine once to avoid infinite loops
        if state["refinement_iteration"] >= 1:
            return "complete"

        # Trigger refinement if significant gaps found and not ready
        if not gaps.get("ready_for_publication", True) and total_issues >= 2:
            return "refine"

        return "complete"

    # ─────────────────────────────────────────────────────────────────────────
    # NODE: gap_driven_refinement_node (TIME-TRAVEL PATTERN)
    # ─────────────────────────────────────────────────────────────────────────
    def gap_driven_refinement_node(self, state: CourseState) -> CourseState:
        """
        Refine lessons based on gap assessment using TIME-TRAVEL pattern.

        KEY LANGGRAPH DIFFERENTIATOR: This demonstrates checkpointer "time-travel"
        where we can reload a previous state and replay the workflow with new context.

        In production LangGraph, this would:
        1. Load checkpoint from before lesson generation
        2. Inject gap_context into state
        3. Replay lesson generation nodes with gap awareness

        For our simulation, we refine existing lessons with gap context.
        """
        self._log(state, "\n" + "═" * 60)
        self._log(state, "GAP-DRIVEN REFINEMENT (TIME-TRAVEL PATTERN)")
        self._log(state, "═" * 60)
        self._log(state, "│  LANGGRAPH DIFFERENTIATOR: Checkpointer rewind + replay")
        self._log(state, f"│  Refinement iteration: {state['refinement_iteration'] + 1}")

        state["refinement_iteration"] += 1

        # Simulate time-travel: load pre-lesson checkpoint
        pre_lesson_state = self.checkpointer.load("course_generation", "pre_lessons")
        if pre_lesson_state:
            self._log(state, "│  → Loaded checkpoint 'pre_lessons' (TIME-TRAVEL)")
            self._log(state, "│  → Injecting gap context into replayed state")

        # Refine each lesson with gap context
        self._log(state, f"│  → Refining {len(state['lessons'])} lessons with gap insights...")

        refined_lessons = []
        gap_context = state["gap_context"]

        for i, lesson in enumerate(state["lessons"]):
            lesson_num = lesson.get("lesson_number", i + 1)

            messages = [
                {"role": "system", "content": f"""Improve this lesson based on student feedback.
Address these issues:
{gap_context}

Return ONLY valid JSON with the same structure but improved content:
{{
  "lesson_number": {lesson_num},
  "title": "...",
  "objectives": ["..."],
  "content_outline": ["..."],
  "activities": ["..."],
  "resources": ["..."],
  "citations": ["..."]
}}"""},
                {"role": "user", "content": f"Original lesson:\n{json.dumps(lesson, indent=2)}"}
            ]

            result = self.balanced_llm.invoke(messages, max_tokens=1500)
            self.api_calls += 1
            self._add_cost(state, "gap_refinement", result["usage"])

            refined = self._parse_json(result.get("content", "{}"))
            if refined:
                refined["lesson_number"] = lesson_num
                refined_lessons.append(refined)
            else:
                refined_lessons.append(lesson)

            if i < 2 or i == len(state["lessons"]) - 1:  # Log first 2 and last
                self._log(state, f"│    Lesson {lesson_num}: refined")

        state["refined_lessons"] = refined_lessons
        state["lessons"] = refined_lessons  # Replace with refined versions

        self._log(state, f"│  → All lessons refined with gap awareness")
        self._log(state, f"└─ Cost: ${state['costs']['total_cost']:.4f}")

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH EXECUTION
    # ─────────────────────────────────────────────────────────────────────────
    def run(self, prompt: str):
        """
        Execute the compiled graph.

        In production LangGraph:
            app = graph.compile(checkpointer=MemorySaver())
            result = app.invoke({"prompt": prompt}, config={"thread_id": "123"})
        """
        from common.models import (
            LessonPlan, Syllabus, CoursePackage, GenerationMetrics, FrameworkResult,
            QualityScore, GapAssessment, CostBreakdown, EnhancedCoursePackage
        )

        self.start_time = datetime.now()

        # Initialize state (TypedDict pattern)
        state: CourseState = {
            "prompt": prompt,
            "topic": "",
            "research": "",
            "parallel_research": [],
            "syllabus_json": "",
            "current_lesson": 0,
            "lessons": [],
            "citations": [],
            "console_log": [],
            "quality_score": 0.0,
            "quality_feedback": "",
            "quality_issues": [],
            "quality_iteration": 0,
            "awaiting_approval": False,
            "approved": False,
            "gap_assessment": {},
            # Gap-driven refinement state (TIME-TRAVEL PATTERN)
            "refinement_iteration": 0,
            "gap_context": "",
            "refined_lessons": [],
            "costs": {
                "research_cost": 0.0,
                "syllabus_cost": 0.0,
                "quality_loop_cost": 0.0,
                "lesson_generation_cost": 0.0,
                "gap_assessment_cost": 0.0,
                "gap_refinement_cost": 0.0,  # NEW: Track refinement costs
                "total_cost": 0.0,
                "total_tokens": 0
            },
            "metrics": {}
        }

        try:
            self._log(state, "═" * 60)
            self._log(state, "LANGGRAPH ENHANCED WORKFLOW")
            self._log(state, "Demonstrating: Send API, interrupt(), Conditional Edges")
            self._log(state, "═" * 60)

            # Execute graph: START -> understand
            state = self.understand_node(state)

            # Parallel research (Send API pattern)
            state = self.parallel_research_node(state)

            # Syllabus creation
            state = self.syllabus_node(state)

            # Quality loop with conditional edges
            self._log(state, "\n" + "─" * 40)
            self._log(state, "ENTERING QUALITY LOOP (Conditional Edges)")
            self._log(state, "─" * 40)

            while True:
                state = self.quality_check_node(state)
                routing = self.should_refine(state)

                self._log(state, f"│  Conditional edge → {routing}")

                if routing == "approve":
                    state = self.approval_node(state)
                    break
                else:
                    state = self.refine_node(state)

            # Lesson generation loop
            self._log(state, "\n" + "─" * 40)
            self._log(state, "ENTERING LESSON LOOP (Conditional Edge)")
            self._log(state, "─" * 40)

            # Save checkpoint BEFORE lessons for time-travel (LANGGRAPH PATTERN)
            self.checkpointer.save("course_generation", state, "pre_lessons")
            self._log(state, "│  [Checkpoint saved: 'pre_lessons' for potential time-travel]")

            while self.should_continue_lessons(state) == "continue":
                state = self.research_lesson_node(state)
                state = self.generate_lesson_node(state)

            self._log(state, f"Exited lesson loop → END")

            # Gap assessment
            state = self.gap_assessment_node(state)

            # Gap-driven refinement conditional edge (TIME-TRAVEL PATTERN)
            self._log(state, "\n" + "─" * 40)
            self._log(state, "GAP-DRIVEN REFINEMENT CONDITIONAL EDGE")
            self._log(state, "─" * 40)

            refinement_decision = self.should_refine_after_gaps(state)
            self._log(state, f"│  Conditional edge → {refinement_decision}")

            if refinement_decision == "refine":
                self._log(state, "│  Triggering TIME-TRAVEL refinement...")
                state = self.gap_driven_refinement_node(state)

                # Re-run gap assessment after refinement
                self._log(state, "│  Re-assessing gaps after refinement...")
                state = self.gap_assessment_node(state)
            else:
                self._log(state, "│  Course ready - no refinement needed")

            # Build output
            self._log(state, "\n┌─ FINAL: Compiling Course Package")

            syllabus_data = self._parse_json(state["syllabus_json"])

            lesson_plans = []
            for l in state["lessons"]:
                lesson_plans.append(LessonPlan(
                    lesson_number=l.get("lesson_number", 0),
                    title=l.get("title", ""),
                    objectives=l.get("objectives", []),
                    content_outline=l.get("content_outline", []),
                    activities=l.get("activities", []),
                    resources=l.get("resources", []),
                    citations=l.get("citations", [])
                ))

            syllabus = Syllabus(
                course_title=syllabus_data.get("course_title", state["topic"]),
                course_objective=syllabus_data.get("course_objective", ""),
                target_audience=syllabus_data.get("target_audience", "General learners"),
                difficulty_level=syllabus_data.get("difficulty_level", "Intermediate"),
                lessons=lesson_plans
            )

            # Build quality score
            final_quality_score = QualityScore(
                score=state["quality_score"],
                feedback=state["quality_feedback"],
                issues=state["quality_issues"],
                iteration=state["quality_iteration"]
            )

            # Build gap assessment
            gap_data = state["gap_assessment"]
            final_gap_assessment = GapAssessment(
                gaps_found=gap_data.get("gaps_found", []),
                missing_prerequisites=gap_data.get("missing_prerequisites", []),
                unclear_concepts=gap_data.get("unclear_concepts", []),
                recommendations=gap_data.get("recommendations", []),
                ready_for_publication=gap_data.get("ready_for_publication", False)
            )

            # Build cost breakdown
            cost_breakdown = CostBreakdown(
                research_cost=state["costs"]["research_cost"],
                syllabus_cost=state["costs"]["syllabus_cost"],
                quality_loop_cost=state["costs"]["quality_loop_cost"],
                lesson_generation_cost=state["costs"]["lesson_generation_cost"],
                gap_assessment_cost=state["costs"]["gap_assessment_cost"],
                total_cost=state["costs"]["total_cost"],
                total_tokens=state["costs"]["total_tokens"]
            )

            # Legacy course package
            course = CoursePackage(
                syllabus=syllabus,
                research_sources=state["citations"],
                generation_metadata={
                    "framework": "LangGraph",
                    "patterns_demonstrated": [
                        "Send API (parallel fan-out)",
                        "interrupt() (human-in-the-loop)",
                        "Conditional edges (quality loop)",
                        "Checkpointer (state persistence)",
                        "TIME-TRAVEL (gap-driven refinement)",  # NEW
                        "TypedDict state"
                    ],
                    "total_cost": cost_breakdown.total_cost,
                    "total_tokens": cost_breakdown.total_tokens,
                    "api_calls": self.api_calls,
                    "jina_calls": self.jina_calls,
                    "quality_iterations": state["quality_iteration"],
                    "refinement_iterations": state["refinement_iteration"]  # NEW
                }
            )

            # Enhanced course package
            enhanced_course = EnhancedCoursePackage(
                syllabus=syllabus,
                quality_score=final_quality_score,
                gap_assessment=final_gap_assessment,
                cost_breakdown=cost_breakdown,
                research_sources=state["citations"],
                generation_metadata={
                    "framework": "LangGraph",
                    "patterns_demonstrated": [
                        "Send API (parallel fan-out)",
                        "interrupt() (human-in-the-loop)",
                        "Conditional edges (quality loop)",
                        "Checkpointer (state persistence)",
                        "TIME-TRAVEL (gap-driven refinement)",  # NEW
                        "TypedDict state"
                    ],
                    "models_used": {
                        "cheap": CHEAP_MODEL,
                        "balanced": BALANCED_MODEL
                    },
                    "quality_iterations": state["quality_iteration"],
                    "refinement_iterations": state["refinement_iteration"]  # NEW
                }
            )

            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()

            self._log(state, f"│")
            self._log(state, f"│  ╔════════════════════════════════════════════╗")
            self._log(state, f"│  ║  LANGGRAPH WORKFLOW COMPLETE               ║")
            self._log(state, f"│  ╠════════════════════════════════════════════╣")
            self._log(state, f"│  ║  Lessons:     {len(lesson_plans):<26} ║")
            self._log(state, f"│  ║  Duration:    {duration:.1f}s{' ':<23} ║")
            self._log(state, f"│  ║  Total Cost:  ${cost_breakdown.total_cost:.4f}{' ':<20} ║")
            self._log(state, f"│  ║  Quality:     {final_quality_score.score:<26} ║")
            self._log(state, f"│  ╚════════════════════════════════════════════╝")
            self._log(state, f"└─")

            metrics = GenerationMetrics(
                framework="LangGraph",
                total_tokens=state["costs"]["total_tokens"],
                api_calls=self.api_calls,
                jina_calls=self.jina_calls
            )
            metrics.start_time = self.start_time
            metrics.end_time = self.end_time

            return FrameworkResult(
                framework="LangGraph",
                success=True,
                course=course,
                enhanced_course=enhanced_course,
                metrics=metrics,
                console_output=state["console_log"]
            )

        except Exception as e:
            self.end_time = datetime.now()
            self.errors.append(str(e))

            import traceback
            self._log(state, f"ERROR: {e}")
            self._log(state, traceback.format_exc())

            metrics = GenerationMetrics(framework="LangGraph")
            metrics.start_time = self.start_time
            metrics.end_time = self.end_time
            metrics.errors = self.errors

            return FrameworkResult(
                framework="LangGraph",
                success=False,
                error=str(e),
                metrics=metrics,
                console_output=state.get("console_log", [])
            )


def generate_course(prompt: str, callback: Callable[[str], None] = None):
    """
    Generate a course using LangGraph-style workflow.

    Entry point that creates the graph and executes it.
    """
    generator = LangGraphCourseGenerator(callback)
    return generator.run(prompt)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add project root to path (relative to this file)
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(PROJECT_ROOT))
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    result = generate_course("Create a course on Python for Beginners")
    print(f"\nSuccess: {result.success}")
    if result.course:
        print(f"Lessons: {len(result.course.syllabus.lessons)}")
    if result.enhanced_course:
        print(f"Quality Score: {result.enhanced_course.quality_score.score if result.enhanced_course.quality_score else 'N/A'}")
        print(f"Total Cost: ${result.enhanced_course.cost_breakdown.total_cost:.4f}" if result.enhanced_course.cost_breakdown else "")
    if result.metrics:
        print(f"Duration: {result.metrics.duration_seconds:.1f}s")
