"""
Google ADK Course Generator - ENHANCED IMPLEMENTATION

Demonstrates Google Agent Development Kit's unique patterns:

FRAMEWORK DIFFERENTIATORS:
1. ParallelAgent - Concurrent execution of multiple agents (parallel research)
2. LoopAgent - Iterative refinement with max_iterations and escalation
3. SequentialAgent - Ordered workflow composition
4. output_key + SessionState - Inter-agent data flow pattern
5. AgentTool - Agent-as-tool pattern for gap assessment
6. {{template}} variable substitution in instructions

ENHANCED FEATURES:
- Parallel research using ParallelAgent (ADK's native parallelism)
- Quality loop using LoopAgent with escalation signal
- Human approval checkpoint via callback hook
- Gap assessment using AgentTool (agent wrapped as tool)
- Cost tracking via SessionState aggregation
"""
import os
import sys
import json
import httpx
from typing import List, Callable, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ============================================================================
# Cost Tracking via SessionState (ADK Pattern)
# ============================================================================
# Google ADK uses SessionState for all inter-agent data sharing.
# We extend this pattern to track costs across the entire workflow.

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
# Google ADK-style LLM Setup (using OpenRouter for compatibility)
# ============================================================================

from openai import OpenAI


class GeminiModel:
    """
    Simulates Google ADK's Gemini model interface.

    In production ADK, you'd use google.genai directly.
    Here we configure OpenRouter for model-agnostic compatibility.

    KEY ADK PATTERN: Model selection per agent for cost optimization.
    """

    def __init__(self, model: str = None):
        self.model = model or BALANCED_MODEL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", "")
        )
        self._last_usage = {}

    def generate_content(
        self,
        prompt: str,
        system_instruction: str = None,
        max_tokens: int = 2000
    ) -> dict:
        """
        Generate content using the model.

        Returns structured response with usage tracking for cost aggregation.
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens
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
            "text": response.choices[0].message.content,
            "usage": self._last_usage
        }


# ============================================================================
# Google ADK-style Tool Definitions (Jina Integration)
# ============================================================================

class FunctionTool:
    """
    Simulates Google ADK FunctionTool.

    In production ADK, tools are defined as Python functions and
    automatically converted to function declarations for the LLM.
    """

    def __init__(self, func: Callable, description: str = None):
        self.func = func
        self.name = func.__name__
        self.description = description or func.__doc__ or ""

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def create_function_tool(description: str = None):
    """Decorator to create a FunctionTool."""
    def decorator(func):
        return FunctionTool(func, description)
    return decorator


@create_function_tool("Search the web for information on a topic")
def jina_search(query: str) -> str:
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
            return response.text[:10000]
        return f"Search failed: Status {response.status_code}"
    except Exception as e:
        return f"Search error: {str(e)}"


@create_function_tool("Read URL content as markdown")
def jina_read(url: str) -> str:
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
            return response.text[:15000]
        return f"Read failed: Status {response.status_code}"
    except Exception as e:
        return f"Read error: {str(e)}"


# ============================================================================
# Google ADK SessionState with Cost Tracking
# ============================================================================

class SessionState:
    """
    Enhanced SessionState for Google ADK with built-in cost tracking.

    KEY ADK PATTERN: SessionState is the central data sharing mechanism.
    All agents read/write to session state via output_key.

    ENHANCEMENT: We add cost tracking per phase, aggregated automatically.
    """

    def __init__(self):
        self._state: Dict[str, Any] = {}
        # Cost tracking extensions
        self.costs_by_phase: Dict[str, float] = {}
        self.tokens_by_phase: Dict[str, int] = {}
        self.total_cost: float = 0.0
        self.total_tokens: int = 0

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state[key] = value

    def add_cost(self, phase: str, cost: float, tokens: int):
        """Track cost by generation phase (ADK cost aggregation pattern)."""
        self.costs_by_phase[phase] = self.costs_by_phase.get(phase, 0) + cost
        self.tokens_by_phase[phase] = self.tokens_by_phase.get(phase, 0) + tokens
        self.total_cost += cost
        self.total_tokens += tokens

    def get_cost_breakdown(self) -> dict:
        """Return cost breakdown for EnhancedCoursePackage."""
        return {
            "research_cost": self.costs_by_phase.get("research", 0),
            "syllabus_cost": self.costs_by_phase.get("syllabus", 0),
            "quality_loop_cost": self.costs_by_phase.get("quality_loop", 0),
            "lesson_generation_cost": self.costs_by_phase.get("lessons", 0),
            "gap_assessment_cost": self.costs_by_phase.get("gap_assessment", 0),
            "gap_refinement_cost": self.costs_by_phase.get("gap_refinement", 0),  # NEW
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens
        }


# ============================================================================
# Google ADK Agent Classes - ENHANCED
# ============================================================================

class LlmAgent:
    """
    Enhanced LlmAgent with cost tracking and model selection.

    KEY ADK PATTERNS:
    - instruction with {{template}} variable substitution
    - output_key for storing results in SessionState
    - model selection per agent (cheap vs balanced)
    - cost tracking aggregation to session
    """

    def __init__(
        self,
        name: str,
        instruction: str,
        tools: List[FunctionTool] = None,
        output_key: str = None,
        model: GeminiModel = None,
        cost_phase: str = "general"
    ):
        self.name = name
        self.instruction = instruction
        self.tools = tools or []
        self.output_key = output_key
        self.model = model or GeminiModel()
        self.cost_phase = cost_phase
        self._last_usage = {}

    def run(self, session: SessionState, context: str = "") -> str:
        """
        Execute the agent with template substitution and cost tracking.
        """
        # Template variable substitution ({{variable}} pattern)
        prompt = self.instruction
        for key, value in session._state.items():
            if isinstance(value, str):
                prompt = prompt.replace(f"{{{{{key}}}}}", value[:2000])

        # Generate response
        result = self.model.generate_content(
            prompt=context,
            system_instruction=prompt
        )
        self._last_usage = result.get("usage", {})
        output = result.get("text", "")

        # Track cost in session (ADK pattern)
        session.add_cost(
            self.cost_phase,
            self._last_usage.get("cost", 0),
            self._last_usage.get("total_tokens", 0)
        )

        # Store in session state (output_key pattern)
        if self.output_key:
            session.set(self.output_key, output)

        return output


class SequentialAgent:
    """
    SequentialAgent - Runs sub-agents in order.

    KEY ADK PATTERN: Hierarchical composition for workflow orchestration.
    """

    def __init__(self, name: str, sub_agents: List[Any]):
        self.name = name
        self.sub_agents = sub_agents

    def run(self, session: SessionState, context: str = "") -> str:
        """Execute all sub-agents sequentially."""
        result = ""
        for agent in self.sub_agents:
            result = agent.run(session, context)
        return result


class ParallelAgent:
    """
    ParallelAgent - Runs sub-agents concurrently.

    KEY ADK DIFFERENTIATOR: Native parallel execution without asyncio.
    In production ADK, ParallelAgent handles concurrent API calls automatically.

    This is a unique ADK pattern not found in other frameworks.
    """

    def __init__(self, name: str, sub_agents: List[Any]):
        self.name = name
        self.sub_agents = sub_agents

    def run(self, session: SessionState, context: str = "") -> List[str]:
        """
        Execute all sub-agents in parallel.

        In production ADK, this uses internal threading/async.
        We simulate parallel execution here.
        """
        import concurrent.futures

        results = []

        # ADK pattern: parallel execution with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.sub_agents)) as executor:
            futures = {
                executor.submit(agent.run, session, context): agent
                for agent in self.sub_agents
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(f"Error: {str(e)}")

        return results


class LoopAgent:
    """
    LoopAgent - Iterative refinement with escalation signal.

    KEY ADK DIFFERENTIATOR: Built-in loop primitive with:
    - max_iterations limit
    - escalation signal for early exit
    - iteration tracking in session state

    Perfect for quality refinement loops.
    """

    def __init__(
        self,
        name: str,
        sub_agents: List[Any],
        max_iterations: int = 10,
        escalation_key: str = "escalate"
    ):
        self.name = name
        self.sub_agents = sub_agents
        self.max_iterations = max_iterations
        self.escalation_key = escalation_key

    def run(self, session: SessionState, context: str = "") -> str:
        """
        Execute sub-agents in a loop until escalation or max iterations.

        ADK PATTERN: Loop with escalation signal
        - Set session[escalation_key] = True to exit loop early
        - Quality score > threshold triggers escalation
        """
        result = ""
        for i in range(self.max_iterations):
            session.set("loop_iteration", i + 1)

            for agent in self.sub_agents:
                result = agent.run(session, context)

            # Check for escalation signal (ADK pattern)
            if session.get(self.escalation_key, False):
                break

        return result


class AgentTool:
    """
    AgentTool - Wrap an agent as a callable tool.

    KEY ADK DIFFERENTIATOR: Agent-as-tool pattern.
    Allows complex agent logic to be used as a tool by other agents.

    Perfect for gap assessment where a specialized agent evaluates content.
    """

    def __init__(self, agent: LlmAgent, name: str = None, description: str = None):
        self.agent = agent
        self.name = name or f"{agent.name}_tool"
        self.description = description or f"Execute {agent.name} agent"

    def __call__(self, session: SessionState, context: str = "") -> str:
        """Execute the wrapped agent as a tool."""
        return self.agent.run(session, context)


# ============================================================================
# Enhanced Course Generator using Google ADK patterns
# ============================================================================

class GoogleADKCourseGenerator:
    """
    Enhanced Course Generator demonstrating Google ADK's unique patterns.

    WORKFLOW ARCHITECTURE:

        SequentialAgent (root)
          │
          ├── LlmAgent: TopicExtractor
          │     └── output_key="topic"
          │
          ├── ParallelAgent: ResearchTeam  ◄── ADK DIFFERENTIATOR
          │     ├── LlmAgent: TutorialResearcher
          │     ├── LlmAgent: PracticesResearcher
          │     └── LlmAgent: ExamplesResearcher
          │     All write to session["parallel_research"]
          │
          ├── LlmAgent: SyllabusCreator
          │     └── output_key="syllabus_json"
          │
          ├── LoopAgent: QualityRefinementLoop  ◄── ADK DIFFERENTIATOR
          │     ├── LlmAgent: QualityEvaluator (cheap model)
          │     │     └── output_key="quality_score"
          │     └── LlmAgent: SyllabusRefiner
          │           └── escalation when score >= 0.8
          │
          ├── [Human Approval Checkpoint]  ◄── Callback hook
          │
          ├── LoopAgent: LessonGenerationLoop
          │     ├── FunctionTool: jina_search
          │     └── LlmAgent: LessonWriter
          │
          └── AgentTool: GapAssessor  ◄── ADK DIFFERENTIATOR
                └── Student simulation agent wrapped as tool

    COST TRACKING:
    - SessionState aggregates costs by phase
    - Different models for different agents (Haiku vs Sonnet)
    - Full breakdown in EnhancedCoursePackage
    """

    def __init__(self, callback: Callable[[str], None] = None):
        self.callback = callback or print
        self.console_log = []
        self.session = SessionState()

        # Metrics
        self.api_calls = 0
        self.jina_calls = 0
        self.start_time = None
        self.end_time = None
        self.errors = []

        # Model setup - different models for different cost profiles
        self.cheap_model = GeminiModel(model=CHEAP_MODEL)
        self.balanced_model = GeminiModel(model=BALANCED_MODEL)

        # Tools
        self.tools = [jina_search, jina_read]

    def _log(self, msg: str):
        """Log to console (visible in Streamlit UI)."""
        full_msg = f"[Google ADK] {msg}"
        self.console_log.append(full_msg)
        print(full_msg)
        if self.callback:
            self.callback(full_msg)

    def _parse_json(self, content: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            return {}

    def run(self, prompt: str):
        """
        Execute the hierarchical agent workflow with all ADK patterns.
        """
        from common.models import (
            LessonPlan, Syllabus, CoursePackage, GenerationMetrics, FrameworkResult,
            QualityScore, GapAssessment, CostBreakdown, EnhancedCoursePackage
        )

        self.start_time = datetime.now()
        self.session = SessionState()
        self.session.set("prompt", prompt)
        citations = []

        try:
            self._log("═" * 60)
            self._log("GOOGLE ADK ENHANCED WORKFLOW")
            self._log("Demonstrating: ParallelAgent, LoopAgent, AgentTool patterns")
            self._log("═" * 60)

            # ═══════════════════════════════════════════════════════════════════
            # PHASE 1: Topic Extraction
            # ═══════════════════════════════════════════════════════════════════
            self._log("\n┌─ PHASE 1: Topic Extraction")
            self._log("│  LlmAgent: TopicExtractor (Haiku - cheap)")

            topic_agent = LlmAgent(
                name="TopicExtractor",
                instruction="Extract the main course topic. Return only the topic name, nothing else.",
                output_key="topic",
                model=self.cheap_model,
                cost_phase="research"
            )
            topic_agent.run(self.session, prompt)
            self.api_calls += 1

            topic = self.session.get("topic", prompt).strip()
            self._log(f"│  → output_key='topic': {topic}")
            self._log(f"└─ Cost so far: ${self.session.total_cost:.4f}")

            # ═══════════════════════════════════════════════════════════════════
            # PHASE 2: Parallel Research (ADK DIFFERENTIATOR)
            # ═══════════════════════════════════════════════════════════════════
            self._log("\n┌─ PHASE 2: Parallel Research (ParallelAgent)")
            self._log("│  ADK PATTERN: Native parallel agent execution")

            # Create parallel research agents
            tutorial_researcher = LlmAgent(
                name="TutorialResearcher",
                instruction=f"Research tutorials and learning resources for {topic}. Focus on beginner-friendly content.",
                output_key="tutorial_research",
                model=self.cheap_model,
                cost_phase="research"
            )

            practices_researcher = LlmAgent(
                name="PracticesResearcher",
                instruction=f"Research best practices and industry standards for {topic}.",
                output_key="practices_research",
                model=self.cheap_model,
                cost_phase="research"
            )

            examples_researcher = LlmAgent(
                name="ExamplesResearcher",
                instruction=f"Research practical examples and hands-on projects for {topic}.",
                output_key="examples_research",
                model=self.cheap_model,
                cost_phase="research"
            )

            # Use Jina for initial research
            self._log("│  FunctionTool: jina_search (parallel queries)")
            research1 = jina_search(f"comprehensive tutorial {topic}")
            research2 = jina_search(f"best practices {topic}")
            research3 = jina_search(f"hands-on projects {topic}")
            self.jina_calls += 3
            citations.extend([
                f"Tutorial research: {topic}",
                f"Best practices: {topic}",
                f"Hands-on projects: {topic}"
            ])

            # Store combined research
            combined_research = f"""
TUTORIAL RESOURCES:
{research1[:3000]}

BEST PRACTICES:
{research2[:3000]}

PRACTICAL EXAMPLES:
{research3[:3000]}
"""
            self.session.set("research", combined_research)

            # Execute parallel agents for synthesis
            parallel_team = ParallelAgent(
                name="ResearchTeam",
                sub_agents=[tutorial_researcher, practices_researcher, examples_researcher]
            )

            self._log("│  → Executing 3 research agents in parallel...")
            parallel_team.run(self.session, f"Based on this research:\n{combined_research[:2000]}\n\nSummarize key insights.")
            self.api_calls += 3

            self._log(f"│  → Parallel research complete (3 agents)")
            self._log(f"└─ Cost so far: ${self.session.total_cost:.4f}")

            # ═══════════════════════════════════════════════════════════════════
            # PHASE 3: Syllabus Creation
            # ═══════════════════════════════════════════════════════════════════
            self._log("\n┌─ PHASE 3: Syllabus Creation")
            self._log("│  LlmAgent: SyllabusCreator (Sonnet - balanced)")

            syllabus_agent = LlmAgent(
                name="SyllabusCreator",
                instruction="""Create a comprehensive 10-lesson syllabus for {{topic}}.

Use this research: {{research}}

Tutorial insights: {{tutorial_research}}
Best practices: {{practices_research}}
Examples: {{examples_research}}

Return ONLY valid JSON:
{
  "course_title": "...",
  "course_objective": "...",
  "target_audience": "...",
  "difficulty_level": "...",
  "lessons": [
    {"lesson_number": 1, "title": "...", "objectives": ["..."], "topics": ["..."]}
  ]
}""",
                output_key="syllabus_json",
                model=self.balanced_model,
                cost_phase="syllabus"
            )
            syllabus_agent.run(self.session, f"Create syllabus for: {topic}")
            self.api_calls += 1

            syllabus_content = self.session.get("syllabus_json", "{}")
            syllabus_data = self._parse_json(syllabus_content)
            if not syllabus_data:
                syllabus_data = {"course_title": topic, "course_objective": "", "lessons": []}

            self._log(f"│  → output_key='syllabus_json': {len(syllabus_data.get('lessons', []))} lessons")
            self._log(f"└─ Cost so far: ${self.session.total_cost:.4f}")

            # ═══════════════════════════════════════════════════════════════════
            # PHASE 4: Quality Refinement Loop (ADK DIFFERENTIATOR)
            # ═══════════════════════════════════════════════════════════════════
            self._log("\n┌─ PHASE 4: Quality Refinement (LoopAgent)")
            self._log("│  ADK PATTERN: LoopAgent with escalation signal")
            self._log("│  max_iterations=3, escalation when quality >= 0.8")

            quality_evaluator = LlmAgent(
                name="QualityEvaluator",
                instruction="""Evaluate the syllabus quality on a scale of 0.0 to 1.0.

Consider:
- Learning progression logic
- Objective clarity
- Topic coverage
- Practical applicability

Return ONLY valid JSON:
{
  "score": 0.85,
  "feedback": "...",
  "issues": ["...", "..."]
}""",
                output_key="quality_evaluation",
                model=self.cheap_model,
                cost_phase="quality_loop"
            )

            syllabus_refiner = LlmAgent(
                name="SyllabusRefiner",
                instruction="""Refine the syllabus based on quality feedback.

Current syllabus: {{syllabus_json}}
Quality feedback: {{quality_evaluation}}

Return the improved syllabus as JSON with the same structure.""",
                output_key="syllabus_json",
                model=self.balanced_model,
                cost_phase="quality_loop"
            )

            # Quality loop with manual control (demonstrating LoopAgent pattern)
            final_quality_score = None
            quality_iteration = 0
            max_quality_iterations = 3
            quality_threshold = 0.8

            for iteration in range(max_quality_iterations):
                quality_iteration = iteration + 1
                self._log(f"│  LoopIteration {quality_iteration}:")

                # Evaluate quality
                quality_evaluator.run(self.session, f"Evaluate this syllabus:\n{self.session.get('syllabus_json', '')}")
                self.api_calls += 1

                quality_json = self._parse_json(self.session.get("quality_evaluation", "{}"))
                score = quality_json.get("score", 0.7)
                feedback = quality_json.get("feedback", "")
                issues = quality_json.get("issues", [])

                self._log(f"│    → Quality score: {score:.2f}")
                self._log(f"│    → Feedback: {feedback[:50]}...")

                final_quality_score = QualityScore(
                    score=score,
                    feedback=feedback,
                    issues=issues,
                    iteration=quality_iteration
                )

                # Check escalation condition (ADK pattern)
                if score >= quality_threshold:
                    self._log(f"│    → ESCALATION: Quality threshold met!")
                    self.session.set("escalate", True)
                    break

                # Refine if not at threshold and not last iteration
                if iteration < max_quality_iterations - 1:
                    self._log(f"│    → Refining syllabus...")
                    syllabus_refiner.run(self.session, "Improve the syllabus")
                    self.api_calls += 1

                    # Update syllabus data
                    syllabus_data = self._parse_json(self.session.get("syllabus_json", "{}"))

            self._log(f"│  → Quality loop complete after {quality_iteration} iterations")
            self._log(f"└─ Cost so far: ${self.session.total_cost:.4f}")

            # ═══════════════════════════════════════════════════════════════════
            # PHASE 5: Human Approval Checkpoint
            # ═══════════════════════════════════════════════════════════════════
            self._log("\n┌─ PHASE 5: Human Approval Checkpoint")
            self._log("│  ADK PATTERN: Callback hook for human-in-the-loop")

            # In production, this would pause for human review
            # For demo, we auto-approve after showing the checkpoint
            self._log(f"│  Syllabus ready for review:")
            self._log(f"│    - Title: {syllabus_data.get('course_title', topic)}")
            self._log(f"│    - Lessons: {len(syllabus_data.get('lessons', []))}")
            self._log(f"│    - Quality: {final_quality_score.score if final_quality_score else 'N/A'}")
            self._log(f"│  [AUTO-APPROVED for demo]")
            self._log(f"└─ Proceeding to lesson generation...")

            # ═══════════════════════════════════════════════════════════════════
            # PHASE 6: Lesson Generation Loop
            # ═══════════════════════════════════════════════════════════════════
            self._log("\n┌─ PHASE 6: Lesson Generation (LoopAgent)")
            self._log("│  ADK PATTERN: LoopAgent with FunctionTool + LlmAgent")

            lessons = []
            lesson_list = syllabus_data.get("lessons", [])[:10]

            for i, lesson_info in enumerate(lesson_list):
                lesson_num = i + 1
                title = lesson_info.get("title", f"Lesson {lesson_num}")
                self.session.set("loop_iteration", lesson_num)
                self.session.set("current_lesson_info", json.dumps(lesson_info))

                self._log(f"│  LoopIteration {lesson_num}: {title}")

                # FunctionTool: research
                self._log(f"│    → FunctionTool: jina_search('{title}')")
                lesson_research = jina_search(f"{topic} {title}")
                self.jina_calls += 1
                self.session.set("lesson_research", lesson_research)
                citations.append(f"Lesson {lesson_num}: {title}")

                # LlmAgent: lesson writer
                lesson_writer = LlmAgent(
                    name="LessonWriter",
                    instruction="""Write a detailed lesson plan.

Lesson info: {{current_lesson_info}}
Research: {{lesson_research}}

Return ONLY valid JSON:
{
  "lesson_number": 1,
  "title": "...",
  "objectives": ["..."],
  "content_outline": ["..."],
  "activities": ["..."],
  "resources": ["..."],
  "citations": ["..."]
}""",
                    output_key="current_lesson",
                    model=self.balanced_model,
                    cost_phase="lessons"
                )
                lesson_writer.run(self.session, f"Write lesson {lesson_num}")
                self.api_calls += 1

                lesson_content = self.session.get("current_lesson", "{}")
                lesson_data = self._parse_json(lesson_content)
                if not lesson_data:
                    lesson_data = {
                        "lesson_number": lesson_num,
                        "title": title,
                        "objectives": lesson_info.get("objectives", []),
                        "content_outline": lesson_info.get("topics", []),
                        "activities": [],
                        "resources": [],
                        "citations": []
                    }
                lesson_data["lesson_number"] = lesson_num
                lessons.append(lesson_data)

                self._log(f"│    → output_key='current_lesson': Complete")

            self._log(f"│  → Generated {len(lessons)} lessons")
            self._log(f"└─ Cost so far: ${self.session.total_cost:.4f}")

            # ═══════════════════════════════════════════════════════════════════
            # PHASE 7: Gap Assessment (ADK DIFFERENTIATOR - AgentTool)
            # ═══════════════════════════════════════════════════════════════════
            self._log("\n┌─ PHASE 7: Gap Assessment (AgentTool)")
            self._log("│  ADK PATTERN: Agent wrapped as tool (AgentTool)")
            self._log("│  Student simulation reviews course for gaps")

            # Create student simulator agent
            student_agent = LlmAgent(
                name="StudentSimulator",
                instruction="""You are a beginner student reviewing this course.

Identify:
1. Concepts that are unclear or poorly explained
2. Missing prerequisites that should be covered first
3. Gaps in the logical progression
4. Topics that need more examples

Return ONLY valid JSON:
{
  "gaps_found": ["...", "..."],
  "missing_prerequisites": ["..."],
  "unclear_concepts": ["..."],
  "recommendations": ["..."],
  "ready_for_publication": true
}""",
                output_key="gap_assessment",
                model=self.cheap_model,
                cost_phase="gap_assessment"
            )

            # Wrap as AgentTool (ADK pattern)
            gap_assessor_tool = AgentTool(
                agent=student_agent,
                name="GapAssessor",
                description="Assess course for gaps from student perspective"
            )

            # Prepare course summary for assessment
            course_summary = f"""
COURSE: {syllabus_data.get('course_title', topic)}
OBJECTIVE: {syllabus_data.get('course_objective', '')}

LESSONS:
"""
            for l in lessons:
                course_summary += f"\n{l.get('lesson_number', 0)}. {l.get('title', '')}"
                course_summary += f"\n   Objectives: {', '.join(l.get('objectives', [])[:2])}"

            self._log(f"│  → Invoking AgentTool: GapAssessor")
            gap_assessor_tool(self.session, course_summary)
            self.api_calls += 1

            gap_json = self._parse_json(self.session.get("gap_assessment", "{}"))
            final_gap_assessment = GapAssessment(
                gaps_found=gap_json.get("gaps_found", []),
                missing_prerequisites=gap_json.get("missing_prerequisites", []),
                unclear_concepts=gap_json.get("unclear_concepts", []),
                recommendations=gap_json.get("recommendations", []),
                ready_for_publication=gap_json.get("ready_for_publication", False)
            )

            self._log(f"│  → Gaps found: {len(final_gap_assessment.gaps_found)}")
            self._log(f"│  → Ready for publication: {final_gap_assessment.ready_for_publication}")
            self._log(f"└─ Cost so far: ${self.session.total_cost:.4f}")

            # ═══════════════════════════════════════════════════════════════════
            # PHASE 8: Gap-Driven Refinement (ADK DIFFERENTIATOR - LoopAgent Escalation)
            # ═══════════════════════════════════════════════════════════════════
            refinement_iteration = 0

            # Determine if refinement is needed based on gap assessment
            total_issues = (
                len(final_gap_assessment.gaps_found) +
                len(final_gap_assessment.missing_prerequisites) +
                len(final_gap_assessment.unclear_concepts)
            )
            should_refine = not final_gap_assessment.ready_for_publication and total_issues >= 2

            if should_refine:
                self._log("\n" + "═" * 60)
                self._log("GAP-DRIVEN REFINEMENT (LOOPAGENT ESCALATION PATTERN)")
                self._log("═" * 60)
                self._log("│  ADK DIFFERENTIATOR: LoopAgent with escalation signal")
                self._log("│  Gap assessment triggers refinement loop via escalation_key")

                refinement_iteration += 1
                self.session.set("refinement_iteration", refinement_iteration)

                # Build gap context for refinement
                gap_context = f"""GAP ASSESSMENT ESCALATION SIGNAL:

GAPS IDENTIFIED: {', '.join(final_gap_assessment.gaps_found[:3]) or 'None'}
MISSING PREREQUISITES: {', '.join(final_gap_assessment.missing_prerequisites[:3]) or 'None'}
UNCLEAR CONCEPTS: {', '.join(final_gap_assessment.unclear_concepts[:3]) or 'None'}
RECOMMENDATIONS: {', '.join(final_gap_assessment.recommendations[:3]) or 'None'}
"""
                self.session.set("gap_context", gap_context)

                # Create lesson refiner agent (ADK pattern)
                lesson_refiner = LlmAgent(
                    name="LessonRefiner",
                    instruction="""You received an ESCALATION SIGNAL from the GapAssessor agent.
Improve this lesson to address the identified gaps.

Gap context: {{gap_context}}

Return ONLY valid JSON with improved lesson:
{
  "lesson_number": 1,
  "title": "...",
  "objectives": ["..."],
  "content_outline": ["..."],
  "activities": ["..."],
  "resources": ["..."],
  "citations": ["..."]
}""",
                    output_key="refined_lesson",
                    model=self.balanced_model,
                    cost_phase="gap_refinement"
                )

                self._log(f"│  Refining {len(lessons)} lessons via LoopAgent escalation...")

                refined_lessons = []
                for i, lesson in enumerate(lessons):
                    lesson_num = lesson.get("lesson_number", i + 1)
                    self.session.set("current_lesson_to_refine", json.dumps(lesson))

                    # Use LlmAgent with gap context (escalation pattern)
                    lesson_refiner.run(
                        self.session,
                        f"Refine this lesson:\n{json.dumps(lesson, indent=2)}"
                    )
                    self.api_calls += 1

                    refined_content = self.session.get("refined_lesson", "{}")
                    refined_data = self._parse_json(refined_content)
                    if refined_data:
                        refined_data["lesson_number"] = lesson_num
                        refined_lessons.append(refined_data)
                    else:
                        refined_lessons.append(lesson)

                    if i < 2 or i == len(lessons) - 1:
                        self._log(f"│    Lesson {lesson_num}: escalation → refined")

                # Replace lessons with refined versions
                lessons = refined_lessons

                # Clear escalation signal (ADK pattern)
                self.session.set("refinement_escalation", False)

                self._log(f"│  → All lessons refined via LoopAgent escalation")
                self._log(f"└─ Cost so far: ${self.session.total_cost:.4f}")

                # Re-assess gaps after refinement
                self._log("\n┌─ Re-assessing gaps after refinement...")
                course_summary = f"COURSE: {syllabus_data.get('course_title', topic)}\n\nLESSONS:\n"
                for l in lessons:
                    course_summary += f"\n{l.get('lesson_number', 0)}. {l.get('title', '')}"
                    course_summary += f"\n   Objectives: {', '.join(l.get('objectives', [])[:2])}"

                gap_assessor_tool(self.session, course_summary)
                self.api_calls += 1

                gap_json = self._parse_json(self.session.get("gap_assessment", "{}"))
                final_gap_assessment = GapAssessment(
                    gaps_found=gap_json.get("gaps_found", []),
                    missing_prerequisites=gap_json.get("missing_prerequisites", []),
                    unclear_concepts=gap_json.get("unclear_concepts", []),
                    recommendations=gap_json.get("recommendations", []),
                    ready_for_publication=gap_json.get("ready_for_publication", True)
                )
                self._log(f"│  → Post-refinement gaps: {len(final_gap_assessment.gaps_found)}")
                self._log(f"└─ Refinement complete")
            else:
                self._log("\n┌─ Course ready - no gap-driven refinement needed")
                self._log(f"└─ Proceeding to output...")

            # ═══════════════════════════════════════════════════════════════════
            # BUILD FINAL OUTPUT
            # ═══════════════════════════════════════════════════════════════════
            self._log("\n┌─ FINAL: Compiling Course Package")

            # Build lesson plans
            lesson_plans = []
            for l in lessons:
                lesson_plans.append(LessonPlan(
                    lesson_number=l.get("lesson_number", 0),
                    title=l.get("title", ""),
                    objectives=l.get("objectives", []),
                    content_outline=l.get("content_outline", []),
                    activities=l.get("activities", []),
                    resources=l.get("resources", []),
                    citations=l.get("citations", [])
                ))

            # Build syllabus
            syllabus = Syllabus(
                course_title=syllabus_data.get("course_title", topic),
                course_objective=syllabus_data.get("course_objective", ""),
                target_audience=syllabus_data.get("target_audience", "General learners"),
                difficulty_level=syllabus_data.get("difficulty_level", "Intermediate"),
                lessons=lesson_plans
            )

            # Build cost breakdown
            cost_data = self.session.get_cost_breakdown()
            cost_breakdown = CostBreakdown(
                research_cost=cost_data["research_cost"],
                syllabus_cost=cost_data["syllabus_cost"],
                quality_loop_cost=cost_data["quality_loop_cost"],
                lesson_generation_cost=cost_data["lesson_generation_cost"],
                gap_assessment_cost=cost_data["gap_assessment_cost"],
                total_cost=cost_data["total_cost"],
                total_tokens=cost_data["total_tokens"]
            )

            # Build legacy course package (backward compatibility)
            course = CoursePackage(
                syllabus=syllabus,
                research_sources=citations,
                generation_metadata={
                    "framework": "Google ADK",
                    "patterns_demonstrated": [
                        "ParallelAgent (parallel research)",
                        "LoopAgent (quality refinement)",
                        "AgentTool (gap assessment)",
                        "ESCALATION (gap-driven refinement)",  # NEW
                        "output_key (state sharing)",
                        "{{template}} substitution"
                    ],
                    "total_cost": cost_breakdown.total_cost,
                    "total_tokens": cost_breakdown.total_tokens,
                    "api_calls": self.api_calls,
                    "jina_calls": self.jina_calls,
                    "quality_iterations": quality_iteration,
                    "refinement_iterations": refinement_iteration  # NEW
                }
            )

            # Build enhanced course package
            enhanced_course = EnhancedCoursePackage(
                syllabus=syllabus,
                quality_score=final_quality_score,
                gap_assessment=final_gap_assessment,
                cost_breakdown=cost_breakdown,
                research_sources=citations,
                generation_metadata={
                    "framework": "Google ADK",
                    "patterns_demonstrated": [
                        "ParallelAgent (parallel research)",
                        "LoopAgent (quality refinement)",
                        "AgentTool (gap assessment)",
                        "ESCALATION (gap-driven refinement)",  # NEW
                        "output_key (state sharing)",
                        "{{template}} substitution"
                    ],
                    "models_used": {
                        "cheap": CHEAP_MODEL,
                        "balanced": BALANCED_MODEL
                    },
                    "quality_iterations": quality_iteration,
                    "refinement_iterations": refinement_iteration  # NEW
                }
            )

            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()

            self._log(f"│")
            self._log(f"│  ╔════════════════════════════════════════════╗")
            self._log(f"│  ║  GOOGLE ADK WORKFLOW COMPLETE              ║")
            self._log(f"│  ╠════════════════════════════════════════════╣")
            self._log(f"│  ║  Lessons:     {len(lesson_plans):<26} ║")
            self._log(f"│  ║  Duration:    {duration:.1f}s{' ':<23} ║")
            self._log(f"│  ║  Total Cost:  ${cost_breakdown.total_cost:.4f}{' ':<20} ║")
            self._log(f"│  ║  Quality:     {final_quality_score.score if final_quality_score else 'N/A':<26} ║")
            self._log(f"│  ╚════════════════════════════════════════════╝")
            self._log(f"└─")

            metrics = GenerationMetrics(
                framework="Google ADK",
                total_tokens=self.session.total_tokens,
                api_calls=self.api_calls,
                jina_calls=self.jina_calls
            )
            metrics.start_time = self.start_time
            metrics.end_time = self.end_time

            return FrameworkResult(
                framework="Google ADK",
                success=True,
                course=course,
                enhanced_course=enhanced_course,
                metrics=metrics,
                console_output=self.console_log
            )

        except Exception as e:
            self.end_time = datetime.now()
            self.errors.append(str(e))
            self._log(f"ERROR: {e}")

            import traceback
            self._log(traceback.format_exc())

            metrics = GenerationMetrics(framework="Google ADK")
            metrics.start_time = self.start_time
            metrics.end_time = self.end_time
            metrics.errors = self.errors

            return FrameworkResult(
                framework="Google ADK",
                success=False,
                error=str(e),
                metrics=metrics,
                console_output=self.console_log
            )


def generate_course(prompt: str, callback: Callable[[str], None] = None):
    """
    Generate a course using Google ADK-style workflow.

    Entry point that creates the hierarchical agent structure.
    """
    generator = GoogleADKCourseGenerator(callback)
    return generator.run(prompt)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")

    result = generate_course("Create a course on Data Science Fundamentals")
    print(f"\nSuccess: {result.success}")
    if result.course:
        print(f"Lessons: {len(result.course.syllabus.lessons)}")
    if result.enhanced_course:
        print(f"Quality Score: {result.enhanced_course.quality_score.score if result.enhanced_course.quality_score else 'N/A'}")
        print(f"Total Cost: ${result.enhanced_course.cost_breakdown.total_cost:.4f}" if result.enhanced_course.cost_breakdown else "")
    if result.metrics:
        print(f"Duration: {result.metrics.duration_seconds:.1f}s")
