"""
Orchestral AI Course Generator - ENHANCED IMPLEMENTATION

Demonstrates Orchestral AI patterns:
- Provider-agnostic design (Claude/GPT switchable with one line)
- CheapLLM for automatic cost optimization
- Built-in cost tracking per phase
- Hooks system for approval workflow
- Subagent pattern for gap assessment
- Synchronous execution for deterministic debugging

All LLM calls go through OpenRouter.
"""
import os
import sys
import json
import httpx
from typing import List, Callable, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI

# ============================================================================
# OpenRouter LLM Setup - Provider Agnostic Design
# ============================================================================

# 2026 Models - Updated for current OpenRouter pricing
PROVIDERS = {
    "cheap": "deepseek/deepseek-v3.2",  # Best value 2026 - $0.25/$0.38
    "balanced": "google/gemini-3-flash-preview",  # 1M context - $0.50/$3
    "quality": "anthropic/claude-sonnet-4",  # Premium quality - $3/$15
    # Free alternatives
    "free-coding": "xiaomi/mimo-v2-flash",  # FREE - 309B MoE
    "free-agents": "mistralai/devstral-2-2512",  # FREE - 123B agentic
}

PRICING = {
    "deepseek/deepseek-v3.2": {"input": 0.25, "output": 0.38},
    "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "xiaomi/mimo-v2-flash": {"input": 0.0, "output": 0.0},
    "mistralai/devstral-2-2512": {"input": 0.0, "output": 0.0},
}


def get_openrouter_client() -> OpenAI:
    """Get OpenAI client configured for OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", "")
    )


# ============================================================================
# Orchestral Context - Built-in Cost Tracking
# ============================================================================

@dataclass
class Context:
    """
    Orchestral Context with built-in cost tracking.
    Core differentiator: automatic cost aggregation across all API calls.
    """
    session_id: str = ""
    total_cost: float = 0.0
    total_tokens: int = 0
    costs_by_phase: Dict[str, float] = field(default_factory=dict)
    tokens_by_phase: Dict[str, int] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    def add_cost(self, phase: str, cost: float, tokens: int):
        """Track cost for a specific phase."""
        self.total_cost += cost
        self.total_tokens += tokens
        self.costs_by_phase[phase] = self.costs_by_phase.get(phase, 0) + cost
        self.tokens_by_phase[phase] = self.tokens_by_phase.get(phase, 0) + tokens

    def save_json(self, path: str):
        """Save context to JSON for persistence."""
        with open(path, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "total_cost": self.total_cost,
                "total_tokens": self.total_tokens,
                "costs_by_phase": self.costs_by_phase,
                "state": self.state
            }, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> 'Context':
        """Load context from JSON - enables mid-conversation provider switching."""
        with open(path, 'r') as f:
            data = json.load(f)
        ctx = cls()
        ctx.session_id = data.get("session_id", "")
        ctx.total_cost = data.get("total_cost", 0.0)
        ctx.total_tokens = data.get("total_tokens", 0)
        ctx.costs_by_phase = data.get("costs_by_phase", {})
        ctx.state = data.get("state", {})
        return ctx


# ============================================================================
# Provider Classes - One-Line Switching
# ============================================================================

class BaseLLM:
    """Base class for LLM providers."""

    def __init__(self, model: str):
        self.model = model
        self.client = get_openrouter_client()
        self.provider_name = "Base"

    def chat(self, messages: List[Dict], max_tokens: int = 2000) -> Dict:
        """Execute chat completion and return content with usage."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        pricing = PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        cost = (prompt_tokens / 1_000_000) * pricing["input"] + \
               (completion_tokens / 1_000_000) * pricing["output"]

        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.total_tokens if usage else 0,
                "cost": cost
            }
        }


class QualityLLM(BaseLLM):
    """Premium quality model via OpenRouter - for critical tasks."""

    def __init__(self, model: str = "quality"):
        model_id = PROVIDERS.get(model, PROVIDERS["quality"])
        super().__init__(model_id)
        self.provider_name = f"QualityLLM ({model})"


class BalancedLLM(BaseLLM):
    """Balanced model via OpenRouter - good quality/cost ratio."""

    def __init__(self, model: str = "balanced"):
        model_id = PROVIDERS.get(model, PROVIDERS["balanced"])
        super().__init__(model_id)
        self.provider_name = f"BalancedLLM ({model})"


class CheapLLM(BaseLLM):
    """
    Auto-selects the cheapest available model.
    Orchestral's unique cost optimization feature.
    2026: DeepSeek V3.2 at $0.25/$0.38 per 1M tokens
    """

    def __init__(self):
        super().__init__(PROVIDERS["cheap"])
        self.provider_name = "CheapLLM (DeepSeek V3.2)"


class FreeLLM(BaseLLM):
    """
    Free model option - uses MiMo-V2-Flash.
    2026: Xiaomi's 309B MoE at $0/0$ - matches Claude Sonnet on SWE-bench!
    """

    def __init__(self, use_case: str = "coding"):
        model_key = "free-coding" if use_case == "coding" else "free-agents"
        super().__init__(PROVIDERS[model_key])
        self.provider_name = f"FreeLLM ({model_key})"


# ============================================================================
# Jina Tools with @define_tool pattern
# ============================================================================

def define_tool(description: str = ""):
    """Decorator simulating Orchestral's @define_tool auto-schema generation."""
    def decorator(func):
        func.is_tool = True
        func.description = description or func.__doc__ or ""
        func.name = func.__name__
        return func
    return decorator


@define_tool("Search the web for information")
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


@define_tool("Read URL content as markdown")
def jina_read(url: str) -> dict:
    """Read URL content using Jina Reader API."""
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
# Orchestral Agent Class
# ============================================================================

class Agent:
    """
    Orchestral Agent with provider-agnostic design and cost tracking.

    Key features:
    - Switch providers: agent.llm = QualityLLM() or FreeLLM()
    - Automatic cost tracking via Context
    - Synchronous execution for debugging
    - Explicit tool invocation with use_tool()
    """

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str = "",
        tools: List[Callable] = None,
        context: Context = None
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in (tools or []) if hasattr(t, 'name')}
        self.context = context or Context()

    def run(self, prompt: str, phase: str = "default", max_tokens: int = 2000) -> str:
        """Execute agent with given prompt. Phase is used for cost breakdown."""
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        result = self.llm.chat(messages, max_tokens)

        self.context.add_cost(
            phase,
            result["usage"]["cost"],
            result["usage"]["total_tokens"]
        )

        return result["content"]

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Explicitly invoke a tool - Orchestral pattern."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        return self.tools[tool_name](**kwargs)


# ============================================================================
# Subagent Pattern - Tool containing Agent
# ============================================================================

class StudentSimulatorSubagent:
    """
    Subagent that simulates student experience.
    Orchestral pattern: Tool containing its own Agent instance.
    """

    def __init__(self, llm: BaseLLM = None):
        self.llm = llm or CheapLLM()
        self.name = "student_simulator"
        self.is_tool = True

    def __call__(self, course_content: str, context: Context) -> dict:
        """Run student simulation and return gap assessment."""
        student = Agent(
            llm=self.llm,
            system_prompt="""You are a beginner student reviewing this course.
Identify:
1. Concepts that confuse you
2. Prerequisites that should have been covered earlier
3. Gaps in the logical progression

Return ONLY valid JSON:
{
  "gaps_found": ["..."],
  "missing_prerequisites": ["..."],
  "unclear_concepts": ["..."],
  "recommendations": ["..."],
  "ready_for_publication": true/false
}""",
            context=context
        )

        result = student.run(
            f"Review this course as a beginner:\n{course_content}",
            phase="gap_assessment"
        )

        try:
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            return json.loads(result.strip())
        except:
            return {
                "gaps_found": [],
                "missing_prerequisites": [],
                "unclear_concepts": [],
                "recommendations": [],
                "ready_for_publication": True
            }


# ============================================================================
# Hook System - Approval Workflow
# ============================================================================

class ToolHookResult:
    """Result from a tool hook."""
    def __init__(self, approved: bool, message: str = "", should_interrupt: bool = False):
        self.approved = approved
        self.message = message
        self.should_interrupt = should_interrupt


class ApprovalHook:
    """Pre-execution hook for human-in-the-loop approval."""

    def __init__(self, approval_callback: Callable = None):
        self.approval_callback = approval_callback or self._auto_approve
        self.approved_items = set()

    def _auto_approve(self, item: Any) -> tuple:
        """Auto-approve for demo purposes."""
        return True, None

    def check(self, item_id: str, item_data: Any) -> ToolHookResult:
        """Check if item is approved."""
        if item_id in self.approved_items:
            return ToolHookResult(approved=True)

        approved, feedback = self.approval_callback(item_data)
        if approved:
            self.approved_items.add(item_id)
            return ToolHookResult(approved=True)

        return ToolHookResult(
            approved=False,
            message=f"Not approved: {feedback}",
            should_interrupt=True
        )


# ============================================================================
# Gap-Driven Refinement Hook (ORCHESTRAL DIFFERENTIATOR)
# ============================================================================

class RefinementHook:
    """
    Post-execution hook that intercepts gap assessment and triggers refinement.

    ORCHESTRAL DIFFERENTIATOR: Hook system for workflow control.
    Hooks can intercept results and modify workflow execution.
    """

    def __init__(self, threshold: int = 2, max_iterations: int = 1):
        self.threshold = threshold  # Minimum issues to trigger refinement
        self.max_iterations = max_iterations
        self.iterations = 0

    def should_refine(self, gap_data: dict) -> ToolHookResult:
        """
        Check if gap assessment warrants refinement.

        ORCHESTRAL PATTERN: Hook inspects result and decides next action.
        """
        if self.iterations >= self.max_iterations:
            return ToolHookResult(
                approved=False,
                message="Max refinement iterations reached",
                should_interrupt=False
            )

        total_issues = (
            len(gap_data.get("gaps_found", [])) +
            len(gap_data.get("missing_prerequisites", [])) +
            len(gap_data.get("unclear_concepts", []))
        )

        if not gap_data.get("ready_for_publication", True) and total_issues >= self.threshold:
            self.iterations += 1
            return ToolHookResult(
                approved=True,
                message=f"Refinement triggered: {total_issues} issues found",
                should_interrupt=True  # Interrupt to trigger refinement
            )

        return ToolHookResult(
            approved=False,
            message="Course ready, no refinement needed",
            should_interrupt=False
        )


class LessonRefinerSubagent:
    """
    Subagent for gap-driven lesson refinement.

    ORCHESTRAL DIFFERENTIATOR: Subagent pattern - Tool containing its own Agent.
    Can be composed and reused across different workflows.
    """

    def __init__(self, llm: BaseLLM = None):
        self.llm = llm or BalancedLLM()
        self.name = "lesson_refiner"
        self.is_tool = True

    def __call__(self, lesson: dict, gap_context: str, context: Context) -> dict:
        """Refine a single lesson based on gap context."""
        refiner = Agent(
            llm=self.llm,
            system_prompt="""You are an expert course designer refining lessons.
Based on student feedback (gap assessment), improve this lesson to address:
- Any unclear concepts
- Missing prerequisites
- Logical progression issues

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
            context=context
        )

        result = refiner.run(
            f"Gap context:\n{gap_context}\n\nLesson to refine:\n{json.dumps(lesson, indent=2)}",
            phase="gap_refinement"
        )

        try:
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            refined = json.loads(result.strip())
            refined["lesson_number"] = lesson.get("lesson_number", 0)
            return refined
        except:
            return lesson  # Return original if parsing fails


# ============================================================================
# Enhanced Course Generator
# ============================================================================

class OrchestralCourseGenerator:
    """
    Enhanced Course Generator using Orchestral AI patterns.

    Features demonstrated:
    1. Provider-agnostic design (Claude/GPT switchable)
    2. CheapLLM for cost optimization
    3. Built-in cost tracking per phase
    4. Hooks system for approval
    5. Subagent pattern for gap assessment
    6. Synchronous execution for debugging
    """

    def __init__(self, callback: Callable[[str], None] = None):
        self.callback = callback or print
        self.console_log = []

        # Orchestral Context for cost tracking
        self.context = Context(session_id=f"course-{datetime.now().isoformat()}")

        # Metrics
        self.api_calls = 0
        self.jina_calls = 0
        self.start_time = None
        self.end_time = None
        self.errors = []

        # Approval hook
        self.approval_hook = ApprovalHook()

        # Subagent for gap assessment
        self.student_subagent = StudentSimulatorSubagent()

        # ─────────────────────────────────────────────────────────────────────
        # Gap-Driven Refinement (HOOK + SUBAGENT PATTERN)
        # ─────────────────────────────────────────────────────────────────────
        # Orchestral's unique patterns: hooks intercept workflow,
        # subagents encapsulate specialized logic

        self.refinement_hook = RefinementHook(threshold=2, max_iterations=1)
        self.lesson_refiner_subagent = LessonRefinerSubagent()
        self.refinement_iterations = 0

        # ─────────────────────────────────────────────────────────────────────
        # Define Agents with different providers for cost optimization
        # ─────────────────────────────────────────────────────────────────────

        self.topic_agent = Agent(
            llm=CheapLLM(),
            system_prompt="Extract the course topic from the user request. Return only the topic name.",
            context=self.context
        )

        self.research_agent = Agent(
            llm=CheapLLM(),
            system_prompt="Synthesize research findings into key points for course creation.",
            tools=[jina_search, jina_read],
            context=self.context
        )

        self.syllabus_agent = Agent(
            llm=BalancedLLM(),
            system_prompt="""Create a 10-lesson course syllabus.
Return ONLY valid JSON:
{
  "course_title": "...",
  "course_objective": "...",
  "lessons": [
    {"lesson_number": 1, "title": "...", "objectives": ["..."], "topics": ["..."]}
  ]
}""",
            context=self.context
        )

        self.quality_agent = Agent(
            llm=CheapLLM(),
            system_prompt="""Evaluate syllabus quality.
Return ONLY valid JSON:
{
  "score": 0.0-1.0,
  "feedback": "...",
  "issues": ["..."]
}""",
            context=self.context
        )

        self.refiner_agent = Agent(
            llm=BalancedLLM(),
            system_prompt="Improve the syllabus based on feedback. Return improved syllabus as JSON.",
            context=self.context
        )

        self.lesson_agent = Agent(
            llm=BalancedLLM(),
            system_prompt="""Create a detailed lesson plan.
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
            context=self.context
        )

    def _log(self, msg: str):
        """Log to console."""
        full_msg = f"[Orchestral] {msg}"
        self.console_log.append(full_msg)
        print(full_msg)
        if self.callback:
            self.callback(full_msg)

    def _research_with_cost_tracking(self, topic: str) -> dict:
        """Research with per-source cost tracking."""
        self._log("Phase 1: Research (with cost tracking)")

        sources = ["academic", "tutorial", "documentation"]
        results = {}

        for source in sources:
            query = f"{topic} {source} guide tutorial"
            self._log(f"  → Searching: {source}")

            search_result = jina_search(query)
            self.jina_calls += 1

            results[source] = search_result.get("results", "")
            self._log(f"    Cost so far: ${self.context.total_cost:.4f}")

        combined = "\n\n".join([
            f"=== {k.upper()} ===\n{v[:3000]}"
            for k, v in results.items()
        ])

        self._log("  → Synthesizing research...")
        synthesis = self.research_agent.run(
            f"Synthesize:\n{combined}",
            phase="research"
        )
        self.api_calls += 1

        self._log(f"  → Research phase cost: ${self.context.costs_by_phase.get('research', 0):.4f}")

        return {"combined": combined, "synthesis": synthesis}

    def _quality_loop_with_cost_per_iteration(self, topic: str, research: str) -> dict:
        """Quality loop tracking cost per iteration."""
        self._log("Phase 2: Syllabus + Quality Loop")

        syllabus = None
        quality_score = None
        iteration_costs = []
        max_iterations = 3

        for i in range(max_iterations):
            iter_start_cost = self.context.total_cost
            self._log(f"  Iteration {i + 1}/{max_iterations}")

            if syllabus is None:
                self._log("    → Generating syllabus...")
                result = self.syllabus_agent.run(
                    f"Topic: {topic}\nResearch:\n{research[:4000]}",
                    phase="syllabus"
                )
            else:
                self._log(f"    → Refining (issues: {quality_score.get('issues', [])[:2]})")
                result = self.refiner_agent.run(
                    f"Improve:\n{json.dumps(syllabus)}\nFeedback: {quality_score}",
                    phase="quality_loop"
                )
            self.api_calls += 1

            try:
                content = result
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                syllabus = json.loads(content.strip())
            except:
                syllabus = {"course_title": topic, "course_objective": "", "lessons": []}

            self._log("    → Checking quality...")
            quality_result = self.quality_agent.run(
                f"Evaluate:\n{json.dumps(syllabus, indent=2)}",
                phase="quality_loop"
            )
            self.api_calls += 1

            try:
                qcontent = quality_result
                if "```json" in qcontent:
                    qcontent = qcontent.split("```json")[1].split("```")[0]
                elif "```" in qcontent:
                    qcontent = qcontent.split("```")[1].split("```")[0]
                quality_score = json.loads(qcontent.strip())
            except:
                quality_score = {"score": 0.5, "feedback": "Parse error", "issues": []}

            iter_cost = self.context.total_cost - iter_start_cost
            iteration_costs.append({"iteration": i + 1, "cost": iter_cost})

            score = quality_score.get("score", 0)
            self._log(f"    → Score: {score:.2f}, Iteration cost: ${iter_cost:.4f}")

            if score >= 0.8:
                self._log("    → Quality threshold met!")
                break

        return {
            "syllabus": syllabus,
            "quality_score": quality_score,
            "iterations": i + 1,
            "iteration_costs": iteration_costs
        }

    def _approval_checkpoint(self, syllabus: dict) -> bool:
        """Approval checkpoint using hook system."""
        self._log("Phase 3: Approval Checkpoint (hook system)")

        result = self.approval_hook.check("syllabus", syllabus)
        if result.approved:
            self._log("  → Approved (auto-approved for demo)")
            return True
        else:
            self._log(f"  → Rejected: {result.message}")
            return False

    def _generate_lessons(self, topic: str, syllabus: dict) -> List[dict]:
        """Generate lessons with cost tracking."""
        self._log("Phase 4: Lesson Generation")

        lessons = []
        lesson_infos = syllabus.get("lessons", [])[:10]

        for i, lesson_info in enumerate(lesson_infos):
            lesson_num = i + 1
            title = lesson_info.get("title", f"Lesson {lesson_num}")

            lesson_start_cost = self.context.total_cost
            self._log(f"  Lesson {lesson_num}/10: {title}")

            self._log("    → Researching...")
            search_result = jina_search(f"{topic} {title} tutorial")
            self.jina_calls += 1
            research = search_result.get("results", "")

            self._log("    → Generating...")
            result = self.lesson_agent.run(
                f"Lesson: {json.dumps(lesson_info)}\nCourse: {topic}\nResearch: {research[:2000]}",
                phase="lessons",
                max_tokens=1500
            )
            self.api_calls += 1

            try:
                content = result
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                lesson_data = json.loads(content.strip())
                lesson_data["lesson_number"] = lesson_num
            except:
                lesson_data = {
                    "lesson_number": lesson_num,
                    "title": title,
                    "objectives": lesson_info.get("objectives", []),
                    "content_outline": lesson_info.get("topics", []),
                    "activities": [],
                    "resources": [],
                    "citations": []
                }

            lesson_cost = self.context.total_cost - lesson_start_cost
            self._log(f"    → Cost: ${lesson_cost:.4f}")
            lessons.append(lesson_data)

        return lessons

    def _gap_assessment_with_subagent(self, course: dict) -> dict:
        """Gap assessment using subagent pattern."""
        self._log("Phase 5: Gap Assessment (subagent)")

        course_summary = json.dumps({
            "title": course.get("syllabus", {}).get("course_title", ""),
            "objective": course.get("syllabus", {}).get("course_objective", ""),
            "lessons": [
                {"number": l.get("lesson_number"), "title": l.get("title")}
                for l in course.get("lessons", [])
            ]
        }, indent=2)

        self._log("  → Running student subagent...")
        gap_data = self.student_subagent(course_summary, self.context)
        self.api_calls += 1

        gaps_count = len(gap_data.get("gaps_found", []))
        self._log(f"  → Found {gaps_count} gaps")
        self._log(f"  → Subagent cost: ${self.context.costs_by_phase.get('gap_assessment', 0):.4f}")

        return gap_data

    def _gap_driven_refinement_with_hook(self, lessons: List[dict], gap_data: dict, topic: str) -> List[dict]:
        """
        Refine lessons using hook + subagent pattern.

        ORCHESTRAL DIFFERENTIATORS:
        1. RefinementHook intercepts gap assessment and decides action
        2. LessonRefinerSubagent encapsulates refinement logic
        3. Synchronous execution for deterministic debugging
        """
        self._log("=" * 50)
        self._log("GAP-DRIVEN REFINEMENT (HOOK + SUBAGENT PATTERN)")
        self._log("=" * 50)
        self._log("│  ORCHESTRAL DIFFERENTIATOR: Hook intercepts + Subagent executes")
        self._log(f"│  Refinement iteration: {self.refinement_iterations + 1}")

        self.refinement_iterations += 1

        # Build gap context
        gap_context = f"""GAP ASSESSMENT HOOK INTERCEPT:

GAPS IDENTIFIED: {', '.join(gap_data.get('gaps_found', [])[:3]) or 'None'}
MISSING PREREQUISITES: {', '.join(gap_data.get('missing_prerequisites', [])[:3]) or 'None'}
UNCLEAR CONCEPTS: {', '.join(gap_data.get('unclear_concepts', [])[:3]) or 'None'}
RECOMMENDATIONS: {', '.join(gap_data.get('recommendations', [])[:3]) or 'None'}
"""

        self._log(f"│  Refining {len(lessons)} lessons via subagent pattern...")

        refined_lessons = []
        for i, lesson in enumerate(lessons):
            lesson_num = lesson.get("lesson_number", i + 1)

            # Use LessonRefinerSubagent (Orchestral subagent pattern)
            refined = self.lesson_refiner_subagent(lesson, gap_context, self.context)
            self.api_calls += 1
            refined_lessons.append(refined)

            if i < 2 or i == len(lessons) - 1:
                self._log(f"│    Lesson {lesson_num}: hook → subagent → refined")

        self._log(f"│  → All lessons refined via hook+subagent pattern")
        self._log(f"└─ Refinement cost: ${self.context.costs_by_phase.get('gap_refinement', 0):.4f}")

        return refined_lessons

    def run(self, prompt: str):
        """Execute enhanced course generation."""
        from common.models import (
            LessonPlan, Syllabus, CoursePackage, EnhancedCoursePackage,
            GenerationMetrics, FrameworkResult, QualityScore, GapAssessment, CostBreakdown
        )

        self.start_time = datetime.now()
        citations = []

        try:
            self._log("=" * 50)
            self._log("Enhanced Orchestral Agent Started")
            self._log(f"Provider: {self.syllabus_agent.llm.provider_name}")
            self._log("=" * 50)

            # Extract topic
            self._log("Extracting topic...")
            topic = self.topic_agent.run(prompt, phase="research", max_tokens=100).strip()
            self.api_calls += 1
            self._log(f"  → Topic: {topic}")

            # Phase 1: Research
            research = self._research_with_cost_tracking(topic)
            citations.append(f"Research: {topic}")

            # Phase 2: Quality Loop
            quality_result = self._quality_loop_with_cost_per_iteration(topic, research["synthesis"])
            syllabus_data = quality_result["syllabus"]
            quality_data = quality_result["quality_score"]

            # Phase 3: Approval
            if not self._approval_checkpoint(syllabus_data):
                raise Exception("Syllabus not approved")

            # Phase 4: Lessons
            lessons = self._generate_lessons(topic, syllabus_data)

            # Phase 5: Gap Assessment
            course_for_review = {"syllabus": syllabus_data, "lessons": lessons}
            gap_data = self._gap_assessment_with_subagent(course_for_review)

            # Phase 6: Gap-Driven Refinement (HOOK + SUBAGENT PATTERN)
            hook_result = self.refinement_hook.should_refine(gap_data)
            self._log(f"Phase 6: Refinement Hook Check")
            self._log(f"  → Hook decision: {hook_result.message}")

            if hook_result.approved and hook_result.should_interrupt:
                # Hook triggered refinement
                lessons = self._gap_driven_refinement_with_hook(lessons, gap_data, topic)

                # Re-assess gaps after refinement
                self._log("Re-assessing gaps after refinement...")
                course_for_review = {"syllabus": syllabus_data, "lessons": lessons}
                gap_data = self._gap_assessment_with_subagent(course_for_review)
            else:
                self._log("  → Course ready - no refinement needed")

            # Build package
            self._log("Compiling enhanced course package...")

            lesson_plans = [
                LessonPlan(
                    lesson_number=l.get("lesson_number", 0),
                    title=l.get("title", ""),
                    objectives=l.get("objectives", []),
                    content_outline=l.get("content_outline", []),
                    activities=l.get("activities", []),
                    resources=l.get("resources", []),
                    citations=l.get("citations", [])
                )
                for l in lessons
            ]

            syllabus = Syllabus(
                course_title=syllabus_data.get("course_title", topic),
                course_objective=syllabus_data.get("course_objective", ""),
                lessons=lesson_plans
            )

            quality_score = QualityScore(
                score=quality_data.get("score", 0.0),
                feedback=quality_data.get("feedback", ""),
                issues=quality_data.get("issues", []),
                iteration=quality_result["iterations"]
            )

            gap_assessment = GapAssessment(
                gaps_found=gap_data.get("gaps_found", []),
                missing_prerequisites=gap_data.get("missing_prerequisites", []),
                unclear_concepts=gap_data.get("unclear_concepts", []),
                recommendations=gap_data.get("recommendations", []),
                ready_for_publication=gap_data.get("ready_for_publication", True)
            )

            cost_breakdown = CostBreakdown(
                research_cost=self.context.costs_by_phase.get("research", 0),
                syllabus_cost=self.context.costs_by_phase.get("syllabus", 0),
                quality_loop_cost=self.context.costs_by_phase.get("quality_loop", 0),
                lesson_generation_cost=self.context.costs_by_phase.get("lessons", 0),
                gap_assessment_cost=self.context.costs_by_phase.get("gap_assessment", 0),
                gap_refinement_cost=self.context.costs_by_phase.get("gap_refinement", 0),  # NEW
                total_tokens=self.context.total_tokens
            )
            cost_breakdown.calculate_total()

            enhanced_course = EnhancedCoursePackage(
                syllabus=syllabus,
                quality_score=quality_score,
                gap_assessment=gap_assessment,
                cost_breakdown=cost_breakdown,
                research_sources=citations,
                generation_metadata={
                    "framework": "Orchestral AI (Enhanced)",
                    "patterns_demonstrated": [
                        "Provider-agnostic design",
                        "CheapLLM (auto cost optimization)",
                        "Subagent pattern (gap assessment)",
                        "HOOK+SUBAGENT (gap-driven refinement)",  # NEW
                        "Context cost tracking",
                        "Synchronous execution"
                    ],
                    "providers_used": {
                        "cheap": "CheapLLM (DeepSeek V3.2)",
                        "balanced": "BalancedLLM (Gemini 3 Flash)"
                    },
                    "iteration_costs": quality_result.get("iteration_costs", []),
                    "refinement_iterations": self.refinement_iterations  # NEW
                }
            )

            course = CoursePackage(
                syllabus=syllabus,
                research_sources=citations,
                generation_metadata=enhanced_course.generation_metadata
            )

            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()

            self._log("=" * 50)
            self._log(f"Complete: {len(lesson_plans)} lessons in {duration:.1f}s")
            self._log(f"Quality: {quality_score.score:.2f}, Gaps: {len(gap_assessment.gaps_found)}")
            self._log(f"Total cost: ${cost_breakdown.total_cost:.4f}")
            self._log("Cost breakdown:")
            for phase, cost in self.context.costs_by_phase.items():
                self._log(f"  {phase}: ${cost:.4f}")
            self._log("=" * 50)

            metrics = GenerationMetrics(
                framework="Orchestral AI (Enhanced)",
                total_tokens=self.context.total_tokens,
                api_calls=self.api_calls,
                jina_calls=self.jina_calls
            )
            metrics.start_time = self.start_time
            metrics.end_time = self.end_time

            return FrameworkResult(
                framework="Orchestral AI (Enhanced)",
                success=True,
                course=course,
                enhanced_course=enhanced_course,
                metrics=metrics,
                console_output=self.console_log
            )

        except Exception as e:
            self.end_time = datetime.now()
            self.errors.append(str(e))
            self._log(f"Error: {e}")

            metrics = GenerationMetrics(framework="Orchestral AI (Enhanced)")
            metrics.start_time = self.start_time
            metrics.end_time = self.end_time
            metrics.errors = self.errors

            return FrameworkResult(
                framework="Orchestral AI (Enhanced)",
                success=False,
                error=str(e),
                metrics=metrics,
                console_output=self.console_log
            )


def generate_course(prompt: str, callback: Callable[[str], None] = None):
    """Generate a course using enhanced Orchestral AI patterns."""
    generator = OrchestralCourseGenerator(callback)
    return generator.run(prompt)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")

    result = generate_course("Create a course on Machine Learning Basics")
    print(f"\nSuccess: {result.success}")
    if result.enhanced_course:
        print(f"Lessons: {len(result.enhanced_course.syllabus.lessons)}")
        print(f"Quality: {result.enhanced_course.quality_score.score:.2f}")
        print(f"Gaps: {len(result.enhanced_course.gap_assessment.gaps_found)}")
        print(f"Cost: ${result.enhanced_course.cost_breakdown.total_cost:.4f}")
    if result.metrics:
        print(f"Duration: {result.metrics.duration_seconds:.1f}s")
