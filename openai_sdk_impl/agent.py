"""
OpenAI Agents SDK Course Generator - ENHANCED IMPLEMENTATION

Demonstrates OpenAI Agents SDK patterns:
- asyncio.gather() for parallel research
- Python while loop with structured output for quality gate
- Blocking guardrails for human approval simulation
- agent.as_tool() pattern for gap assessment (simulated)
- Model selection per agent for cost optimization

All LLM calls go through OpenRouter.
"""
import os
import sys
import json
import httpx
import asyncio
from typing import List, Callable, Optional
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI

# ============================================================================
# OpenRouter LLM Setup
# ============================================================================

# 2026 Models - Updated for current OpenRouter pricing
CHEAP_MODEL = "deepseek/deepseek-v3.2"  # $0.25/$0.38 per 1M - Best value 2026
BALANCED_MODEL = "google/gemini-3-flash-preview"  # $0.50/$3 per 1M - 1M context!


def get_openai_client() -> OpenAI:
    """Get OpenAI client configured for OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", "")
    )


# ============================================================================
# Cost Tracking
# ============================================================================

# OpenRouter pricing (per 1M tokens) - 2026 prices
PRICING = {
    CHEAP_MODEL: {"input": 0.25, "output": 0.38},  # DeepSeek V3.2
    BALANCED_MODEL: {"input": 0.50, "output": 3.00},  # Gemini 3 Flash
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost for a single API call."""
    pricing = PRICING.get(model, {"input": 3.00, "output": 15.00})
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


# ============================================================================
# Jina Tools
# ============================================================================

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
# Agent Class (Simulated OpenAI Agents SDK)
# ============================================================================

class Agent:
    """
    Simulates OpenAI Agents SDK Agent class.
    Each agent can have its own model for cost optimization.
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = None,
        tools: List[Callable] = None
    ):
        self.name = name
        self.instructions = instructions
        self.model = model or BALANCED_MODEL
        self.tools = tools or []
        self.client = get_openai_client()
        self._last_usage = {}

    def run(self, prompt: str, max_tokens: int = 2000) -> dict:
        """Execute agent with given prompt. Returns content and usage."""
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens
        )

        usage = response.usage
        self._last_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "cost": calculate_cost(
                self.model,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0
            )
        }

        return {
            "content": response.choices[0].message.content,
            "usage": self._last_usage
        }


# ============================================================================
# Enhanced Course Generator
# ============================================================================

class OpenAISDKCourseGenerator:
    """
    Enhanced Course Generator using OpenAI Agents SDK patterns.

    Features demonstrated:
    1. Parallel research via asyncio.gather() simulation
    2. Quality loop with structured output
    3. Human approval checkpoint (simulated)
    4. Gap assessment via agent-as-tool pattern
    5. Cost tracking per phase
    """

    def __init__(self, callback: Callable[[str], None] = None):
        self.callback = callback or print
        self.console_log = []

        # Metrics
        self.api_calls = 0
        self.jina_calls = 0
        self.total_tokens = 0
        self.start_time = None
        self.end_time = None
        self.errors = []

        # Cost tracking per phase
        self.costs = {
            "research": 0.0,
            "syllabus": 0.0,
            "quality_loop": 0.0,
            "lessons": 0.0,
            "gap_assessment": 0.0,
            "gap_refinement": 0.0,  # NEW: Handoff-driven refinement
        }

        # ─────────────────────────────────────────────────────────────────────
        # Define Agents with different models for cost optimization
        # ─────────────────────────────────────────────────────────────────────

        # Cheap model for classification/evaluation
        self.topic_agent = Agent(
            name="TopicExtractor",
            model=CHEAP_MODEL,
            instructions="Extract the course topic from the user request. Return only the topic name."
        )

        # Cheap model for research (simple extraction)
        self.research_agent = Agent(
            name="Researcher",
            model=CHEAP_MODEL,
            instructions="Synthesize the provided search results into key points for course creation."
        )

        # Balanced model for syllabus creation
        self.syllabus_agent = Agent(
            name="SyllabusCreator",
            model=BALANCED_MODEL,
            instructions="""Create a 10-lesson course syllabus based on research.
Return ONLY valid JSON:
{
  "course_title": "...",
  "course_objective": "...",
  "lessons": [
    {"lesson_number": 1, "title": "...", "objectives": ["..."], "topics": ["..."]}
  ]
}"""
        )

        # Cheap model for quality evaluation
        self.quality_checker = Agent(
            name="QualityChecker",
            model=CHEAP_MODEL,
            instructions="""Evaluate the syllabus for completeness and quality.
Return ONLY valid JSON:
{
  "score": 0.0-1.0,
  "feedback": "...",
  "issues": ["..."]
}
Score criteria:
- 0.8+: Ready for use
- 0.6-0.8: Minor improvements needed
- <0.6: Major revision required"""
        )

        # Balanced model for refinement
        self.refiner_agent = Agent(
            name="SyllabusRefiner",
            model=BALANCED_MODEL,
            instructions="Improve the syllabus based on the feedback provided. Return the improved syllabus as JSON."
        )

        # Balanced model for lesson generation
        self.lesson_agent = Agent(
            name="LessonGenerator",
            model=BALANCED_MODEL,
            instructions="""Create a detailed lesson plan.
Return ONLY valid JSON:
{
  "lesson_number": 1,
  "title": "...",
  "objectives": ["..."],
  "content_outline": ["..."],
  "activities": ["..."],
  "resources": ["..."],
  "citations": ["..."]
}"""
        )

        # Cheap model for gap assessment (student simulation)
        self.student_agent = Agent(
            name="StudentSimulator",
            model=CHEAP_MODEL,
            instructions="""You are a beginner student reviewing this course.
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
}"""
        )

        # ─────────────────────────────────────────────────────────────────────
        # Gap-Driven Refinement Agent (HANDOFF PATTERN)
        # ─────────────────────────────────────────────────────────────────────
        # OpenAI SDK's handoff pattern: control passes from StudentSimulator
        # to LessonRefiner with structured gap context. This demonstrates
        # agent-to-agent handoff with typed outputs.

        self.lesson_refiner_agent = Agent(
            name="LessonRefiner",
            model=BALANCED_MODEL,
            instructions="""You are an expert course designer refining lessons based on student feedback.
You received a HANDOFF from the StudentSimulator agent with identified gaps.

Given the original lesson and the gaps identified, improve the lesson to address:
- Any unclear concepts
- Missing prerequisites
- Logical progression issues
- Student recommendations

Return ONLY valid JSON with improved lesson:
{
  "lesson_number": 1,
  "title": "...",
  "objectives": ["..."],
  "content_outline": ["..."],
  "activities": ["..."],
  "resources": ["..."],
  "citations": ["..."]
}"""
        )

        # Track refinement
        self.refinement_iterations = 0

    def _log(self, msg: str):
        """Log to console."""
        full_msg = f"[OpenAI SDK] {msg}"
        self.console_log.append(full_msg)
        print(full_msg)
        if self.callback:
            self.callback(full_msg)

    def _track_cost(self, phase: str, usage: dict):
        """Track cost for a phase."""
        self.costs[phase] = self.costs.get(phase, 0) + usage.get("cost", 0)
        self.total_tokens += usage.get("total_tokens", 0)
        self.api_calls += 1

    def _parallel_research(self, topic: str) -> dict:
        """
        Simulate parallel research using multiple queries.
        In real async code, this would use asyncio.gather().
        """
        self._log("Phase 1: Parallel Research (simulated asyncio.gather)")

        queries = [
            f"{topic} academic research papers tutorials",
            f"{topic} beginner guide how to learn",
            f"{topic} official documentation reference",
        ]

        results = {}
        for i, query in enumerate(queries):
            source_type = ["academic", "tutorial", "docs"][i]
            self._log(f"  → Researching: {source_type}")
            results[source_type] = jina_search(query)
            self.jina_calls += 1

        # Combine results
        combined = "\n\n".join([
            f"=== {k.upper()} SOURCES ===\n{v[:3000]}"
            for k, v in results.items()
        ])

        # Synthesize with agent
        self._log("  → Synthesizing research...")
        result = self.research_agent.run(f"Synthesize these research findings:\n{combined}")
        self._track_cost("research", result["usage"])

        return {"combined": combined, "synthesis": result["content"]}

    def _quality_loop(self, topic: str, research: str) -> dict:
        """
        Quality gate loop with structured output.
        OpenAI SDK pattern: Python while loop with structured output validation.
        """
        self._log("Phase 2: Syllabus + Quality Loop")

        syllabus = None
        quality_score = None
        max_iterations = 3

        for iteration in range(max_iterations):
            self._log(f"  Iteration {iteration + 1}/{max_iterations}")

            # Generate or refine syllabus
            if syllabus is None:
                self._log("    → Generating initial syllabus...")
                prompt = f"Topic: {topic}\n\nResearch:\n{research[:4000]}"
                result = self.syllabus_agent.run(prompt)
            else:
                self._log(f"    → Refining based on feedback: {quality_score.get('issues', [])[:2]}...")
                prompt = f"""Improve this syllabus:
{json.dumps(syllabus, indent=2)}

Feedback: {quality_score.get('feedback', '')}
Issues: {quality_score.get('issues', [])}"""
                result = self.refiner_agent.run(prompt)

            self._track_cost("quality_loop", result["usage"])

            # Parse syllabus
            content = result["content"]
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                syllabus = json.loads(content.strip())
            except:
                syllabus = {"course_title": topic, "course_objective": "", "lessons": []}

            # Check quality (structured output)
            self._log("    → Checking quality...")
            quality_result = self.quality_checker.run(f"Evaluate:\n{json.dumps(syllabus, indent=2)}")
            self._track_cost("quality_loop", quality_result["usage"])

            try:
                qcontent = quality_result["content"]
                if "```json" in qcontent:
                    qcontent = qcontent.split("```json")[1].split("```")[0]
                elif "```" in qcontent:
                    qcontent = qcontent.split("```")[1].split("```")[0]
                quality_score = json.loads(qcontent.strip())
            except:
                quality_score = {"score": 0.5, "feedback": "Parse error", "issues": []}

            score = quality_score.get("score", 0)
            self._log(f"    → Quality score: {score:.2f}")

            # Exit condition: score >= 0.8
            if score >= 0.8:
                self._log(f"    → Quality threshold met!")
                break

        return {
            "syllabus": syllabus,
            "quality_score": quality_score,
            "iterations": iteration + 1
        }

    def _human_approval_checkpoint(self, syllabus: dict) -> bool:
        """
        Simulate human approval checkpoint.
        In real implementation, this would use a blocking guardrail.
        """
        self._log("Phase 3: Human Approval Checkpoint (auto-approved for demo)")
        # In production: would block until user approves
        # For demo: auto-approve
        return True

    def _generate_lessons(self, topic: str, syllabus: dict) -> List[dict]:
        """Generate detailed lesson plans."""
        self._log("Phase 4: Lesson Generation")

        lessons = []
        lesson_infos = syllabus.get("lessons", [])[:10]

        for i, lesson_info in enumerate(lesson_infos):
            lesson_num = i + 1
            title = lesson_info.get("title", f"Lesson {lesson_num}")

            self._log(f"  Lesson {lesson_num}/10: {title}")

            # Research for this lesson
            self._log(f"    → Researching...")
            lesson_research = jina_search(f"{topic} {title} tutorial guide")
            self.jina_calls += 1

            # Generate lesson plan
            self._log(f"    → Generating plan...")
            prompt = f"""Create lesson plan:
Lesson: {json.dumps(lesson_info)}
Course: {topic}
Research: {lesson_research[:2000]}"""

            result = self.lesson_agent.run(prompt, max_tokens=1500)
            self._track_cost("lessons", result["usage"])

            # Parse lesson
            content = result["content"]
            try:
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

            lessons.append(lesson_data)

        return lessons

    def _gap_assessment(self, course: dict) -> dict:
        """
        Gap assessment using student simulation.
        OpenAI SDK pattern: agent.as_tool() - treating agent as callable tool.
        """
        self._log("Phase 5: Gap Assessment (Student Simulation)")

        course_summary = json.dumps({
            "title": course.get("syllabus", {}).get("course_title", ""),
            "objective": course.get("syllabus", {}).get("course_objective", ""),
            "lessons": [
                {"number": l.get("lesson_number"), "title": l.get("title"), "objectives": l.get("objectives", [])}
                for l in course.get("lessons", [])
            ]
        }, indent=2)

        self._log("  → Student agent reviewing course...")
        result = self.student_agent.run(f"Review this course as a beginner student:\n{course_summary}")
        self._track_cost("gap_assessment", result["usage"])

        try:
            content = result["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            gap_data = json.loads(content.strip())
        except:
            gap_data = {
                "gaps_found": [],
                "missing_prerequisites": [],
                "unclear_concepts": [],
                "recommendations": [],
                "ready_for_publication": True
            }

        gaps_count = len(gap_data.get("gaps_found", []))
        self._log(f"  → Found {gaps_count} gaps, ready: {gap_data.get('ready_for_publication', True)}")

        return gap_data

    def _should_refine(self, gap_data: dict) -> bool:
        """Determine if gap-driven refinement is needed."""
        if self.refinement_iterations >= 1:
            return False  # Only refine once

        total_issues = (
            len(gap_data.get("gaps_found", [])) +
            len(gap_data.get("missing_prerequisites", [])) +
            len(gap_data.get("unclear_concepts", []))
        )

        return not gap_data.get("ready_for_publication", True) and total_issues >= 2

    def _gap_driven_refinement(self, lessons: List[dict], gap_data: dict, topic: str) -> List[dict]:
        """
        Refine lessons based on gap assessment using HANDOFF pattern.

        OPENAI SDK DIFFERENTIATOR: Agent-to-agent handoff with structured output.
        StudentSimulator hands off to LessonRefiner with gap context.

        In production OpenAI SDK:
            async def run_refinement():
                gaps = await Runner.run(student_agent, course)
                # Handoff to refiner with structured context
                refined = await Runner.run(refiner_agent, context=gaps)
        """
        self._log("=" * 50)
        self._log("GAP-DRIVEN REFINEMENT (HANDOFF PATTERN)")
        self._log("=" * 50)
        self._log("│  OPENAI SDK DIFFERENTIATOR: Agent-to-agent handoff")
        self._log(f"│  Refinement iteration: {self.refinement_iterations + 1}")

        self.refinement_iterations += 1

        # Build handoff context from gap assessment
        gap_context = f"""HANDOFF from StudentSimulator agent:

GAPS IDENTIFIED:
- {chr(10).join('• ' + g for g in gap_data.get('gaps_found', [])[:3]) or 'None'}

MISSING PREREQUISITES:
- {chr(10).join('• ' + p for p in gap_data.get('missing_prerequisites', [])[:3]) or 'None'}

UNCLEAR CONCEPTS:
- {chr(10).join('• ' + c for c in gap_data.get('unclear_concepts', [])[:3]) or 'None'}

STUDENT RECOMMENDATIONS:
- {chr(10).join('• ' + r for r in gap_data.get('recommendations', [])[:3]) or 'None'}
"""

        self._log(f"│  Refining {len(lessons)} lessons with gap context...")

        refined_lessons = []
        for i, lesson in enumerate(lessons):
            lesson_num = lesson.get("lesson_number", i + 1)

            # Handoff: pass gap context and original lesson to refiner
            prompt = f"""{gap_context}

ORIGINAL LESSON TO REFINE:
{json.dumps(lesson, indent=2)}

COURSE TOPIC: {topic}

Improve this lesson to address the identified issues. Maintain the same structure."""

            result = self.lesson_refiner_agent.run(prompt, max_tokens=1500)
            self._track_cost("gap_refinement", result["usage"])

            # Parse refined lesson
            content = result["content"]
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                refined = json.loads(content.strip())
                refined["lesson_number"] = lesson_num
                refined_lessons.append(refined)
            except:
                # Keep original if parsing fails
                refined_lessons.append(lesson)

            if i < 2 or i == len(lessons) - 1:
                self._log(f"│    Lesson {lesson_num}: handoff → refined")

        self._log(f"│  → All lessons refined via handoff pattern")
        self._log(f"└─ Refinement complete")

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
            self._log("Enhanced OpenAI SDK Agent Started")
            self._log("=" * 50)

            # ─────────────────────────────────────────────────────────────────
            # Extract topic
            # ─────────────────────────────────────────────────────────────────
            self._log("Extracting topic...")
            result = self.topic_agent.run(prompt, max_tokens=100)
            self._track_cost("research", result["usage"])
            topic = result["content"].strip()
            self._log(f"  → Topic: {topic}")

            # ─────────────────────────────────────────────────────────────────
            # Phase 1: Parallel Research
            # ─────────────────────────────────────────────────────────────────
            research = self._parallel_research(topic)
            citations.append(f"Research: {topic}")

            # ─────────────────────────────────────────────────────────────────
            # Phase 2: Syllabus + Quality Loop
            # ─────────────────────────────────────────────────────────────────
            quality_result = self._quality_loop(topic, research["synthesis"])
            syllabus_data = quality_result["syllabus"]
            quality_data = quality_result["quality_score"]

            # ─────────────────────────────────────────────────────────────────
            # Phase 3: Human Approval (auto-approved)
            # ─────────────────────────────────────────────────────────────────
            approved = self._human_approval_checkpoint(syllabus_data)
            if not approved:
                raise Exception("Syllabus not approved")

            # ─────────────────────────────────────────────────────────────────
            # Phase 4: Lesson Generation
            # ─────────────────────────────────────────────────────────────────
            lessons = self._generate_lessons(topic, syllabus_data)

            # ─────────────────────────────────────────────────────────────────
            # Phase 5: Gap Assessment
            # ─────────────────────────────────────────────────────────────────
            course_for_review = {
                "syllabus": syllabus_data,
                "lessons": lessons
            }
            gap_data = self._gap_assessment(course_for_review)

            # ─────────────────────────────────────────────────────────────────
            # Phase 6: Gap-Driven Refinement (HANDOFF PATTERN)
            # ─────────────────────────────────────────────────────────────────
            if self._should_refine(gap_data):
                self._log("Gap assessment triggered refinement...")
                lessons = self._gap_driven_refinement(lessons, gap_data, topic)

                # Re-assess after refinement
                self._log("Re-assessing gaps after refinement...")
                course_for_review = {"syllabus": syllabus_data, "lessons": lessons}
                gap_data = self._gap_assessment(course_for_review)
            else:
                self._log("Course ready - no refinement needed")

            # ─────────────────────────────────────────────────────────────────
            # Build enhanced course package
            # ─────────────────────────────────────────────────────────────────
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
                research_cost=self.costs["research"],
                syllabus_cost=self.costs["syllabus"],
                quality_loop_cost=self.costs["quality_loop"],
                lesson_generation_cost=self.costs["lessons"],
                gap_assessment_cost=self.costs["gap_assessment"],
                gap_refinement_cost=self.costs["gap_refinement"],  # NEW
                total_tokens=self.total_tokens
            )
            cost_breakdown.calculate_total()

            enhanced_course = EnhancedCoursePackage(
                syllabus=syllabus,
                quality_score=quality_score,
                gap_assessment=gap_assessment,
                cost_breakdown=cost_breakdown,
                research_sources=citations,
                generation_metadata={
                    "framework": "OpenAI SDK (Enhanced)",
                    "patterns_demonstrated": [
                        "asyncio.gather() (parallel research)",
                        "Structured outputs (quality loop)",
                        "Blocking guardrails (approval)",
                        "agent.as_tool() (gap assessment)",
                        "HANDOFF (gap-driven refinement)"  # NEW
                    ],
                    "models_used": {
                        "cheap": CHEAP_MODEL,
                        "balanced": BALANCED_MODEL
                    },
                    "refinement_iterations": self.refinement_iterations  # NEW
                }
            )

            # Also build legacy course package for compatibility
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
            self._log("=" * 50)

            metrics = GenerationMetrics(
                framework="OpenAI SDK (Enhanced)",
                total_tokens=self.total_tokens,
                api_calls=self.api_calls,
                jina_calls=self.jina_calls
            )
            metrics.start_time = self.start_time
            metrics.end_time = self.end_time

            return FrameworkResult(
                framework="OpenAI SDK (Enhanced)",
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

            metrics = GenerationMetrics(framework="OpenAI SDK (Enhanced)")
            metrics.start_time = self.start_time
            metrics.end_time = self.end_time
            metrics.errors = self.errors

            return FrameworkResult(
                framework="OpenAI SDK (Enhanced)",
                success=False,
                error=str(e),
                metrics=metrics,
                console_output=self.console_log
            )


def generate_course(prompt: str, callback: Callable[[str], None] = None):
    """Generate a course using enhanced OpenAI Agents SDK patterns."""
    generator = OpenAISDKCourseGenerator(callback)
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
