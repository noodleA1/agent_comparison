"""
Streamlit Comparison UI for Agent Framework Comparison - ENHANCED VERSION

Features:
- Saved runs list with course prompts
- 4-panel layout showing all frameworks side-by-side
- Parallel execution for fair timing comparison
- Live console output
- Artifact viewer for generated courses
- Quality scores and gap assessment display
- Cost breakdown per phase
- Framework pattern demonstration
"""
import streamlit as st
import sys
import os
import concurrent.futures
from datetime import datetime
import json
from pathlib import Path

# Add project root to path (relative to this file)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from common.models import FrameworkResult, CoursePackage, EnhancedCoursePackage

# Import generators
from langgraph_impl.agent import generate_course as langgraph_generate
from openai_sdk_impl.agent import generate_course as openai_generate
from google_adk_impl.agent import generate_course as google_generate
from orchestral_impl.agent import generate_course as orchestral_generate

# Import CLI result loader
from cli_runner import load_results, dict_to_framework_result, save_results, run_all_frameworks
RESULTS_DIR = PROJECT_ROOT / "results"

# Page config
st.set_page_config(
    page_title="Agent Framework Comparison",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.framework-header {
    font-size: 1.2em;
    font-weight: bold;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
.langgraph { background-color: #e3f2fd; }
.openai { background-color: #e8f5e9; }
.google { background-color: #fff3e0; }
.orchestral { background-color: #f3e5f5; }
.run-card {
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #ddd;
    margin: 5px 0;
    cursor: pointer;
}
.run-card:hover {
    background-color: #f5f5f5;
    border-color: #1976d2;
}
.run-card-selected {
    background-color: #e3f2fd;
    border-color: #1976d2;
    border-width: 2px;
}
.quality-score {
    font-size: 2em;
    font-weight: bold;
}
.quality-high { color: #2e7d32; }
.quality-medium { color: #f57c00; }
.quality-low { color: #c62828; }
.cost-badge {
    background-color: #e3f2fd;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 0.9em;
}
.pattern-tag {
    background-color: #f5f5f5;
    padding: 3px 8px;
    border-radius: 3px;
    font-size: 0.75em;
    margin: 2px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Helper Functions
# ============================================================================

def get_saved_runs():
    """Get list of saved runs with metadata."""
    runs = []
    if not RESULTS_DIR.exists():
        return runs

    for folder in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if folder.is_dir() and folder.name != "latest":
            results_file = folder / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        data = json.load(f)

                    # Get summary info
                    frameworks = list(data.get("frameworks", {}).keys())
                    successful = sum(1 for f in data.get("frameworks", {}).values() if f.get("success"))

                    runs.append({
                        "folder": folder.name,
                        "path": folder,
                        "prompt": data.get("prompt", "Unknown"),
                        "timestamp": data.get("timestamp", ""),
                        "frameworks": frameworks,
                        "successful": successful,
                        "total": len(frameworks)
                    })
                except Exception as e:
                    pass

    return runs


def get_quality_class(score: float) -> str:
    """Get CSS class for quality score."""
    if score >= 0.8:
        return "quality-high"
    elif score >= 0.6:
        return "quality-medium"
    return "quality-low"


def format_cost(cost: float) -> str:
    """Format cost value."""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def run_framework(name: str, generator_func, prompt: str) -> FrameworkResult:
    """Run a single framework generator."""
    try:
        return generator_func(prompt)
    except Exception as e:
        from common.models import GenerationMetrics
        import traceback
        return FrameworkResult(
            framework=name,
            success=False,
            error=str(e),
            metrics=GenerationMetrics(framework=name),
            console_output=[f"Error: {e}", traceback.format_exc()]
        )


def run_all_frameworks_ui(prompt: str, frameworks: dict) -> dict:
    """Run all selected frameworks in parallel."""
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for name, generator in frameworks.items():
            futures[executor.submit(run_framework, name, generator, prompt)] = name

        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                from common.models import GenerationMetrics
                results[name] = FrameworkResult(
                    framework=name,
                    success=False,
                    error=str(e),
                    metrics=GenerationMetrics(framework=name),
                    console_output=[f"Error: {e}"]
                )

    return results


# ============================================================================
# Initialize Session State
# ============================================================================

if "results" not in st.session_state:
    st.session_state.results = {}
if "running" not in st.session_state:
    st.session_state.running = False
if "selected_run" not in st.session_state:
    st.session_state.selected_run = None
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "list"  # "list", "compare", or "patterns"


# ============================================================================
# Framework Pattern Definitions (for Pattern Detail View)
# ============================================================================

FRAMEWORK_PATTERNS = {
    "LangGraph": {
        "color": "#e3f2fd",
        "icon": "🔄",
        "tagline": "Graph-based Workflow with Time-Travel",
        "core_patterns": [
            "Send API (parallel fan-out)",
            "interrupt() (human-in-the-loop)",
            "Conditional edges (quality loop)",
            "Checkpointer (state persistence)",
            "TIME-TRAVEL (gap-driven refinement)",
            "TypedDict state"
        ],
        "refinement_pattern": "TIME-TRAVEL",
        "refinement_description": """
**How LangGraph Handles Gap-Driven Refinement:**

LangGraph uses its unique **Checkpointer** to enable "time-travel" - the ability to rewind to a previous state and replay with new context.

**The Flow:**
1. **Checkpoint Saved** - Before lesson generation, state is saved as `pre_lessons`
2. **Gap Assessment** - Student simulation identifies gaps in the course
3. **Conditional Edge** - `should_refine_after_gaps()` checks if refinement needed
4. **Time-Travel** - If gaps exist, we can load the `pre_lessons` checkpoint
5. **Replay with Context** - Lessons regenerated with gap insights injected into state

**Code Pattern:**
```python
# Save checkpoint before lessons
checkpointer.save("course_gen", state, "pre_lessons")

# After gap assessment, check conditional edge
if should_refine_after_gaps(state) == "refine":
    # Time-travel: load prior state
    prior_state = checkpointer.load("course_gen", "pre_lessons")
    # Inject gap context and replay
    prior_state["gap_context"] = gap_insights
    # Re-run lesson generation nodes
```

**Why This Pattern?**
- **State Immutability**: Each checkpoint is a frozen point in time
- **Reproducibility**: Can replay any part of the workflow
- **Debugging**: Inspect state at any checkpoint
- **Recovery**: Resume from failures without starting over
        """,
        "architecture_diagram": """
```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Flow                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  START → research → syllabus → quality_loop              │
│                                    │                     │
│                         ┌──────────┴──────────┐          │
│                         ▼                     ▼          │
│                    [CHECKPOINT]           [continue]     │
│                    "pre_lessons"              │          │
│                         │                     │          │
│                         ▼                     ▼          │
│                  lesson_loop ──────────► gap_assess      │
│                                              │           │
│                         ┌────────────────────┤           │
│                         ▼                    ▼           │
│                 should_refine?          [complete]       │
│                    │    │                    │           │
│              refine│    │no                  │           │
│                    ▼    │                    ▼           │
│           [TIME-TRAVEL] └─────────────────► END         │
│           load checkpoint                                │
│           inject gap_context                             │
│           replay lessons                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```
        """
    },
    "OpenAI SDK": {
        "color": "#e8f5e9",
        "icon": "🤝",
        "tagline": "Agent-to-Agent Handoffs",
        "core_patterns": [
            "asyncio.gather() (parallel research)",
            "Structured outputs (quality loop)",
            "Blocking guardrails (approval)",
            "agent.as_tool() (gap assessment)",
            "HANDOFF (gap-driven refinement)"
        ],
        "refinement_pattern": "HANDOFF",
        "refinement_description": """
**How OpenAI SDK Handles Gap-Driven Refinement:**

OpenAI SDK uses the **Handoff** pattern - control passes from one specialized agent to another with structured context.

**The Flow:**
1. **StudentSimulator Agent** - Identifies gaps in the course
2. **Structured Output** - Gaps returned as typed JSON (gaps_found, unclear_concepts, etc.)
3. **Handoff Decision** - `_should_refine()` checks if handoff needed
4. **LessonRefiner Agent** - Receives handoff with full gap context
5. **Refined Output** - Each lesson improved addressing identified gaps

**Code Pattern:**
```python
# Student agent produces structured gap output
gap_data = await Runner.run(student_agent, course_content)

# Check if handoff needed
if _should_refine(gap_data):
    # Build handoff context
    handoff_context = f"HANDOFF from StudentSimulator:\\n{gap_data}"

    # Handoff to refiner agent
    for lesson in lessons:
        refined = await Runner.run(
            lesson_refiner_agent,
            context=handoff_context,
            lesson=lesson
        )
```

**Why This Pattern?**
- **Specialization**: Each agent has focused expertise
- **Clear Contracts**: Structured outputs define handoff interface
- **Composability**: Agents can be chained in different orders
- **Traceability**: Each handoff is explicit and logged
        """,
        "architecture_diagram": """
```
┌─────────────────────────────────────────────────────────┐
│                  OpenAI SDK Flow                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │TopicExtractor│───►│  Researcher  │───►│  Syllabus  │ │
│  │    Agent     │    │    Agent     │    │   Agent    │ │
│  └──────────────┘    └──────────────┘    └─────┬──────┘ │
│                                                │         │
│                                                ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │QualityEval   │◄───│LessonWriter  │◄───│ Approval   │ │
│  │    Agent     │    │    Agent     │    │ Guardrail  │ │
│  └──────┬───────┘    └──────────────┘    └────────────┘ │
│         │                                               │
│         ▼                                               │
│  ┌──────────────┐         HANDOFF                       │
│  │StudentSimul. │─────────────────────┐                 │
│  │    Agent     │                     │                 │
│  └──────────────┘                     ▼                 │
│         │                      ┌──────────────┐         │
│         │ structured           │LessonRefiner │         │
│         │ gap_data             │    Agent     │         │
│         │                      └──────┬───────┘         │
│         │                             │                 │
│         └─────────────────────────────┘                 │
│                   re-assess                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```
        """
    },
    "Google ADK": {
        "color": "#fff3e0",
        "icon": "🎚️",
        "tagline": "Hierarchical Agents with Escalation",
        "core_patterns": [
            "ParallelAgent (parallel research)",
            "LoopAgent (quality refinement)",
            "AgentTool (gap assessment)",
            "ESCALATION (gap-driven refinement)",
            "output_key (state sharing)",
            "{{template}} substitution"
        ],
        "refinement_pattern": "ESCALATION",
        "refinement_description": """
**How Google ADK Handles Gap-Driven Refinement:**

Google ADK uses the **LoopAgent with Escalation Signal** pattern - an escalation key triggers the refinement loop.

**The Flow:**
1. **AgentTool** - StudentSimulator wrapped as callable tool
2. **Gap Assessment** - Tool invoked, returns gap data
3. **Escalation Check** - If gaps exceed threshold, set `escalation_key`
4. **LoopAgent Triggered** - Escalation activates refinement loop
5. **LessonRefiner Agent** - Iterates through lessons with gap context

**Code Pattern:**
```python
# Wrap student as AgentTool
gap_assessor = AgentTool(
    agent=student_agent,
    name="GapAssessor"
)

# Invoke and check result
gap_data = gap_assessor(session, course_summary)

# Set escalation signal if needed
if total_issues >= threshold:
    session.set("refinement_escalation", True)  # Escalation!

    # LoopAgent refinement triggered
    for lesson in lessons:
        lesson_refiner.run(session, lesson)

    # Clear escalation after handling
    session.set("refinement_escalation", False)
```

**Why This Pattern?**
- **Hierarchical Control**: Parent agents can escalate to specialists
- **Declarative Loops**: LoopAgent handles iteration automatically
- **State Sharing**: SessionState + output_key for inter-agent data
- **Composable**: Agents as tools enables nesting
        """,
        "architecture_diagram": """
```
┌─────────────────────────────────────────────────────────┐
│                  Google ADK Flow                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              SequentialAgent (Root)              │    │
│  ├─────────────────────────────────────────────────┤    │
│  │                                                  │    │
│  │  ┌────────────────────────────────────────────┐ │    │
│  │  │         ParallelAgent (Research)           │ │    │
│  │  │  [Tutorial] [Practices] [Examples]         │ │    │
│  │  └────────────────────────────────────────────┘ │    │
│  │                       │                         │    │
│  │                       ▼                         │    │
│  │  ┌────────────────────────────────────────────┐ │    │
│  │  │    LoopAgent (Quality) max_iterations=3    │ │    │
│  │  │    [Evaluate] ──► [Refine] ──► [Escalate]  │ │    │
│  │  └────────────────────────────────────────────┘ │    │
│  │                       │                         │    │
│  │                       ▼                         │    │
│  │  ┌────────────────────────────────────────────┐ │    │
│  │  │    AgentTool: GapAssessor                  │ │    │
│  │  │    [StudentSimulator] ──► gap_data         │ │    │
│  │  └────────────────────┬───────────────────────┘ │    │
│  │                       │                         │    │
│  │              ESCALATION SIGNAL?                 │    │
│  │                  ╱          ╲                   │    │
│  │                yes           no                 │    │
│  │                 │             │                 │    │
│  │                 ▼             ▼                 │    │
│  │  ┌──────────────────┐   ┌──────────┐          │    │
│  │  │ LoopAgent Refine │   │   END    │          │    │
│  │  │ [LessonRefiner]  │   └──────────┘          │    │
│  │  └──────────────────┘                          │    │
│  │                                                  │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```
        """
    },
    "Orchestral": {
        "color": "#f3e5f5",
        "icon": "🪝",
        "tagline": "Hooks + Subagents for Workflow Control",
        "core_patterns": [
            "Provider-agnostic design",
            "CheapLLM (auto cost optimization)",
            "Subagent pattern (gap assessment)",
            "HOOK+SUBAGENT (gap-driven refinement)",
            "Context cost tracking",
            "Synchronous execution"
        ],
        "refinement_pattern": "HOOK+SUBAGENT",
        "refinement_description": """
**How Orchestral Handles Gap-Driven Refinement:**

Orchestral uses the **Hook + Subagent** pattern - hooks intercept workflow results and trigger specialized subagents.

**The Flow:**
1. **StudentSimulatorSubagent** - Encapsulates gap assessment logic
2. **RefinementHook** - Post-execution hook inspects gap results
3. **Hook Decision** - `should_refine()` returns approval + interrupt signal
4. **LessonRefinerSubagent** - Triggered by hook to refine lessons
5. **Synchronous Execution** - Deterministic, debuggable flow

**Code Pattern:**
```python
# Subagent for gap assessment
class StudentSimulatorSubagent:
    def __call__(self, course_content, context):
        student = Agent(llm=CheapLLM(), ...)
        return student.run(course_content, phase="gap_assessment")

# Hook for workflow control
class RefinementHook:
    def should_refine(self, gap_data) -> ToolHookResult:
        if issues >= threshold:
            return ToolHookResult(
                approved=True,
                should_interrupt=True  # Trigger refinement
            )

# In workflow
gap_data = student_subagent(course, context)
hook_result = refinement_hook.should_refine(gap_data)

if hook_result.approved and hook_result.should_interrupt:
    # Hook triggered refinement
    for lesson in lessons:
        refined = lesson_refiner_subagent(lesson, gap_context, context)
```

**Why This Pattern?**
- **Encapsulation**: Subagents are reusable, self-contained units
- **Workflow Control**: Hooks intercept and modify execution flow
- **Cost Tracking**: Context tracks costs per phase automatically
- **Synchronous**: Easy to debug, deterministic behavior
        """,
        "architecture_diagram": """
```
┌─────────────────────────────────────────────────────────┐
│                  Orchestral Flow                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │                  Context                         │    │
│  │  [session_id] [total_cost] [costs_by_phase]     │    │
│  └─────────────────────────────────────────────────┘    │
│                         │                                │
│    ┌────────────────────┼────────────────────┐          │
│    ▼                    ▼                    ▼          │
│  ┌─────────┐      ┌───────────┐       ┌───────────┐    │
│  │CheapLLM │      │BalancedLLM│       │ QualityLLM│    │
│  │Research │      │ Syllabus  │       │   (opt)   │    │
│  └─────────┘      └───────────┘       └───────────┘    │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │             ApprovalHook                         │    │
│  │  check() ──► [approved: true/false]             │    │
│  └─────────────────────────────────────────────────┘    │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │         StudentSimulatorSubagent                 │    │
│  │  __call__(course, context) ──► gap_data         │    │
│  └─────────────────────────────────────────────────┘    │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │             RefinementHook                       │    │
│  │  should_refine(gap_data)                         │    │
│  │     │                                            │    │
│  │     ├─► approved=True, interrupt=True            │    │
│  │     │        │                                   │    │
│  │     │        ▼                                   │    │
│  │     │   ┌─────────────────────────────────┐     │    │
│  │     │   │    LessonRefinerSubagent        │     │    │
│  │     │   │  __call__(lesson, gap, context) │     │    │
│  │     │   └─────────────────────────────────┘     │    │
│  │     │                                            │    │
│  │     └─► approved=False ──► END                  │    │
│  │                                                  │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```
        """
    }
}


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    # API Key status
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    jina_key = os.getenv("JINA_API_KEY", "")

    st.subheader("API Keys")
    if openrouter_key and openrouter_key != "your_openrouter_key_here":
        st.success("✅ OpenRouter configured")
    else:
        st.warning("⚠️ OpenRouter key not set")
        openrouter_key = st.text_input("OpenRouter API Key", type="password")
        if openrouter_key:
            os.environ["OPENROUTER_API_KEY"] = openrouter_key

    if jina_key and jina_key != "your_jina_key_here":
        st.success("✅ Jina configured")
    else:
        st.info("ℹ️ Jina optional (free tier)")

    st.divider()

    # Models info - Updated for 2026
    st.subheader("Models (2026)")
    st.markdown("""
    - **Cheap:** DeepSeek V3.2
    - **Balanced:** Gemini 3 Flash
    - **Provider:** OpenRouter
    """)

    st.divider()

    # Framework patterns legend
    st.subheader("Framework Patterns")
    with st.expander("LangGraph"):
        st.markdown("Send API, interrupt(), Conditional edges, Checkpointer")
    with st.expander("OpenAI SDK"):
        st.markdown("asyncio.gather(), Structured outputs, Agent-as-tool, Guardrails")
    with st.expander("Google ADK"):
        st.markdown("ParallelAgent, LoopAgent, AgentTool, SessionState")
    with st.expander("Orchestral"):
        st.markdown("CheapLLM, Context tracking, Hooks, Subagents")


# ============================================================================
# Main Content
# ============================================================================

st.title("🤖 Agent Framework Comparison")

# Navigation
if st.session_state.view_mode == "compare" and st.session_state.results:
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Run List"):
            st.session_state.view_mode = "list"
            st.session_state.results = {}
            st.rerun()
    with col2:
        if st.button("📖 View Pattern Details"):
            st.session_state.view_mode = "patterns"
            st.rerun()

if st.session_state.view_mode == "patterns":
    if st.button("← Back to Comparison"):
        st.session_state.view_mode = "compare"
        st.rerun()

# ============================================================================
# VIEW: Run List
# ============================================================================

if st.session_state.view_mode == "list":
    st.markdown("Compare **LangGraph**, **OpenAI SDK**, **Google ADK**, and **Orchestral AI** on course generation.")

    # New run section
    st.subheader("🚀 Generate New Course")

    col1, col2 = st.columns([4, 1])
    with col1:
        new_prompt = st.text_input(
            "Course Topic",
            placeholder="e.g., Create a course on Machine Learning Fundamentals",
            label_visibility="collapsed"
        )
    with col2:
        generate_btn = st.button("Generate", type="primary", use_container_width=True)

    # Framework selection
    fw_cols = st.columns(4)
    with fw_cols[0]:
        run_langgraph = st.checkbox("LangGraph", value=True)
    with fw_cols[1]:
        run_openai = st.checkbox("OpenAI SDK", value=True)
    with fw_cols[2]:
        run_google = st.checkbox("Google ADK", value=True)
    with fw_cols[3]:
        run_orchestral = st.checkbox("Orchestral", value=True)

    if generate_btn and new_prompt:
        if not os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") == "your_openrouter_key_here":
            st.error("⚠️ Please configure your OpenRouter API key")
        else:
            # Build frameworks dict
            frameworks = {}
            if run_langgraph:
                frameworks["LangGraph"] = langgraph_generate
            if run_openai:
                frameworks["OpenAI SDK"] = openai_generate
            if run_google:
                frameworks["Google ADK"] = google_generate
            if run_orchestral:
                frameworks["Orchestral"] = orchestral_generate

            if not frameworks:
                st.warning("Please select at least one framework")
            else:
                with st.spinner(f"Running {len(frameworks)} frameworks in parallel... This may take a few minutes."):
                    results = run_all_frameworks_ui(new_prompt, frameworks)

                # Save results
                from cli_runner import save_results as cli_save, framework_result_to_dict
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = RESULTS_DIR / timestamp
                output_dir.mkdir(parents=True, exist_ok=True)

                combined = {
                    "prompt": new_prompt,
                    "timestamp": datetime.now().isoformat(),
                    "frameworks": {name: framework_result_to_dict(result) for name, result in results.items()}
                }

                with open(output_dir / "results.json", "w") as f:
                    json.dump(combined, f, indent=2, default=str)

                # Update latest symlink
                latest_link = RESULTS_DIR / "latest"
                if latest_link.is_symlink():
                    latest_link.unlink()
                latest_link.symlink_to(timestamp)

                # Load into session and switch to compare view
                st.session_state.results = results
                st.session_state.current_prompt = new_prompt
                st.session_state.view_mode = "compare"
                st.success("✅ Generation complete!")
                st.rerun()

    st.divider()

    # Saved runs list
    st.subheader("📚 Saved Runs")

    saved_runs = get_saved_runs()

    if not saved_runs:
        st.info("No saved runs yet. Generate a new course or run from CLI.")
        st.code("python cli_runner.py \"Create a course on Python\"", language="bash")
    else:
        for run in saved_runs:
            col1, col2, col3 = st.columns([5, 2, 1])

            with col1:
                # Truncate long prompts
                prompt_display = run["prompt"][:80] + "..." if len(run["prompt"]) > 80 else run["prompt"]
                st.markdown(f"**{prompt_display}**")
                st.caption(f"🕐 {run['timestamp'][:19].replace('T', ' ')}")

            with col2:
                # Framework badges
                fw_str = ", ".join(run["frameworks"])
                st.markdown(f"✅ {run['successful']}/{run['total']} frameworks")
                st.caption(fw_str)

            with col3:
                if st.button("View →", key=f"view_{run['folder']}"):
                    try:
                        loaded = load_results(run["path"])
                        st.session_state.results = loaded['results']
                        st.session_state.current_prompt = loaded['prompt']
                        st.session_state.view_mode = "compare"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load: {e}")

            st.divider()


# ============================================================================
# VIEW: Compare Results
# ============================================================================

elif st.session_state.view_mode == "compare" and st.session_state.results:
    results = st.session_state.results
    prompt = st.session_state.current_prompt

    st.markdown(f"### 📝 {prompt}")
    st.divider()

    # Framework order and colors
    framework_order = ["LangGraph", "OpenAI SDK", "Google ADK", "Orchestral"]
    colors = {"LangGraph": "langgraph", "OpenAI SDK": "openai", "Google ADK": "google", "Orchestral": "orchestral"}

    # ═══════════════════════════════════════════════════════════════════
    # Overview Metrics
    # ═══════════════════════════════════════════════════════════════════
    st.subheader("📊 Overview")

    metric_cols = st.columns(4)
    for i, name in enumerate(framework_order):
        if name in results:
            result = results[name]
            with metric_cols[i]:
                st.markdown(f'<div class="framework-header {colors[name]}">{name}</div>', unsafe_allow_html=True)

                if result.success:
                    st.metric("Status", "✅ Success")
                else:
                    st.metric("Status", "❌ Failed")

                if result.metrics:
                    duration = result.metrics.duration_seconds
                    st.metric("Duration", f"{duration:.1f}s" if duration else "--")
                    st.metric("Tokens", f"{result.metrics.total_tokens:,}" if result.metrics.total_tokens else "--")

                if result.enhanced_course:
                    ec = result.enhanced_course
                    if ec.quality_score:
                        score = ec.quality_score.score
                        st.metric("Quality", f"{score:.2f}", delta=f"iter {ec.quality_score.iteration}")
                    if ec.cost_breakdown:
                        st.metric("Total Cost", format_cost(ec.cost_breakdown.total_cost))
        else:
            with metric_cols[i]:
                st.markdown(f'<div class="framework-header {colors[name]}">{name}</div>', unsafe_allow_html=True)
                st.metric("Status", "⏭️ Skipped")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # Quality Assessment
    # ═══════════════════════════════════════════════════════════════════
    st.subheader("🎯 Quality Assessment")

    quality_cols = st.columns(4)
    for i, name in enumerate(framework_order):
        if name in results:
            result = results[name]
            with quality_cols[i]:
                st.markdown(f"**{name}**")

                if result.enhanced_course and result.enhanced_course.quality_score:
                    qs = result.enhanced_course.quality_score
                    quality_class = get_quality_class(qs.score)

                    st.markdown(f'<div class="quality-score {quality_class}">{qs.score:.2f}</div>', unsafe_allow_html=True)
                    st.caption(f"Iteration {qs.iteration}")

                    if qs.feedback:
                        st.markdown(f"*{qs.feedback[:100]}...*" if len(qs.feedback) > 100 else f"*{qs.feedback}*")

                    if qs.issues:
                        with st.expander(f"Issues ({len(qs.issues)})"):
                            for issue in qs.issues[:5]:
                                st.markdown(f"- {issue}")
                else:
                    st.markdown("*No quality data*")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # Gap Assessment
    # ═══════════════════════════════════════════════════════════════════
    st.subheader("📋 Gap Assessment (Student Simulation)")

    gap_cols = st.columns(4)
    for i, name in enumerate(framework_order):
        if name in results:
            result = results[name]
            with gap_cols[i]:
                st.markdown(f"**{name}**")

                if result.enhanced_course and result.enhanced_course.gap_assessment:
                    ga = result.enhanced_course.gap_assessment

                    if ga.ready_for_publication:
                        st.success("✅ Ready for Publication")
                    else:
                        st.warning("⚠️ Needs Improvement")

                    if ga.gaps_found:
                        with st.expander(f"Gaps ({len(ga.gaps_found)})"):
                            for gap in ga.gaps_found[:5]:
                                st.markdown(f"- {gap}")

                    if ga.missing_prerequisites:
                        with st.expander(f"Missing Prerequisites ({len(ga.missing_prerequisites)})"):
                            for prereq in ga.missing_prerequisites[:5]:
                                st.markdown(f"- {prereq}")

                    if ga.recommendations:
                        with st.expander(f"Recommendations ({len(ga.recommendations)})"):
                            for rec in ga.recommendations[:5]:
                                st.markdown(f"- {rec}")
                else:
                    st.markdown("*No gap assessment*")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # Cost Breakdown
    # ═══════════════════════════════════════════════════════════════════
    st.subheader("💰 Cost Breakdown")

    cost_cols = st.columns(4)
    for i, name in enumerate(framework_order):
        if name in results:
            result = results[name]
            with cost_cols[i]:
                st.markdown(f"**{name}**")

                if result.enhanced_course and result.enhanced_course.cost_breakdown:
                    cb = result.enhanced_course.cost_breakdown

                    st.markdown(f"Research: {format_cost(cb.research_cost)}")
                    st.markdown(f"Syllabus: {format_cost(cb.syllabus_cost)}")
                    st.markdown(f"Quality: {format_cost(cb.quality_loop_cost)}")
                    st.markdown(f"Lessons: {format_cost(cb.lesson_generation_cost)}")
                    st.markdown(f"Gap Assess: {format_cost(cb.gap_assessment_cost)}")
                    # NEW: Gap refinement cost
                    refinement_cost = getattr(cb, 'gap_refinement_cost', 0) or 0
                    if refinement_cost > 0:
                        st.markdown(f"**Refinement: {format_cost(refinement_cost)}**")
                    st.markdown(f"**Total: {format_cost(cb.total_cost)}**")
                    st.caption(f"Tokens: {cb.total_tokens:,}")
                else:
                    st.markdown("*No cost data*")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # Console Output
    # ═══════════════════════════════════════════════════════════════════
    st.subheader("📟 Console Output")

    console_cols = st.columns(4)
    for i, name in enumerate(framework_order):
        if name in results:
            result = results[name]
            with console_cols[i]:
                st.markdown(f"**{name}**")
                console_text = "\n".join(result.console_output[-20:]) if result.console_output else "No output"
                st.code(console_text, language="text")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # Generated Courses (Tabbed View)
    # ═══════════════════════════════════════════════════════════════════
    st.subheader("📚 Generated Courses")

    available_frameworks = [n for n in framework_order if n in results and results[n].success]

    if available_frameworks:
        tabs = st.tabs(available_frameworks)

        for tab, name in zip(tabs, available_frameworks):
            result = results[name]
            with tab:
                if result.course:
                    course = result.course

                    # Course overview
                    st.markdown(f"### {course.syllabus.course_title}")
                    st.markdown(f"**Objective:** {course.syllabus.course_objective}")
                    st.markdown(f"**Audience:** {course.syllabus.target_audience} | **Difficulty:** {course.syllabus.difficulty_level}")

                    # Lessons
                    st.markdown("### Lessons")
                    for lesson in course.syllabus.lessons:
                        with st.expander(f"Lesson {lesson.lesson_number}: {lesson.title}"):
                            if lesson.objectives:
                                st.markdown("**Objectives:**")
                                for obj in lesson.objectives:
                                    st.markdown(f"- {obj}")

                            if lesson.content_outline:
                                st.markdown("**Content:**")
                                for item in lesson.content_outline:
                                    st.markdown(f"- {item}")

                            if lesson.activities:
                                st.markdown("**Activities:**")
                                for act in lesson.activities:
                                    st.markdown(f"- {act}")

                            if lesson.citations:
                                st.markdown("**Citations:**")
                                for cite in lesson.citations:
                                    st.markdown(f"- {cite}")

                    # Raw JSON export
                    with st.expander("📥 Export JSON"):
                        if result.enhanced_course:
                            json_str = json.dumps(result.enhanced_course.model_dump(), indent=2, default=str)
                        else:
                            json_str = json.dumps(course.model_dump(), indent=2, default=str)
                        st.download_button(
                            f"Download {name} JSON",
                            json_str,
                            file_name=f"{name.lower().replace(' ', '_')}_course.json",
                            mime="application/json"
                        )
                else:
                    st.error(f"Error: {result.error or 'Unknown error'}")
    else:
        st.warning("No successful course generations to display.")

    # ═══════════════════════════════════════════════════════════════════
    # Framework Patterns Summary
    # ═══════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🔧 Framework Patterns Used")

    pattern_cols = st.columns(4)
    for i, name in enumerate(framework_order):
        if name in results and name in FRAMEWORK_PATTERNS:
            result = results[name]
            pattern = FRAMEWORK_PATTERNS[name]
            with pattern_cols[i]:
                st.markdown(f"**{pattern['icon']} {name}**")
                st.caption(pattern['tagline'])

                # Get patterns from result metadata if available
                if result.enhanced_course and result.enhanced_course.generation_metadata:
                    meta = result.enhanced_course.generation_metadata
                    patterns = meta.get('patterns_demonstrated', pattern['core_patterns'])
                    refinement_iters = meta.get('refinement_iterations', 0)

                    # Show refinement pattern with badge
                    refinement_pat = pattern['refinement_pattern']
                    st.markdown(f"**Refinement:** `{refinement_pat}`")
                    if refinement_iters > 0:
                        st.success(f"✓ Refined ({refinement_iters} iteration)")
                    else:
                        st.info("No refinement needed")

                    # Show pattern tags
                    with st.expander("All Patterns"):
                        for p in patterns:
                            st.markdown(f"• {p}")
                else:
                    st.markdown(f"**Refinement:** `{pattern['refinement_pattern']}`")

    st.info("💡 Click **View Pattern Details** above to see how each framework implements gap-driven refinement differently.")


# ============================================================================
# VIEW: Patterns Detail
# ============================================================================

elif st.session_state.view_mode == "patterns":
    st.header("📖 Framework Pattern Deep Dive")
    st.markdown("""
    This view explains how each framework achieves **gap-driven refinement** - the same goal implemented
    using each framework's unique architectural patterns. Understanding these patterns helps you choose
    the right framework for your use case.
    """)

    st.divider()

    # Overview comparison table
    st.subheader("📊 Pattern Comparison")

    comparison_data = {
        "Framework": ["LangGraph", "OpenAI SDK", "Google ADK", "Orchestral"],
        "Refinement Pattern": ["TIME-TRAVEL", "HANDOFF", "ESCALATION", "HOOK+SUBAGENT"],
        "Key Mechanism": [
            "Checkpointer rewind + replay",
            "Agent-to-agent with context",
            "LoopAgent with escalation_key",
            "Hook intercept + subagent"
        ],
        "State Management": [
            "TypedDict + Checkpointer",
            "Context object",
            "SessionState + output_key",
            "Context + costs_by_phase"
        ],
        "Best For": [
            "Complex stateful workflows",
            "Agent composition chains",
            "Hierarchical agent teams",
            "Cost-optimized pipelines"
        ]
    }

    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # Detailed framework sections
    for name in ["LangGraph", "OpenAI SDK", "Google ADK", "Orchestral"]:
        pattern = FRAMEWORK_PATTERNS[name]

        st.markdown(f"## {pattern['icon']} {name}")
        st.markdown(f"*{pattern['tagline']}*")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### How It Works")
            st.markdown(pattern['refinement_description'])

        with col2:
            st.markdown("### Architecture")
            st.code(pattern['architecture_diagram'].strip().strip('`'), language="text")

        st.divider()

    # Summary
    st.subheader("🎯 Choosing the Right Pattern")
    st.markdown("""
    | Use Case | Recommended Framework | Why |
    |----------|----------------------|-----|
    | **Need to replay/resume workflows** | LangGraph | Checkpointer enables time-travel and recovery |
    | **Composing specialized agents** | OpenAI SDK | Clean handoff contracts between agents |
    | **Hierarchical agent orchestration** | Google ADK | LoopAgent/ParallelAgent/SequentialAgent |
    | **Cost-sensitive applications** | Orchestral | Built-in cost tracking and CheapLLM auto-selection |
    | **Debugging agent behavior** | Orchestral | Synchronous execution is deterministic |
    | **Production at scale** | LangGraph or OpenAI SDK | Battle-tested with observability |
    """)


# ============================================================================
# Footer
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em;">
    <strong>Agent Framework Comparison Tool (2026)</strong><br/>
    Using OpenRouter | Models: DeepSeek V3.2 (cheap) + Gemini 3 Flash (balanced)<br/>
    Enhanced with: Quality Loops, Gap Assessment, Cost Tracking, Double-Prompting
</div>
""", unsafe_allow_html=True)
