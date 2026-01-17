#!/usr/bin/env python3
"""
CLI Runner for Agent Framework Comparison

Runs all 4 frameworks and saves results to JSON files that can be loaded by the Streamlit UI.

Usage:
    python cli_runner.py "Create a course on Python for Beginners"
    python cli_runner.py --topic "Machine Learning Fundamentals" --frameworks langgraph,openai
    python cli_runner.py --load results/latest  # Load results in UI

Output:
    - results/<timestamp>/results.json - Combined results for all frameworks
    - results/<timestamp>/<framework>.json - Individual framework results
    - results/latest -> symlink to most recent run
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from common.models import FrameworkResult, GenerationMetrics, EnhancedCoursePackage

# ============================================================================
# Framework Imports
# ============================================================================

FRAMEWORK_GENERATORS = {}

def load_frameworks():
    """Lazily load framework generators."""
    global FRAMEWORK_GENERATORS

    if FRAMEWORK_GENERATORS:
        return FRAMEWORK_GENERATORS

    try:
        from langgraph_impl.agent import generate_course as langgraph_generate
        FRAMEWORK_GENERATORS["langgraph"] = ("LangGraph", langgraph_generate)
    except Exception as e:
        print(f"⚠️  Failed to load LangGraph: {e}")

    try:
        from openai_sdk_impl.agent import generate_course as openai_generate
        FRAMEWORK_GENERATORS["openai"] = ("OpenAI SDK", openai_generate)
    except Exception as e:
        print(f"⚠️  Failed to load OpenAI SDK: {e}")

    try:
        from google_adk_impl.agent import generate_course as google_generate
        FRAMEWORK_GENERATORS["google"] = ("Google ADK", google_generate)
    except Exception as e:
        print(f"⚠️  Failed to load Google ADK: {e}")

    try:
        from orchestral_impl.agent import generate_course as orchestral_generate
        FRAMEWORK_GENERATORS["orchestral"] = ("Orchestral", orchestral_generate)
    except Exception as e:
        print(f"⚠️  Failed to load Orchestral: {e}")

    return FRAMEWORK_GENERATORS


# ============================================================================
# Result Serialization (UI-Compatible)
# ============================================================================

def framework_result_to_dict(result: FrameworkResult) -> dict:
    """Convert FrameworkResult to JSON-serializable dict matching UI expectations."""
    return {
        "framework": result.framework,
        "success": result.success,
        "error": result.error,
        "console_output": result.console_output,
        "course": result.course.model_dump() if result.course else None,
        "enhanced_course": result.enhanced_course.model_dump() if result.enhanced_course else None,
        "metrics": {
            "framework": result.metrics.framework,
            "start_time": result.metrics.start_time.isoformat() if result.metrics and result.metrics.start_time else None,
            "end_time": result.metrics.end_time.isoformat() if result.metrics and result.metrics.end_time else None,
            "total_tokens": result.metrics.total_tokens if result.metrics else 0,
            "prompt_tokens": result.metrics.prompt_tokens if result.metrics else 0,
            "completion_tokens": result.metrics.completion_tokens if result.metrics else 0,
            "api_calls": result.metrics.api_calls if result.metrics else 0,
            "jina_calls": result.metrics.jina_calls if result.metrics else 0,
            "errors": result.metrics.errors if result.metrics else [],
            "duration_seconds": result.metrics.duration_seconds if result.metrics else 0,
        } if result.metrics else None
    }


def dict_to_framework_result(data: dict) -> FrameworkResult:
    """Convert dict back to FrameworkResult for UI loading."""
    from common.models import (
        FrameworkResult, GenerationMetrics, CoursePackage,
        EnhancedCoursePackage, Syllabus, LessonPlan,
        QualityScore, GapAssessment, CostBreakdown
    )

    # Reconstruct metrics
    metrics = None
    if data.get("metrics"):
        m = data["metrics"]
        metrics = GenerationMetrics(
            framework=m.get("framework", ""),
            total_tokens=m.get("total_tokens", 0),
            prompt_tokens=m.get("prompt_tokens", 0),
            completion_tokens=m.get("completion_tokens", 0),
            api_calls=m.get("api_calls", 0),
            jina_calls=m.get("jina_calls", 0),
            errors=m.get("errors", [])
        )
        if m.get("start_time"):
            metrics.start_time = datetime.fromisoformat(m["start_time"])
        if m.get("end_time"):
            metrics.end_time = datetime.fromisoformat(m["end_time"])

    # Reconstruct course
    course = None
    if data.get("course"):
        c = data["course"]
        lessons = [LessonPlan(**l) for l in c.get("syllabus", {}).get("lessons", [])]
        syllabus = Syllabus(
            course_title=c["syllabus"]["course_title"],
            course_objective=c["syllabus"]["course_objective"],
            target_audience=c["syllabus"].get("target_audience", "General learners"),
            difficulty_level=c["syllabus"].get("difficulty_level", "Intermediate"),
            lessons=lessons
        )
        course = CoursePackage(
            syllabus=syllabus,
            research_sources=c.get("research_sources", []),
            generation_metadata=c.get("generation_metadata", {})
        )

    # Reconstruct enhanced_course
    enhanced_course = None
    if data.get("enhanced_course"):
        ec = data["enhanced_course"]
        lessons = [LessonPlan(**l) for l in ec.get("syllabus", {}).get("lessons", [])]
        syllabus = Syllabus(
            course_title=ec["syllabus"]["course_title"],
            course_objective=ec["syllabus"]["course_objective"],
            target_audience=ec["syllabus"].get("target_audience", "General learners"),
            difficulty_level=ec["syllabus"].get("difficulty_level", "Intermediate"),
            lessons=lessons
        )

        quality_score = None
        if ec.get("quality_score"):
            quality_score = QualityScore(**ec["quality_score"])

        gap_assessment = None
        if ec.get("gap_assessment"):
            gap_assessment = GapAssessment(**ec["gap_assessment"])

        cost_breakdown = None
        if ec.get("cost_breakdown"):
            cost_breakdown = CostBreakdown(**ec["cost_breakdown"])

        enhanced_course = EnhancedCoursePackage(
            syllabus=syllabus,
            quality_score=quality_score,
            gap_assessment=gap_assessment,
            cost_breakdown=cost_breakdown,
            research_sources=ec.get("research_sources", []),
            generation_metadata=ec.get("generation_metadata", {})
        )

    return FrameworkResult(
        framework=data["framework"],
        success=data["success"],
        error=data.get("error"),
        console_output=data.get("console_output", []),
        course=course,
        enhanced_course=enhanced_course,
        metrics=metrics
    )


# ============================================================================
# Runner Functions
# ============================================================================

def run_single_framework(name: str, display_name: str, generator_func, prompt: str) -> FrameworkResult:
    """Run a single framework and return result."""
    print(f"\n{'='*60}")
    print(f"🚀 Starting {display_name}...")
    print(f"{'='*60}")

    start_time = datetime.now()

    try:
        result = generator_func(prompt)

        # Log success info
        if result.success:
            print(f"✅ {display_name} completed successfully!")
            if result.enhanced_course:
                ec = result.enhanced_course
                if ec.quality_score:
                    print(f"   Quality: {ec.quality_score.score:.2f}")
                if ec.cost_breakdown:
                    print(f"   Cost: ${ec.cost_breakdown.total_cost:.4f}")
                print(f"   Lessons: {len(ec.syllabus.lessons)}")
        else:
            print(f"❌ {display_name} failed: {result.error}")

        return result

    except Exception as e:
        print(f"❌ {display_name} crashed: {e}")
        traceback.print_exc()

        return FrameworkResult(
            framework=display_name,
            success=False,
            error=str(e),
            metrics=GenerationMetrics(
                framework=display_name,
                start_time=start_time,
                end_time=datetime.now(),
                errors=[str(e)]
            ),
            console_output=[f"Error: {e}", traceback.format_exc()]
        )


def run_all_frameworks(prompt: str, framework_keys: list = None, parallel: bool = True) -> dict:
    """Run selected frameworks and return results dict."""
    generators = load_frameworks()

    if framework_keys is None:
        framework_keys = list(generators.keys())

    # Filter to requested frameworks
    to_run = {k: v for k, v in generators.items() if k in framework_keys}

    if not to_run:
        print("❌ No frameworks selected or available")
        return {}

    print(f"\n🎯 Running {len(to_run)} frameworks: {', '.join(to_run.keys())}")
    print(f"📝 Prompt: {prompt}")

    results = {}

    if parallel and len(to_run) > 1:
        print(f"\n⚡ Running in parallel...")
        with ThreadPoolExecutor(max_workers=len(to_run)) as executor:
            futures = {
                executor.submit(run_single_framework, key, name, func, prompt): name
                for key, (name, func) in to_run.items()
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results[name] = result
                except Exception as e:
                    print(f"❌ {name} failed with exception: {e}")
                    results[name] = FrameworkResult(
                        framework=name,
                        success=False,
                        error=str(e),
                        metrics=GenerationMetrics(framework=name),
                        console_output=[f"Error: {e}"]
                    )
    else:
        print(f"\n🔄 Running sequentially...")
        for key, (name, func) in to_run.items():
            results[name] = run_single_framework(key, name, func, prompt)

    return results


def save_results(results: dict, prompt: str, output_dir: Path = None) -> Path:
    """Save results to JSON files."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined results
    combined = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "frameworks": {name: framework_result_to_dict(result) for name, result in results.items()}
    }

    combined_path = output_dir / "results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n💾 Saved combined results to: {combined_path}")

    # Save individual framework results
    for name, result in results.items():
        framework_path = output_dir / f"{name.lower().replace(' ', '_')}.json"
        with open(framework_path, "w") as f:
            json.dump(framework_result_to_dict(result), f, indent=2, default=str)
        print(f"   Saved {name} to: {framework_path}")

    # Update 'latest' symlink
    latest_link = PROJECT_ROOT / "results" / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        import shutil
        shutil.rmtree(latest_link)
    latest_link.symlink_to(output_dir.name)
    print(f"   Updated 'latest' symlink")

    return output_dir


def load_results(results_path: Path) -> dict:
    """Load results from JSON file."""
    if results_path.is_dir():
        results_path = results_path / "results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    # Convert to FrameworkResult objects
    results = {}
    for name, result_data in data.get("frameworks", {}).items():
        results[name] = dict_to_framework_result(result_data)

    return {
        "prompt": data.get("prompt", ""),
        "timestamp": data.get("timestamp", ""),
        "results": results
    }


# ============================================================================
# Summary Display
# ============================================================================

def print_summary(results: dict):
    """Print a summary table of results."""
    print(f"\n{'='*80}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Framework':<15} {'Status':<10} {'Quality':<10} {'Cost':<12} {'Tokens':<12} {'Duration':<10}")
    print("-" * 80)

    for name, result in results.items():
        status = "✅" if result.success else "❌"

        quality = "--"
        cost = "--"
        tokens = "--"
        duration = "--"

        if result.enhanced_course:
            ec = result.enhanced_course
            if ec.quality_score:
                quality = f"{ec.quality_score.score:.2f}"
            if ec.cost_breakdown:
                cost = f"${ec.cost_breakdown.total_cost:.4f}"
                tokens = f"{ec.cost_breakdown.total_tokens:,}"

        if result.metrics and result.metrics.duration_seconds:
            duration = f"{result.metrics.duration_seconds:.1f}s"

        print(f"{name:<15} {status:<10} {quality:<10} {cost:<12} {tokens:<12} {duration:<10}")

    print(f"\n{'='*80}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Agent Framework Comparison from CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli_runner.py "Create a course on Python for Beginners"
    python cli_runner.py --topic "Machine Learning" --frameworks langgraph,openai
    python cli_runner.py --topic "Data Science" --sequential
    python cli_runner.py --load results/latest

The results can be loaded in the Streamlit UI for visualization.
        """
    )

    parser.add_argument("topic", nargs="?", help="Course topic (prompt)")
    parser.add_argument("--topic", "-t", dest="topic_flag", help="Course topic (alternative)")
    parser.add_argument(
        "--frameworks", "-f",
        help="Comma-separated frameworks to run (langgraph,openai,google,orchestral)",
        default="langgraph,openai,google,orchestral"
    )
    parser.add_argument("--sequential", "-s", action="store_true", help="Run sequentially instead of parallel")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--load", "-l", help="Load and display existing results")
    parser.add_argument("--list", action="store_true", help="List available frameworks")

    args = parser.parse_args()

    # List frameworks
    if args.list:
        generators = load_frameworks()
        print("\nAvailable frameworks:")
        for key, (name, _) in generators.items():
            print(f"  {key}: {name}")
        return

    # Load existing results
    if args.load:
        load_path = Path(args.load)
        if not load_path.is_absolute():
            load_path = PROJECT_ROOT / load_path

        print(f"\n📂 Loading results from: {load_path}")
        loaded = load_results(load_path)
        print(f"   Prompt: {loaded['prompt']}")
        print(f"   Timestamp: {loaded['timestamp']}")
        print_summary(loaded['results'])

        print(f"\n💡 To view in UI, run:")
        print(f"   streamlit run streamlit_app/app.py")
        print(f"   Then use 'Load Results' with path: {load_path}")
        return

    # Get topic
    topic = args.topic or args.topic_flag
    if not topic:
        parser.print_help()
        print("\n❌ Error: Please provide a course topic")
        sys.exit(1)

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your_openrouter_key_here":
        print("❌ Error: OPENROUTER_API_KEY not configured in .env")
        sys.exit(1)

    # Parse frameworks
    framework_keys = [f.strip().lower() for f in args.frameworks.split(",")]

    # Run frameworks
    results = run_all_frameworks(
        prompt=topic,
        framework_keys=framework_keys,
        parallel=not args.sequential
    )

    # Print summary
    print_summary(results)

    # Save results
    output_dir = Path(args.output) if args.output else None
    save_path = save_results(results, topic, output_dir)

    print(f"\n💡 To view in Streamlit UI:")
    print(f"   streamlit run streamlit_app/app.py")
    print(f"   Then load results from: {save_path}")


if __name__ == "__main__":
    main()
