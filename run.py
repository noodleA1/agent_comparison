#!/usr/bin/env python3
"""
Quick test runner for all framework implementations.
Run this to verify everything works before using Streamlit UI.
"""
import sys
import os
from pathlib import Path

# Add project root to path (relative to this file)
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")

    try:
        from common.models import LessonPlan, Syllabus, CoursePackage
        print("  ✅ common.models")
    except Exception as e:
        print(f"  ❌ common.models: {e}")
        return False

    try:
        from langgraph_impl.agent import generate_course as lg_gen
        print("  ✅ langgraph_impl.agent (self-contained)")
    except Exception as e:
        print(f"  ❌ langgraph_impl.agent: {e}")
        return False

    try:
        from openai_sdk_impl.agent import generate_course as oai_gen
        print("  ✅ openai_sdk_impl.agent (self-contained)")
    except Exception as e:
        print(f"  ❌ openai_sdk_impl.agent: {e}")
        return False

    try:
        from google_adk_impl.agent import generate_course as goog_gen
        print("  ✅ google_adk_impl.agent (self-contained)")
    except Exception as e:
        print(f"  ❌ google_adk_impl.agent: {e}")
        return False

    try:
        from orchestral_impl.agent import generate_course as orch_gen
        print("  ✅ orchestral_impl.agent (self-contained)")
    except Exception as e:
        print(f"  ❌ orchestral_impl.agent: {e}")
        return False

    print("\n✅ All imports successful!")
    return True


def check_env():
    """Check environment variables."""
    print("\nChecking environment...")

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    jina_key = os.getenv("JINA_API_KEY", "")

    if openrouter_key and openrouter_key != "your_openrouter_key_here":
        print(f"  ✅ OPENROUTER_API_KEY: {openrouter_key[:10]}...")
    else:
        print("  ⚠️  OPENROUTER_API_KEY not set - LLM calls will fail")
        return False

    if jina_key and jina_key != "your_jina_key_here":
        print(f"  ✅ JINA_API_KEY: {jina_key[:10]}...")
    else:
        print("  ℹ️  JINA_API_KEY not set - using free tier (rate limited)")

    return True


def run_quick_test():
    """Run a quick test with one framework."""
    print("\n" + "="*60)
    print("Running quick smoke test with LangGraph...")
    print("="*60 + "\n")

    from langgraph_impl.agent import generate_course

    result = generate_course("Create a short course on Python basics")

    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Success: {result.success}")

    if result.success and result.course:
        print(f"Course: {result.course.syllabus.course_title}")
        print(f"Lessons: {len(result.course.syllabus.lessons)}")
        for lesson in result.course.syllabus.lessons[:3]:
            print(f"  - Lesson {lesson.lesson_number}: {lesson.title}")
        if len(result.course.syllabus.lessons) > 3:
            print(f"  ... and {len(result.course.syllabus.lessons) - 3} more")
    else:
        print(f"Error: {result.error}")

    if result.metrics:
        print(f"\nMetrics:")
        print(f"  Duration: {result.metrics.duration_seconds:.1f}s")
        print(f"  Tokens: {result.metrics.total_tokens}")
        print(f"  API Calls: {result.metrics.api_calls}")
        print(f"  Jina Calls: {result.metrics.jina_calls}")

    return result.success


if __name__ == "__main__":
    print("="*60)
    print("Agent Framework Comparison - Smoke Test")
    print("="*60 + "\n")

    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Fix errors and retry.")
        sys.exit(1)

    # Check env
    env_ok = check_env()

    if not env_ok:
        print("\n⚠️  Environment not fully configured.")
        print("To run the full test, set OPENROUTER_API_KEY in .env file")
        print("\nTo start Streamlit UI anyway:")
        print("  streamlit run streamlit_app/app.py")
        sys.exit(0)

    # Run quick test
    if run_quick_test():
        print("\n" + "="*60)
        print("✅ SMOKE TEST PASSED!")
        print("="*60)
        print("\nTo run the full comparison UI:")
        print("  streamlit run streamlit_app/app.py")
    else:
        print("\n❌ Smoke test failed. Check errors above.")
        sys.exit(1)
