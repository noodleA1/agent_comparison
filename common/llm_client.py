"""OpenRouter LLM client for all framework implementations.

Includes 2026 double-prompting technique for improved output quality.
Research shows sending the same prompt twice can improve output by up to 76%.
"""
import os
from openai import OpenAI
from typing import Optional, List, Dict, Any

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 2026 Model defaults
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "deepseek/deepseek-v3.2")
BALANCED_MODEL = os.getenv("BALANCED_MODEL", "google/gemini-3-flash-preview")

# Double-prompting configuration (2026 research shows this improves output quality)
DOUBLE_PROMPT_ENABLED = os.getenv("DOUBLE_PROMPT_ENABLED", "true").lower() == "true"


def get_llm_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Get an OpenAI-compatible client configured for OpenRouter.

    Args:
        api_key: Optional API key (uses env var if not provided)

    Returns:
        OpenAI client configured for OpenRouter
    """
    key = api_key or OPENROUTER_API_KEY
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=key,
    )


def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Make a chat completion request via OpenRouter.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model to use (defaults to env var or claude-3.5-sonnet)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        api_key: Optional API key
        **kwargs: Additional parameters to pass to the API

    Returns:
        Dictionary with response content and metadata
    """
    client = get_llm_client(api_key)
    model_name = model or DEFAULT_MODEL

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return {
            "success": True,
            "content": response.choices[0].message.content,
            "model": model_name,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": "",
            "model": model_name,
            "usage": {}
        }


def chat_completion_with_tools(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Make a chat completion request with tool definitions.

    Args:
        messages: List of message dicts
        tools: List of tool definitions in OpenAI format
        model: Model to use
        api_key: Optional API key
        **kwargs: Additional parameters

    Returns:
        Dictionary with response, tool calls, and metadata
    """
    client = get_llm_client(api_key)
    model_name = model or DEFAULT_MODEL

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            **kwargs
        )

        message = response.choices[0].message
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                })

        return {
            "success": True,
            "content": message.content or "",
            "tool_calls": tool_calls,
            "model": model_name,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": "",
            "tool_calls": [],
            "model": model_name,
            "usage": {}
        }


# ============================================================================
# Double-Prompting Technique (2026 Research)
# ============================================================================
# Google Research found that sending the same prompt twice improves output
# quality by up to 76% on non-reasoning tasks. The model reconsiders its
# response, catching errors and providing more refined output.

def double_prompt_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    enabled: Optional[bool] = None,
    strategy: str = "second",  # "second", "synthesis", or "comparison"
    **kwargs
) -> Dict[str, Any]:
    """
    Make a chat completion with double-prompting for improved quality.

    The double-prompting technique sends the same prompt twice and uses
    the second response (which research shows is typically better).

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        api_key: Optional API key
        enabled: Override global DOUBLE_PROMPT_ENABLED setting
        strategy: How to use the two responses:
            - "second": Return only the second response (default, most cost-effective)
            - "synthesis": Ask model to synthesize best parts of both
            - "comparison": Return both for manual comparison
        **kwargs: Additional parameters

    Returns:
        Dictionary with response content and metadata (includes both responses if strategy="comparison")
    """
    use_double = enabled if enabled is not None else DOUBLE_PROMPT_ENABLED

    # First prompt
    response1 = chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        **kwargs
    )

    if not response1["success"]:
        return response1

    if not use_double:
        return response1

    # Second prompt (identical)
    response2 = chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        **kwargs
    )

    if not response2["success"]:
        # Fall back to first response if second fails
        return response1

    # Aggregate usage
    total_usage = {
        "prompt_tokens": response1["usage"].get("prompt_tokens", 0) + response2["usage"].get("prompt_tokens", 0),
        "completion_tokens": response1["usage"].get("completion_tokens", 0) + response2["usage"].get("completion_tokens", 0),
        "total_tokens": response1["usage"].get("total_tokens", 0) + response2["usage"].get("total_tokens", 0),
    }

    if strategy == "second":
        # Return second response (research shows it's typically better)
        return {
            "success": True,
            "content": response2["content"],
            "model": response2["model"],
            "usage": total_usage,
            "double_prompted": True,
            "strategy": strategy
        }

    elif strategy == "comparison":
        # Return both for manual comparison
        return {
            "success": True,
            "content": response2["content"],  # Primary is still second
            "response1": response1["content"],
            "response2": response2["content"],
            "model": response2["model"],
            "usage": total_usage,
            "double_prompted": True,
            "strategy": strategy
        }

    elif strategy == "synthesis":
        # Ask model to synthesize the best parts of both responses
        synthesis_messages = [
            {"role": "system", "content": "You are a quality synthesizer. Given two responses to the same prompt, create the best possible response by combining the strongest elements of both. Focus on accuracy, completeness, and clarity."},
            {"role": "user", "content": f"Response 1:\n{response1['content']}\n\n---\n\nResponse 2:\n{response2['content']}\n\n---\n\nSynthesize the best possible response from these two:"}
        ]

        synthesis = chat_completion(
            messages=synthesis_messages,
            model=model,
            temperature=0.3,  # Lower temp for synthesis
            max_tokens=max_tokens,
            api_key=api_key
        )

        if synthesis["success"]:
            # Add synthesis tokens to total
            total_usage["prompt_tokens"] += synthesis["usage"].get("prompt_tokens", 0)
            total_usage["completion_tokens"] += synthesis["usage"].get("completion_tokens", 0)
            total_usage["total_tokens"] += synthesis["usage"].get("total_tokens", 0)

            return {
                "success": True,
                "content": synthesis["content"],
                "model": synthesis["model"],
                "usage": total_usage,
                "double_prompted": True,
                "strategy": strategy
            }
        else:
            # Fall back to second response if synthesis fails
            return {
                "success": True,
                "content": response2["content"],
                "model": response2["model"],
                "usage": total_usage,
                "double_prompted": True,
                "strategy": "second_fallback"
            }

    # Default to second response
    return {
        "success": True,
        "content": response2["content"],
        "model": response2["model"],
        "usage": total_usage,
        "double_prompted": True,
        "strategy": strategy
    }
