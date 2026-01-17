"""Jina AI API integration for web research."""
import os
import httpx
from urllib.parse import quote
from typing import Optional

# Get API key from environment
JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def jina_search(query: str, api_key: Optional[str] = None) -> dict:
    """
    Search the web using Jina Search API.

    Args:
        query: The search query
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Dictionary with search results and metadata
    """
    key = api_key or JINA_API_KEY
    encoded_query = quote(query)

    headers = {"Accept": "text/plain"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        response = httpx.get(
            f"https://s.jina.ai/?q={encoded_query}",
            headers=headers,
            timeout=30.0
        )
        return {
            "success": True,
            "results": response.text,
            "query": query,
            "status_code": response.status_code
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results": ""
        }


def jina_read(url: str, api_key: Optional[str] = None) -> dict:
    """
    Read and convert a URL to markdown using Jina Reader API.

    Args:
        url: The URL to read
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Dictionary with page content and metadata
    """
    key = api_key or JINA_API_KEY

    headers = {"Accept": "text/plain"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        response = httpx.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=30.0
        )
        return {
            "success": True,
            "content": response.text,
            "url": url,
            "status_code": response.status_code
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "content": ""
        }


async def jina_search_async(query: str, api_key: Optional[str] = None) -> dict:
    """Async version of jina_search."""
    key = api_key or JINA_API_KEY
    encoded_query = quote(query)

    headers = {"Accept": "text/plain"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://s.jina.ai/?q={encoded_query}",
                headers=headers,
                timeout=30.0
            )
            return {
                "success": True,
                "results": response.text,
                "query": query,
                "status_code": response.status_code
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results": ""
        }


async def jina_read_async(url: str, api_key: Optional[str] = None) -> dict:
    """Async version of jina_read."""
    key = api_key or JINA_API_KEY

    headers = {"Accept": "text/plain"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=30.0
            )
            return {
                "success": True,
                "content": response.text,
                "url": url,
                "status_code": response.status_code
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "content": ""
        }
