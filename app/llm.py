from functools import lru_cache

from openai import OpenAI

from .config import settings


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    # Any OpenAI-compatible endpoint works (Groq, OpenAI, Gemini's compat endpoint, ...).
    return OpenAI(
        api_key=settings.llm_api_key or "missing",
        base_url=settings.llm_base_url,
        timeout=45.0,
        max_retries=0,
    )


def chat(messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2):
    kwargs = {"model": settings.llm_model, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return _client().chat.completions.create(**kwargs)
