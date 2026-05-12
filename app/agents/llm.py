from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm():
    model = settings.ACTIVE_LLM_MODEL

    if model == "openai/gpt-5.4":
        return ChatOpenAI(
            model="gpt-5.4",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1,
        )
    if model == "anthropic/claude-sonnet-4-6":
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.1,
        )
    if model == "google/gemini-3.1-pro-preview":
        return ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
        )

    raise ValueError(f"Unknown model: {model}")
