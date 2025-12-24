"""
LLM utilities for agent system
"""
from typing import Optional
from langchain_openai import ChatOpenAI

# Try to import Google Gemini, but make it optional
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    ChatGoogleGenerativeAI = None

from config import settings


def get_llm(temperature: Optional[float] = None, model: Optional[str] = None):
    """
    Get LLM instance based on configuration
    
    Args:
        temperature: Model temperature (0.0 to 1.0)
        model: Model name (optional override)
        
    Returns:
        LLM instance
    """
    temp = temperature if temperature is not None else settings.temperature
    
    if settings.llm_provider == "openai":
        from openai import OpenAI
        
        model_name = model or settings.openai_model
        base_url = getattr(settings, 'openai_base_url', None)
        
        # For OpenRouter, use custom wrapper that properly sets headers
        if base_url and 'openrouter' in base_url:
            from utils.openrouter_llm import OpenRouterLLM
            
            return OpenRouterLLM(
                model=model_name,
                temperature=temp,
                max_tokens=settings.max_tokens,
                api_key=settings.openai_api_key,
                base_url=base_url
            )
        else:
            return ChatOpenAI(
                model=model_name,
                temperature=temp,
                max_tokens=settings.max_tokens,
                api_key=settings.openai_api_key,
                base_url=base_url if base_url else None
            )
    elif settings.llm_provider == "google":
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Gemini is not available. Install with: pip install langchain-google-genai"
            )
        model_name = model or settings.gemini_model
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temp,
            max_output_tokens=settings.max_tokens,
            google_api_key=settings.google_api_key
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def create_prompt_template(role: str, instructions: str) -> str:
    """
    Create a standardized prompt template
    
    Args:
        role: Agent role
        instructions: Specific instructions for the agent
        
    Returns:
        Formatted prompt template
    """
    return f"""You are a {role} for a Retail Insights Assistant.

{instructions}

Be precise, analytical, and focus on providing actionable insights.
"""
