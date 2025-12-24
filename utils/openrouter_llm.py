"""
Custom LangChain-compatible wrapper for OpenRouter API
"""
from typing import Any, List, Optional
from openai import OpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class OpenRouterLLM(BaseChatModel):
    """
    Custom ChatModel wrapper for OpenRouter API that properly sets required headers
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Store parameters
        self._model_name = kwargs.get('model', "mistralai/mistral-7b-instruct:free")
        self._temperature = kwargs.get('temperature', 0.1)
        self._max_tokens = kwargs.get('max_tokens', 2000)
        
        # Create OpenAI client with OpenRouter headers
        self._client = OpenAI(
            base_url=kwargs.get('base_url', 'https://openrouter.ai/api/v1'),
            api_key=kwargs.get('api_key'),
            default_headers={
                "HTTP-Referer": "https://retail-insights.local",
                "X-Title": "Retail Insights Assistant"
            }
        )
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion"""
        
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                role = "user"
            
            openai_messages.append({
                "role": role,
                "content": msg.content
            })
        
        # Call OpenAI API
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stop=stop
        )
        
        # Convert response to LangChain format
        message = AIMessage(content=response.choices[0].message.content)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM"""
        return "openrouter"
    
    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters"""
        return {
            "model_name": self._model_name,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens
        }
