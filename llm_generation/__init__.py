"""
LLM Generation Module
Handles answer generation with multiple backends and hallucination validation
"""
from llm_generation.generation import (
    AnswerGenerator,
    GenerationConfig,
    GenerationResult,
    LLMBackend,
    IntegratedValidator,
    ValidatedAnswer,
    ValidatedAnswerConfig
)
from llm_generation.prompt import (
    PromptType,
    PromptTemplate,
    PromptTemplateLibrary,
    format_context_with_sources,
    extract_query_intent
)

__all__ = [
    # Generation
    "AnswerGenerator",
    "GenerationConfig",
    "GenerationResult",
    "LLMBackend",
    "IntegratedValidator",
    "ValidatedAnswer",
    "ValidatedAnswerConfig",
    
    # Prompts
    "PromptType",
    "PromptTemplate",
    "PromptTemplateLibrary",
    "format_context_with_sources",
    "extract_query_intent"
]

