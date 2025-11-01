"""
Prompt Templates for LLM Answer Generation
Provides optimized prompts for different query types and use cases
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum


class PromptType(Enum):
    """Types of prompts for different use cases"""
    STANDARD = "standard"
    CONCISE = "concise"
    DETAILED = "detailed"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    EXPLANATION = "explanation"


@dataclass
class PromptTemplate:
    """
    Prompt template with context formatting
    
    Attributes:
        template: The prompt template string with placeholders
        system_message: System message for chat models
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
        requires_sources: Whether to include source citations
    """
    template: str
    system_message: str
    max_tokens: int = 512
    temperature: float = 0.7
    requires_sources: bool = True
    
    def format(
        self,
        query: str,
        context: str,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Format the prompt with query and context
        
        Args:
            query: User query
            context: Retrieved context
            additional_context: Optional additional context
            **kwargs: Additional template variables
            
        Returns:
            Formatted prompt string
        """
        # Combine contexts if additional context provided
        full_context = context
        if additional_context:
            full_context = f"{context}\n\n{additional_context}"
        
        # Format template
        formatted = self.template.format(
            query=query,
            context=full_context,
            **kwargs
        )
        
        return formatted
    
    def to_messages(
        self,
        query: str,
        context: str,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Format as chat messages for chat-based models
        
        Args:
            query: User query
            context: Retrieved context
            additional_context: Optional additional context
            **kwargs: Additional template variables
            
        Returns:
            List of message dicts with 'role' and 'content'
        """
        formatted_prompt = self.format(query, context, additional_context, **kwargs)
        
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": formatted_prompt}
        ]


class PromptTemplateLibrary:
    """
    Library of prompt templates for different use cases
    
    All templates are optimized for:
    - Clear instruction following
    - Hallucination reduction
    - Source attribution
    - Concise yet complete answers
    """
    
    # System messages
    SYSTEM_STANDARD = """You are a precise and reliable AI assistant specialized in answering questions based on provided context.

Your responsibilities:
1. Answer ONLY using information from the provided context
2. Be accurate and factual - never make up information
3. If the context doesn't contain the answer, explicitly state that
4. Cite specific parts of the context when possible
5. Be clear, concise, and well-structured

Remember: Accuracy is more important than completeness. If you're unsure, say so."""

    SYSTEM_RESEARCH = """You are an expert research assistant helping users understand complex documents.

Your approach:
1. Analyze the provided context carefully
2. Extract key information relevant to the query
3. Provide well-reasoned answers with evidence
4. Highlight important findings and insights
5. Note any limitations or gaps in the available information

Prioritize precision and intellectual honesty."""

    # Standard QA template
    STANDARD_TEMPLATE = PromptTemplate(
        template="""Based on the following context, please answer the question accurately and concisely.

Context:
{context}

Question: {query}

Instructions:
- Answer using ONLY the information from the context above
- Be specific and cite relevant parts of the context
- If the context doesn't contain enough information, say so clearly
- Keep your answer focused and well-structured

Answer:""",
        system_message=SYSTEM_STANDARD,
        max_tokens=512,
        temperature=0.3,
        requires_sources=True
    )
    
    # Concise template for brief answers
    CONCISE_TEMPLATE = PromptTemplate(
        template="""Context:
{context}

Question: {query}

Provide a concise, direct answer (2-3 sentences maximum) based solely on the context above.

Answer:""",
        system_message=SYSTEM_STANDARD,
        max_tokens=150,
        temperature=0.2,
        requires_sources=True
    )
    
    # Detailed template for comprehensive answers
    DETAILED_TEMPLATE = PromptTemplate(
        template="""Based on the following context, provide a comprehensive answer to the question.

Context:
{context}

Question: {query}

Instructions:
- Provide a thorough, well-structured answer
- Include all relevant details from the context
- Organize your response with clear sections if needed
- Cite specific evidence from the context
- Explain relationships and implications where relevant

Answer:""",
        system_message=SYSTEM_RESEARCH,
        max_tokens=1024,
        temperature=0.4,
        requires_sources=True
    )
    
    # Comparison template
    COMPARISON_TEMPLATE = PromptTemplate(
        template="""Context:
{context}

Question: {query}

This question asks for a comparison. Please:
1. Identify the items being compared
2. List key similarities
3. List key differences
4. Provide a brief summary of the comparison

Base your comparison entirely on the provided context.

Answer:""",
        system_message=SYSTEM_RESEARCH,
        max_tokens=768,
        temperature=0.3,
        requires_sources=True
    )
    
    # Summary template
    SUMMARY_TEMPLATE = PromptTemplate(
        template="""Context:
{context}

Task: {query}

Please provide a clear, well-organized summary based on the context above. Include:
- Main points and key findings
- Important details and evidence
- Logical structure and flow

Answer:""",
        system_message=SYSTEM_RESEARCH,
        max_tokens=512,
        temperature=0.3,
        requires_sources=True
    )
    
    # Explanation template
    EXPLANATION_TEMPLATE = PromptTemplate(
        template="""Context:
{context}

Question: {query}

This question requires an explanation. Please:
1. Define key concepts from the context
2. Explain the mechanism/process/relationship
3. Provide examples or evidence from the context
4. Ensure the explanation is clear and accessible

Answer:""",
        system_message=SYSTEM_RESEARCH,
        max_tokens=768,
        temperature=0.4,
        requires_sources=True
    )
    
    @classmethod
    def get_template(cls, prompt_type: PromptType) -> PromptTemplate:
        """
        Get prompt template by type
        
        Args:
            prompt_type: Type of prompt to retrieve
            
        Returns:
            PromptTemplate instance
        """
        templates = {
            PromptType.STANDARD: cls.STANDARD_TEMPLATE,
            PromptType.CONCISE: cls.CONCISE_TEMPLATE,
            PromptType.DETAILED: cls.DETAILED_TEMPLATE,
            PromptType.COMPARISON: cls.COMPARISON_TEMPLATE,
            PromptType.SUMMARY: cls.SUMMARY_TEMPLATE,
            PromptType.EXPLANATION: cls.EXPLANATION_TEMPLATE
        }
        
        return templates.get(prompt_type, cls.STANDARD_TEMPLATE)
    
    @classmethod
    def get_template_for_query(cls, query: str) -> PromptTemplate:
        """
        Automatically select appropriate template based on query
        
        Args:
            query: User query
            
        Returns:
            Most appropriate PromptTemplate
        """
        query_lower = query.lower()
        
        # Comparison queries
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs", "contrast"]):
            return cls.get_template(PromptType.COMPARISON)
        
        # Summary queries
        if any(word in query_lower for word in ["summarize", "summary", "overview", "brief"]):
            return cls.get_template(PromptType.SUMMARY)
        
        # Explanation queries
        if any(word in query_lower for word in ["explain", "how does", "why does", "what is", "describe"]):
            return cls.get_template(PromptType.EXPLANATION)
        
        # Concise queries (questions asking for specific facts)
        if any(word in query_lower for word in ["when", "where", "who", "which year", "what year"]):
            return cls.get_template(PromptType.CONCISE)
        
        # Detailed queries
        if any(word in query_lower for word in ["detailed", "comprehensive", "thorough", "all", "everything"]):
            return cls.get_template(PromptType.DETAILED)
        
        # Default to standard template
        return cls.get_template(PromptType.STANDARD)
    
    @classmethod
    def create_custom_template(
        cls,
        template: str,
        system_message: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3
    ) -> PromptTemplate:
        """
        Create a custom prompt template
        
        Args:
            template: Template string with {query} and {context} placeholders
            system_message: Optional system message (uses default if None)
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            
        Returns:
            Custom PromptTemplate instance
        """
        return PromptTemplate(
            template=template,
            system_message=system_message or cls.SYSTEM_STANDARD,
            max_tokens=max_tokens,
            temperature=temperature,
            requires_sources=True
        )


def format_context_with_sources(
    context_text: str,
    sources: List[Dict[str, Any]],
    include_metadata: bool = True
) -> str:
    """
    Format context with source attributions
    
    Args:
        context_text: Main context text
        sources: List of source metadata dicts
        include_metadata: Whether to include source metadata
        
    Returns:
        Formatted context string with sources
    """
    if not sources:
        return context_text
    
    formatted_parts = []
    
    # Add main context
    formatted_parts.append("=== Context ===")
    formatted_parts.append(context_text)
    formatted_parts.append("")
    
    # Add source information
    if include_metadata:
        formatted_parts.append("=== Sources ===")
        for i, source in enumerate(sources, 1):
            source_info = [f"[Source {i}]"]
            
            if "document_id" in source:
                source_info.append(f"Document: {source['document_id']}")
            
            if "section_title" in source:
                source_info.append(f"Section: {source['section_title']}")
            
            if "relevance_score" in source:
                score = source['relevance_score']
                source_info.append(f"Relevance: {score:.2f}")
            
            formatted_parts.append(" | ".join(source_info))
        
        formatted_parts.append("")
    
    return "\n".join(formatted_parts)


def extract_query_intent(query: str) -> Dict[str, Any]:
    """
    Extract intent and metadata from query
    
    Args:
        query: User query
        
    Returns:
        Dict with intent information
    """
    query_lower = query.lower()
    
    intent = {
        "type": "general",
        "requires_comparison": False,
        "requires_explanation": False,
        "requires_summary": False,
        "is_factual": False,
        "is_opinion": False,
        "complexity": "medium"
    }
    
    # Detect question type
    if any(word in query_lower for word in ["compare", "difference", "versus"]):
        intent["type"] = "comparison"
        intent["requires_comparison"] = True
        intent["complexity"] = "high"
    
    elif any(word in query_lower for word in ["explain", "how", "why"]):
        intent["type"] = "explanation"
        intent["requires_explanation"] = True
        intent["complexity"] = "high"
    
    elif any(word in query_lower for word in ["summarize", "summary", "overview"]):
        intent["type"] = "summary"
        intent["requires_summary"] = True
        intent["complexity"] = "medium"
    
    elif any(word in query_lower for word in ["when", "where", "who", "what year"]):
        intent["type"] = "factual"
        intent["is_factual"] = True
        intent["complexity"] = "low"
    
    # Detect opinion vs factual
    if any(word in query_lower for word in ["should", "better", "best", "worst", "opinion"]):
        intent["is_opinion"] = True
    
    return intent

