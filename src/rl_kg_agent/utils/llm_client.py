"""LLM client for interfacing with language models."""

from typing import List, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters

        Returns:
            Generated response string
        """
        pass


class LLMClient(BaseLLMClient):
    """Simple LLM client (placeholder implementation)."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize LLM client.

        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        logger.info(f"Initialized LLM client with model: {model_name}")

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response (placeholder implementation).

        This is a placeholder implementation. In a real system, you would:
        1. Connect to OpenAI API, Anthropic API, or local model
        2. Send the messages to the model
        3. Return the generated response

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        # Extract the user's question and system message from messages
        user_message = ""
        system_content = ""
        for message in messages:
            if message.get("role") == "user":
                user_message = message.get("content", "")
            elif message.get("role") == "system":
                system_content = message.get("content", "")

        # Determine action type based on system message to provide diverse responses
        action_type = self._determine_action_type(system_content)
        user_lower = user_message.lower()

        # Check if this is a SPARQL generation request
        if "generate a valid sparql" in user_lower or "sparql query" in user_lower:
            return self._generate_sparql_response(user_message)

        # Action-specific responses for better diversity during training
        if action_type == "direct_response":
            return self._generate_direct_response(user_message, user_lower)
        elif action_type == "kg_enhanced":
            return self._generate_kg_enhanced_response(user_message, user_lower, system_content)
        elif action_type == "planned":
            return self._generate_planned_response(user_message, user_lower)
        elif action_type == "clarifying":
            return self._generate_clarifying_question(user_message, user_lower)
        elif action_type == "store_and_respond":
            return self._generate_store_response(user_message, user_lower)
        else:
            # Default fallback
            return self._generate_direct_response(user_message, user_lower)

    def _determine_action_type(self, system_content: str) -> str:
        """Determine the action type based on system message."""
        system_lower = system_content.lower()
        if "planning assistant" in system_lower or "create a brief plan" in system_lower:
            return "planned"
        elif "knowledge base" in system_lower or "provided information" in system_lower:
            return "kg_enhanced"
        elif "unclear" in system_lower or "clarifying question" in system_lower:
            return "clarifying"
        elif "saved this interaction" in system_lower or "future reference" in system_lower:
            return "store_and_respond"
        else:
            return "direct_response"

    def _generate_direct_response(self, user_message: str, user_lower: str) -> str:
        """Generate direct responses."""
        if "capital" in user_lower and "france" in user_lower:
            return "The capital of France is Paris."
        elif "romeo" in user_lower and "juliet" in user_lower:
            return "Romeo and Juliet was written by William Shakespeare."
        elif "photosynthesis" in user_lower:
            return "Photosynthesis is the process by which plants convert light energy into chemical energy using chlorophyll."
        elif "sky" in user_lower and "color" in user_lower:
            return "The sky appears blue due to light scattering."
        else:
            return f"Based on my knowledge, here's what I can tell you about your question: {user_message[:50]}..."

    def _generate_kg_enhanced_response(self, user_message: str, user_lower: str, system_content: str) -> str:
        """Generate knowledge-graph enhanced responses."""
        # Check if KG information is provided in context
        has_kg_info = "Knowledge Graph Information:" in system_content

        if "capital" in user_lower and "france" in user_lower:
            if has_kg_info:
                return "According to the knowledge graph, Paris is the capital of France, as indicated by the relationship ex:isCapitalOf."
            else:
                return "The capital of France is Paris, though I couldn't find specific information in the knowledge graph."
        elif "romeo" in user_lower and "juliet" in user_lower:
            if has_kg_info:
                return "The knowledge graph shows that Romeo and Juliet was written by William Shakespeare, as indicated by the ex:writtenBy relationship."
            else:
                return "Romeo and Juliet was written by William Shakespeare, but I couldn't find this information in the current knowledge graph."
        else:
            if has_kg_info:
                return f"Based on the knowledge graph data provided, I can tell you about {user_message[:30]}... The graph contains relevant information about this topic."
            else:
                return f"I searched the knowledge graph for information about '{user_message[:30]}...' but didn't find specific details. Let me provide what I know from general knowledge."

    def _generate_planned_response(self, user_message: str, user_lower: str) -> str:
        """Generate planned, comprehensive responses."""
        if "capital" in user_lower and "france" in user_lower:
            return "To answer your question about France's capital: First, I'll identify that you're asking about a geographical relationship. Then, I'll provide the factual answer that Paris is the capital of France. Additionally, I can mention that Paris has been France's capital since 987 AD."
        elif "romeo" in user_lower and "juliet" in user_lower:
            return "To address your question about Romeo and Juliet: First, I'll identify this as a literary inquiry. The play was written by William Shakespeare around 1595. It's one of his most famous tragedies, exploring themes of love, fate, and family conflict in Renaissance Verona."
        elif "how" in user_lower or "why" in user_lower:
            return f"To properly answer your complex question about '{user_message[:40]}...', let me break this down systematically. I'll consider multiple perspectives, examine the underlying principles, and provide a comprehensive explanation that addresses the key aspects of your inquiry."
        else:
            return f"After planning my approach to your question '{user_message[:40]}...', I'll provide a structured response that covers the main points, relevant context, and practical implications of this topic."

    def _generate_clarifying_question(self, user_message: str, user_lower: str) -> str:
        """Generate clarifying questions."""
        if len(user_message.split()) < 3:
            return "Could you provide more details about what specifically you'd like to know?"
        elif "tell me about" in user_lower:
            return "What specific aspect would you like me to focus on? There are many angles I could explore."
        elif "what about" in user_lower:
            return "Could you be more specific about what information you're looking for?"
        else:
            return f"I want to make sure I understand your question correctly. When you ask about '{user_message[:40]}...', are you looking for a general overview or specific details about a particular aspect?"

    def _generate_store_response(self, user_message: str, user_lower: str) -> str:
        """Generate responses for store-and-respond actions."""
        base_response = self._generate_direct_response(user_message, user_lower)
        return base_response + " I've saved this interaction to help with future similar questions."

    def _generate_sparql_response(self, user_message: str) -> str:
        """Generate SPARQL query responses based on the request.

        Args:
            user_message: The SPARQL generation request

        Returns:
            Generated SPARQL query string
        """
        # Extract the actual question from the SPARQL generation request
        lines = user_message.split('\n')
        question_line = ""
        for line in lines:
            if line.startswith('Question: "') and line.endswith('"'):
                question_line = line[10:-1]  # Extract text between 'Question: "' and '"'
                break

        if not question_line:
            # Try to find the question in other formats
            for line in lines:
                if "capital" in line.lower() and "france" in line.lower():
                    question_line = line
                    break
                elif "wrote" in line.lower() or "written" in line.lower():
                    question_line = line
                    break

        # Generate appropriate SPARQL based on the question
        question_lower = question_line.lower() if question_line else user_message.lower()

        if "capital" in question_lower and "france" in question_lower:
            return """PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?capital ?capitalLabel WHERE {
    ?capital ex:isCapitalOf ?country .
    ?country rdfs:label "France" .
    OPTIONAL { ?capital rdfs:label ?capitalLabel }
} LIMIT 10"""

        elif "wrote" in question_lower or "written" in question_lower:
            if "romeo" in question_lower and "juliet" in question_lower:
                return """PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?author ?authorLabel WHERE {
    ?work rdfs:label "Romeo and Juliet" .
    ?work ex:writtenBy ?author .
    OPTIONAL { ?author rdfs:label ?authorLabel }
} LIMIT 10"""
            else:
                return """PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?work ?author ?authorLabel WHERE {
    ?work ex:writtenBy ?author .
    OPTIONAL { ?author rdfs:label ?authorLabel }
    OPTIONAL { ?work rdfs:label ?workLabel }
} LIMIT 20"""

        elif "located" in question_lower or "where" in question_lower:
            return """PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?entity ?location ?locationLabel WHERE {
    ?entity ex:locatedIn ?location .
    OPTIONAL { ?location rdfs:label ?locationLabel }
    OPTIONAL { ?entity rdfs:label ?entityLabel }
} LIMIT 20"""

        else:
            # General exploration query
            return """PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?subject ?predicate ?object WHERE {
    ?subject ?predicate ?object .
    OPTIONAL { ?subject rdfs:label ?subjectLabel }
    OPTIONAL { ?object rdfs:label ?objectLabel }
} LIMIT 20"""


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing."""

    def __init__(self, responses: Optional[List[str]] = None):
        """Initialize mock client.

        Args:
            responses: Predefined responses to cycle through
        """
        self.responses = responses or [
            "This is a mock response.",
            "Another mock response for testing.",
            "Mock explanation of the topic."
        ]
        self.call_count = 0

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Return mock response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response