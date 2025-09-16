"""Redesigned action space that always ends with user responses."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from enum import IntEnum
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class ActionType(IntEnum):
    """Enumeration of available action types - all end with user response."""
    RESPOND_DIRECTLY = 0           # Direct LLM response without additional info
    QUERY_KG_THEN_RESPOND = 1     # Query static KG, then respond with LLM
    PLAN_THEN_RESPOND = 2         # Plan approach, then respond with LLM
    ASK_CLARIFYING_QUESTION = 3   # Ask for more information from user
    STORE_AND_RESPOND = 4         # Store to internal KG, then respond with LLM


@dataclass
class ActionResult:
    """Result of an action execution - always contains user response."""
    success: bool
    response: str                    # Always contains user-facing response
    metadata: Dict[str, Any]
    entities_discovered: List[str] = None
    relations_discovered: List[str] = None
    confidence: float = 1.0
    is_final: bool = True           # All actions now produce final responses

    def __post_init__(self):
        if self.entities_discovered is None:
            self.entities_discovered = []
        if self.relations_discovered is None:
            self.relations_discovered = []


class BaseAction(ABC):
    """Base class for all actions."""

    def __init__(self, action_type: ActionType):
        self.action_type = action_type

    @abstractmethod
    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Execute the action and always return a user response."""
        pass

    @abstractmethod
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if this action is applicable in the current context."""
        pass

    def get_description(self) -> str:
        """Get human-readable description of this action."""
        return f"Action {self.action_type.name}"


class RespondDirectlyAction(BaseAction):
    """Respond directly using LLM without additional information gathering."""

    def __init__(self, llm_client):
        super().__init__(ActionType.RESPOND_DIRECTLY)
        self.llm_client = llm_client

    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Execute direct LLM response."""
        query = context.get("query", "")
        internal_knowledge = context.get("internal_knowledge", "")

        try:
            # Create messages for LLM
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question directly and concisely."},
                {"role": "user", "content": query}
            ]

            # Add internal knowledge context if available
            if internal_knowledge:
                messages[0]["content"] += f"\\n\\nRelevant context from memory: {internal_knowledge}"

            # Generate response
            response = self.llm_client.generate_response(messages)

            return ActionResult(
                success=True,
                response=response,
                metadata={
                    "action": "direct_response",
                    "used_internal_knowledge": bool(internal_knowledge)
                },
                confidence=0.8
            )

        except Exception as e:
            logger.error(f"Direct response failed: {e}")
            return ActionResult(
                success=False,
                response="I'm sorry, I had trouble processing your question. Could you please rephrase it?",
                metadata={"action": "direct_response", "error": str(e)},
                confidence=0.0
            )

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Always applicable - this is the fallback action."""
        return True

    def get_description(self) -> str:
        return "Respond directly to the user's question using available context"


class QueryKGThenRespondAction(BaseAction):
    """Query knowledge graph for information, then respond with enhanced context."""

    def __init__(self, kg_loader, sparql_generator, llm_client):
        super().__init__(ActionType.QUERY_KG_THEN_RESPOND)
        self.kg_loader = kg_loader
        self.sparql_generator = sparql_generator
        self.llm_client = llm_client

    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Query KG then provide enhanced response."""
        query = context.get("query", "")
        entities = context.get("entities", [])

        try:
            # First, query the knowledge graph
            kg_info = ""
            sparql_query = None

            # Try to generate SPARQL query
            if hasattr(self.sparql_generator, 'generate_sparql_from_context'):
                # Use LLM-powered SPARQL generation
                sparql_query = self.sparql_generator.generate_sparql_from_context(context)
            else:
                # Fallback to simple entity-based queries
                sparql_query = self._generate_simple_query(query, entities)

            if sparql_query and self.kg_loader.validate_sparql_syntax(sparql_query):
                results = self.kg_loader.execute_sparql(sparql_query)
                if results:
                    kg_info = self._format_kg_results(results)

            # Now generate response with KG information
            enhanced_context = context.get("internal_knowledge", "")
            if kg_info:
                enhanced_context = f"{enhanced_context}\\n\\nKnowledge Graph Information: {kg_info}".strip()

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to a knowledge base. Answer the user's question using the provided information."
                },
                {"role": "user", "content": query}
            ]

            if enhanced_context:
                messages[0]["content"] += f"\\n\\nAvailable information: {enhanced_context}"

            response = self.llm_client.generate_response(messages)

            return ActionResult(
                success=True,
                response=response,
                metadata={
                    "action": "kg_query_response",
                    "sparql_query": sparql_query,
                    "kg_results_found": len(results) if 'results' in locals() else 0,
                    "used_kg_info": bool(kg_info)
                },
                confidence=0.9 if kg_info else 0.6
            )

        except Exception as e:
            logger.error(f"KG query and response failed: {e}")
            # Fallback to direct response
            try:
                response = self.llm_client.generate_response([
                    {"role": "system", "content": "You are a helpful assistant. Answer the user's question."},
                    {"role": "user", "content": query}
                ])
                return ActionResult(
                    success=True,
                    response=response,
                    metadata={"action": "kg_query_response", "fallback": True, "error": str(e)},
                    confidence=0.5
                )
            except Exception as fallback_error:
                return ActionResult(
                    success=False,
                    response="I'm sorry, I couldn't find information to answer your question.",
                    metadata={"action": "kg_query_response", "error": str(e), "fallback_error": str(fallback_error)},
                    confidence=0.0
                )

    def _generate_simple_query(self, query: str, entities: List[str]) -> Optional[str]:
        """Generate simple SPARQL query based on entities."""
        if not entities:
            return None

        entity = entities[0]
        return f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?s ?p ?o WHERE {{
            {{
                ?s rdfs:label "{entity}" .
                ?s ?p ?o
            }}
            UNION
            {{
                ?o rdfs:label "{entity}" .
                ?s ?p ?o
            }}
        }} LIMIT 10
        """

    def _format_kg_results(self, results: List[Dict]) -> str:
        """Format KG query results for context."""
        if not results:
            return ""

        formatted = []
        for i, result in enumerate(results[:5]):  # Limit to 5 results
            items = [f"{k}: {v}" for k, v in result.items()]
            formatted.append(f"- {', '.join(items)}")

        return "\\n".join(formatted)

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Applicable when we have entities or factual questions."""
        entities = context.get("entities", [])
        query = context.get("query", "").lower()

        # Good for factual questions or when entities are present
        factual_keywords = ["what", "who", "when", "where", "which", "how many"]
        has_factual_keyword = any(keyword in query for keyword in factual_keywords)

        return bool(entities) or has_factual_keyword

    def get_description(self) -> str:
        return "Query knowledge graph for information, then provide informed response"


class PlanThenRespondAction(BaseAction):
    """Plan the response approach, then provide a comprehensive answer."""

    def __init__(self, llm_client):
        super().__init__(ActionType.PLAN_THEN_RESPOND)
        self.llm_client = llm_client

    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Plan approach then respond."""
        query = context.get("query", "")

        try:
            # First, plan the approach
            planning_messages = [
                {
                    "role": "system",
                    "content": "You are an expert planning assistant. Analyze the user's question and create a brief plan for how to answer it comprehensively."
                },
                {
                    "role": "user",
                    "content": f"How should I approach answering this question: '{query}'"
                }
            ]

            plan = self.llm_client.generate_response(planning_messages)

            # Now generate the actual response using the plan
            response_messages = [
                {
                    "role": "system",
                    "content": f"You are a knowledgeable assistant. Answer the user's question following this plan: {plan}"
                },
                {"role": "user", "content": query}
            ]

            # Add internal knowledge if available
            internal_knowledge = context.get("internal_knowledge", "")
            if internal_knowledge:
                response_messages[0]["content"] += f"\\n\\nRelevant context: {internal_knowledge}"

            response = self.llm_client.generate_response(response_messages)

            return ActionResult(
                success=True,
                response=response,
                metadata={
                    "action": "planned_response",
                    "plan": plan,
                    "used_internal_knowledge": bool(internal_knowledge)
                },
                confidence=0.85
            )

        except Exception as e:
            logger.error(f"Planned response failed: {e}")
            # Fallback to direct response
            try:
                response = self.llm_client.generate_response([
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ])
                return ActionResult(
                    success=True,
                    response=response,
                    metadata={"action": "planned_response", "fallback": True, "error": str(e)},
                    confidence=0.5
                )
            except Exception as fallback_error:
                return ActionResult(
                    success=False,
                    response="I need more information to answer your question properly.",
                    metadata={"action": "planned_response", "error": str(e)},
                    confidence=0.0
                )

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Applicable for complex questions that benefit from planning."""
        query = context.get("query", "").lower()

        # Good for complex, analytical questions
        complex_indicators = ["explain", "analyze", "compare", "why", "how", "what are the differences"]
        is_complex = any(indicator in query for indicator in complex_indicators)

        # Also good for longer questions
        is_long = len(query.split()) > 8

        return is_complex or is_long

    def get_description(self) -> str:
        return "Plan a comprehensive approach, then provide detailed response"


class AskClarifyingQuestionAction(BaseAction):
    """Ask the user for clarification when the question is ambiguous."""

    def __init__(self, llm_client):
        super().__init__(ActionType.ASK_CLARIFYING_QUESTION)
        self.llm_client = llm_client

    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Ask for clarification."""
        query = context.get("query", "")

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. The user's question is unclear or ambiguous. Generate a clarifying question to better understand what they're asking."
                },
                {
                    "role": "user",
                    "content": f"This question is unclear: '{query}'. What should I ask to clarify?"
                }
            ]

            clarifying_question = self.llm_client.generate_response(messages)

            return ActionResult(
                success=True,
                response=clarifying_question,
                metadata={
                    "action": "clarifying_question",
                    "original_query": query
                },
                confidence=0.7
            )

        except Exception as e:
            logger.error(f"Clarifying question generation failed: {e}")
            return ActionResult(
                success=True,
                response="Could you please provide more details about what specifically you'd like to know?",
                metadata={"action": "clarifying_question", "error": str(e)},
                confidence=0.5
            )

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Applicable for short, vague, or ambiguous questions."""
        query = context.get("query", "")

        if len(query.split()) < 3:
            return True

        # Vague question indicators
        vague_indicators = ["tell me about", "what about", "anything about", "stuff about"]
        return any(indicator in query.lower() for indicator in vague_indicators)

    def get_description(self) -> str:
        return "Ask clarifying questions to better understand user's intent"


class StoreAndRespondAction(BaseAction):
    """Store information to internal KG, then respond with confirmation."""

    def __init__(self, internal_kg, llm_client):
        super().__init__(ActionType.STORE_AND_RESPOND)
        self.internal_kg = internal_kg
        self.llm_client = llm_client

    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Store information then respond."""
        query = context.get("query", "")
        entities = context.get("entities", [])

        try:
            # Store the interaction context
            stored_successfully = False

            if entities or query:
                # Create context for storage
                storage_context = {
                    "query": query,
                    "entities": entities,
                    "response": kwargs.get("response", ""),
                    "timestamp": kwargs.get("timestamp")
                }

                # Store to internal KG
                context_id = self.internal_kg.add_context(
                    query=query,
                    response=storage_context.get("response", "Processing query..."),
                    entities=entities,
                    success=True
                )

                if context_id:
                    stored_successfully = True

            # Now generate response with storage confirmation
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the user's question and briefly mention that you've saved this interaction for future reference."
                },
                {"role": "user", "content": query}
            ]

            # Add internal knowledge context
            internal_knowledge = context.get("internal_knowledge", "")
            if internal_knowledge:
                messages[0]["content"] += f"\\n\\nRelevant context from memory: {internal_knowledge}"

            response = self.llm_client.generate_response(messages)

            # Add storage confirmation to response
            if stored_successfully:
                response += "\\n\\n(I've saved this interaction to help with future similar questions.)"

            return ActionResult(
                success=True,
                response=response,
                metadata={
                    "action": "store_and_respond",
                    "stored_successfully": stored_successfully,
                    "entities_stored": len(entities)
                },
                confidence=0.8 if stored_successfully else 0.6
            )

        except Exception as e:
            logger.error(f"Store and respond failed: {e}")
            # Still try to respond even if storage failed
            try:
                response = self.llm_client.generate_response([
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ])
                return ActionResult(
                    success=True,
                    response=response,
                    metadata={"action": "store_and_respond", "storage_error": str(e)},
                    confidence=0.5
                )
            except Exception as fallback_error:
                return ActionResult(
                    success=False,
                    response="I'm having trouble processing your request right now.",
                    metadata={"action": "store_and_respond", "error": str(e)},
                    confidence=0.0
                )

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Applicable when we have new information worth storing."""
        entities = context.get("entities", [])
        query = context.get("query", "")

        # Good for storing factual information, new entities, or successful interactions
        has_entities = bool(entities)
        has_meaningful_content = len(query.split()) > 3

        return has_entities and has_meaningful_content

    def get_description(self) -> str:
        return "Store interaction in memory and respond with confirmation"