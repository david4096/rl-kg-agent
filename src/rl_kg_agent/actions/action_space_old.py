"""Action space definition for the RL agent."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from enum import IntEnum
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class ActionType(IntEnum):
    """Enumeration of available action types."""
    RESPOND_DIRECTLY = 0           # Direct LLM response without additional info
    QUERY_KG_THEN_RESPOND = 1     # Query static KG, then respond with LLM
    PLAN_THEN_RESPOND = 2         # Plan approach, then respond with LLM
    ASK_CLARIFYING_QUESTION = 3   # Ask for more information from user
    STORE_AND_RESPOND = 4         # Store to internal KG, then respond with LLM


@dataclass
class ActionResult:
    """Result of an action execution."""
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
        """Execute the action.

        Args:
            context: Current context including query, history, etc.
            **kwargs: Additional action-specific parameters

        Returns:
            ActionResult with execution outcome
        """
        pass

    @abstractmethod
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if this action is applicable in the current context.

        Args:
            context: Current context

        Returns:
            True if action can be applied, False otherwise
        """
        pass

    def get_description(self) -> str:
        """Get human-readable description of this action."""
        return f"Action {self.action_type.name}"


class RespondDirectlyAction(BaseAction):
    """Action to respond directly using LLM without additional information gathering."""

    def __init__(self, llm_client):
        super().__init__(ActionType.RESPOND_DIRECTLY)
        self.llm_client = llm_client

    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Execute direct LLM response action."""
        query = context.get("query", "")
        internal_knowledge = context.get("internal_knowledge", "")

        # Create simple prompt for direct response
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question directly and concisely."},
            {"role": "user", "content": query}
        ]

        # Add internal knowledge if available
        if internal_knowledge:
            messages[0]["content"] += f"\n\nRelevant context from memory: {internal_knowledge}"

        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent history
        for entry in conversation_history[-5:]:  # Last 5 exchanges
            messages.extend(entry)

        messages.append({"role": "user", "content": query})

        try:
            response = self.llm_client.generate_response(messages)

            return ActionResult(
                success=True,
                response=response,
                metadata={"action": "llm_response", "used_context": bool(internal_kg_context)},
                confidence=0.8
            )
        except Exception as e:
            logger.error(f"LLM response failed: {e}")
            return ActionResult(
                success=False,
                response=f"I apologize, I encountered an error while processing your question: {str(e)}",
                metadata={"action": "llm_response", "error": str(e)},
                confidence=0.0
            )

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """LLM response is always applicable as a fallback."""
        return True

    def get_description(self) -> str:
        return "Respond to the question using the language model based on available context"


class QueryStaticKGAction(BaseAction):
    """Action to write and execute a SPARQL query against the static knowledge graph."""

    def __init__(self, kg_loader, sparql_generator, llm_client=None):
        super().__init__(ActionType.QUERY_STATIC_KG)
        self.kg_loader = kg_loader
        self.sparql_generator = sparql_generator
        self.llm_client = llm_client

        # Initialize LLM-powered SPARQL generator if LLM client is available
        if llm_client:
            from ..knowledge.llm_sparql_generator import LLMSPARQLGenerator
            self.llm_sparql_generator = LLMSPARQLGenerator(llm_client, kg_loader)
        else:
            self.llm_sparql_generator = None

    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Execute SPARQL query action."""
        query = context.get("query", "")
        entities = context.get("entities", [])

        # Try to generate appropriate SPARQL query using LLM first, then fallback
        sparql_query = self._generate_sparql_query_with_llm(context)

        if not sparql_query:
            # Fallback to rule-based generation
            sparql_query = self._generate_sparql_query(query, entities)

        if not sparql_query:
            return ActionResult(
                success=False,
                response="Could not generate appropriate SPARQL query for this question.",
                metadata={"action": "sparql_query", "query_generated": False},
                confidence=0.0
            )

        # Validate syntax
        if not self.kg_loader.validate_sparql_syntax(sparql_query):
            return ActionResult(
                success=False,
                response="Generated SPARQL query has invalid syntax.",
                metadata={"action": "sparql_query", "sparql": sparql_query, "syntax_valid": False},
                confidence=0.0
            )

        # Execute query
        try:
            results = self.kg_loader.execute_sparql(sparql_query)

            if not results:
                response = "No results found for your query in the knowledge graph."
                success = False
            else:
                response = self._format_sparql_results(results, query)
                success = True

            # Extract entities and relations from results
            entities_discovered = self._extract_entities_from_results(results)
            relations_discovered = self._extract_relations_from_sparql(sparql_query)

            return ActionResult(
                success=success,
                response=response,
                metadata={
                    "action": "sparql_query",
                    "sparql": sparql_query,
                    "result_count": len(results),
                    "results": results[:5]  # First 5 results for metadata
                },
                entities_discovered=entities_discovered,
                relations_discovered=relations_discovered,
                confidence=0.9 if success else 0.3
            )

        except Exception as e:
            logger.error(f"SPARQL query execution failed: {e}")
            return ActionResult(
                success=False,
                response=f"Failed to execute knowledge graph query: {str(e)}",
                metadata={"action": "sparql_query", "sparql": sparql_query, "error": str(e)},
                confidence=0.0
            )

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """SPARQL query is applicable if entities are detected or question seems factual."""
        entities = context.get("entities", [])
        query = context.get("query", "").lower()

        # Check if entities are present or query contains question words suggesting factual lookup
        factual_indicators = ["what", "who", "when", "where", "which", "how many", "list", "find"]

        return len(entities) > 0 or any(indicator in query for indicator in factual_indicators)

    def _generate_sparql_query_with_llm(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate SPARQL query using LLM-powered generator.

        Args:
            context: Full conversation context

        Returns:
            Generated SPARQL query or None if LLM generation is not available/fails
        """
        if not self.llm_sparql_generator:
            logger.debug("LLM SPARQL generator not available, skipping LLM-based generation")
            return None

        try:
            sparql_query = self.llm_sparql_generator.generate_sparql_from_context(context)
            if sparql_query:
                logger.info("Successfully generated SPARQL query using LLM")
                return sparql_query
            else:
                logger.debug("LLM SPARQL generation returned no query")
                return None
        except Exception as e:
            logger.error(f"LLM SPARQL generation failed: {e}")
            return None

    def _generate_sparql_query(self, query: str, entities: List[str]) -> Optional[str]:
        """Generate SPARQL query based on natural language query and entities."""
        query_lower = query.lower()

        # Simple heuristics for query generation
        if entities:
            if "what" in query_lower or "which" in query_lower:
                return self.sparql_generator.generate_entity_search_query(entities[0])
            elif "relationship" in query_lower or "related" in query_lower:
                if len(entities) >= 2:
                    return self.sparql_generator.generate_relationship_query(entities[0], entities[1])
                else:
                    return self.sparql_generator.generate_neighbors_query(entities[0])

        # Fallback: try to extract keywords for general search
        keywords = [word for word in query.split() if len(word) > 3 and word.isalpha()]
        if keywords:
            return self.sparql_generator.generate_entity_search_query(keywords[0])

        return None

    def _format_sparql_results(self, results: List[Dict[str, Any]], original_query: str) -> str:
        """Format SPARQL results into human-readable response."""
        if len(results) == 1:
            result = results[0]
            if len(result) == 2:
                key, value = list(result.items())
                return f"The answer is: {value}"
            else:
                formatted = ", ".join([f"{k}: {v}" for k, v in result.items()])
                return f"I found: {formatted}"
        else:
            # Multiple results
            response = f"I found {len(results)} results:\\n"
            for i, result in enumerate(results[:10], 1):  # Limit to 10 results
                formatted = ", ".join([f"{k}: {v}" for k, v in result.items()])
                response += f"{i}. {formatted}\\n"

            if len(results) > 10:
                response += f"... and {len(results) - 10} more results."

            return response

    def _extract_entities_from_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract entity URIs from SPARQL results."""
        entities = set()
        for result in results:
            for value in result.values():
                if isinstance(value, str) and (value.startswith("http://") or value.startswith("https://")):
                    entities.add(value)
        return list(entities)

    def _extract_relations_from_sparql(self, sparql_query: str) -> List[str]:
        """Extract relations mentioned in SPARQL query."""
        # Simple extraction - could be enhanced
        relations = []
        if "?predicate" in sparql_query or "?relation" in sparql_query:
            relations.append("predicate_query")
        return relations

    def get_description(self) -> str:
        return "Query the static knowledge graph using SPARQL to find factual information"


class StoreToInternalKGAction(BaseAction):
    """Action to store information to the agent's internal knowledge graph."""

    def __init__(self, internal_kg):
        super().__init__(ActionType.STORE_TO_INTERNAL_KG)
        self.internal_kg = internal_kg

    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Execute storage action."""
        query = context.get("query", "")
        response = kwargs.get("response", "")
        entities = context.get("entities", [])
        relations = context.get("relations", [])
        action_taken = kwargs.get("previous_action", "unknown")
        success = kwargs.get("action_success", True)

        try:
            # Store the query context
            self.internal_kg.add_context(
                query=query,
                response=response,
                entities=entities,
                relations=relations,
                action=action_taken,
                success=success
            )

            # Store discovered entities as nodes
            stored_entities = 0
            for entity in entities:
                if entity:
                    self.internal_kg.add_node(
                        node_id=entity,
                        node_type="entity",
                        properties={"discovered_in_query": query[:100]}
                    )
                    stored_entities += 1

            # Store discovered relations as edges (simplified)
            stored_relations = 0
            if len(entities) >= 2 and relations:
                for relation in relations:
                    for i in range(len(entities) - 1):
                        self.internal_kg.add_edge(
                            source=entities[i],
                            target=entities[i + 1],
                            relation=relation,
                            derived_from=f"query: {query[:50]}..."
                        )
                        stored_relations += 1

            # Save to disk
            self.internal_kg.save()

            return ActionResult(
                success=True,
                response=f"Stored interaction context with {stored_entities} entities and {stored_relations} relations to memory.",
                metadata={
                    "action": "store_knowledge",
                    "entities_stored": stored_entities,
                    "relations_stored": stored_relations
                },
                confidence=1.0
            )

        except Exception as e:
            logger.error(f"Failed to store to internal KG: {e}")
            return ActionResult(
                success=False,
                response=f"Failed to store information to memory: {str(e)}",
                metadata={"action": "store_knowledge", "error": str(e)},
                confidence=0.0
            )

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Storage is applicable when we have learned something new."""
        entities = context.get("entities", [])
        return len(entities) > 0 or context.get("learned_something", False)

    def get_description(self) -> str:
        return "Store learned information and context to the agent's internal knowledge graph"


class AskRefiningQuestionAction(BaseAction):
    """Action to ask a clarifying question to improve understanding."""

    def __init__(self, llm_client):
        super().__init__(ActionType.ASK_REFINING_QUESTION)
        self.llm_client = llm_client

    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Execute refining question action."""
        query = context.get("query", "")
        ambiguity_reason = kwargs.get("ambiguity_reason", "unclear query")

        try:
            # Generate clarifying question using LLM
            prompt = f"""The user asked: "{query}"

This query is ambiguous or unclear because: {ambiguity_reason}

Generate a helpful clarifying question to better understand what the user is looking for.
The question should be specific and help narrow down the scope of the inquiry.
Keep it concise and friendly."""

            clarifying_question = self.llm_client.generate_response([
                {"role": "system", "content": "You generate helpful clarifying questions."},
                {"role": "user", "content": prompt}
            ])

            return ActionResult(
                success=True,
                response=clarifying_question,
                metadata={
                    "action": "refining_question",
                    "ambiguity_reason": ambiguity_reason
                },
                confidence=0.7
            )

        except Exception as e:
            logger.error(f"Failed to generate refining question: {e}")
            # Fallback to generic clarifying questions
            fallback_questions = [
                "Could you provide more specific details about what you're looking for?",
                "What particular aspect of this topic interests you most?",
                "Are you looking for a specific type of information or just general knowledge?",
                "Could you rephrase your question to be more specific?"
            ]

            return ActionResult(
                success=True,
                response=fallback_questions[hash(query) % len(fallback_questions)],
                metadata={
                    "action": "refining_question",
                    "fallback_used": True,
                    "ambiguity_reason": ambiguity_reason
                },
                confidence=0.5
            )

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Refining question is applicable when query is ambiguous or too broad."""
        query = context.get("query", "").lower()

        # Check for ambiguous/broad queries
        ambiguous_indicators = [
            len(query.split()) < 3,  # Very short queries
            "tell me about" in query,
            "what about" in query,
            "anything about" in query,
            not any(w in query for w in ["what", "who", "when", "where", "how", "why"])  # No question words
        ]

        return any(ambiguous_indicators)

    def get_description(self) -> str:
        return "Ask a clarifying question to better understand the user's intent"


class LLMPlanningStageAction(BaseAction):
    """Action to invoke LLM planning/reasoning stage."""

    def __init__(self, llm_client):
        super().__init__(ActionType.LLM_PLANNING_STAGE)
        self.llm_client = llm_client

    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Execute planning stage action."""
        query = context.get("query", "")
        available_actions = kwargs.get("available_actions", [])
        internal_knowledge = context.get("internal_knowledge", "")

        try:
            # Create planning prompt
            planning_prompt = f"""You are helping an AI agent decide how to answer the user's question: "{query}"

Available information:
- Internal knowledge from past interactions: {internal_knowledge or "None"}

Available actions the agent can take:
1. Respond directly using language model knowledge
2. Query the static knowledge graph for factual information
3. Store learned information to memory
4. Ask a clarifying question
5. Use this planning stage (current action)

Think step by step about:
1. What type of question this is
2. What information would be needed to answer it well
3. What would be the best sequence of actions to take
4. Any potential challenges or ambiguities

Provide a clear reasoning path and recommend the next best action."""

            planning_response = self.llm_client.generate_response([
                {"role": "system", "content": "You are an AI planning assistant that helps determine the best course of action."},
                {"role": "user", "content": planning_prompt}
            ])

            return ActionResult(
                success=True,
                response=f"Let me think through this step by step:\\n\\n{planning_response}",
                metadata={
                    "action": "llm_planning",
                    "planning_complete": True
                },
                confidence=0.8
            )

        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return ActionResult(
                success=False,
                response="I encountered an error while planning my approach to your question.",
                metadata={"action": "llm_planning", "error": str(e)},
                confidence=0.0
            )

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Planning is applicable for complex queries or when previous actions failed."""
        query = context.get("query", "")
        previous_failures = context.get("failed_actions", 0)

        # Apply planning for complex queries or after failures
        complex_indicators = [
            "how" in query.lower() and "why" in query.lower(),
            "explain" in query.lower(),
            "analyze" in query.lower(),
            "compare" in query.lower(),
            len(query.split()) > 10,  # Long queries
            previous_failures > 0
        ]

        return any(complex_indicators)

    def get_description(self) -> str:
        return "Engage in step-by-step planning and reasoning about how to approach the question"