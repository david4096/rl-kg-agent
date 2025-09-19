"""Action manager for coordinating actions in the RL agent."""

from typing import Dict, List, Any, Optional, Tuple
import logging
from .action_space import (
    ActionType, ActionResult, BaseAction,
    RespondDirectlyAction, QueryKGThenRespondAction, PlanThenRespondAction,
    AskClarifyingQuestionAction, StoreAndRespondAction, QueryMCPThenRespondAction
)


logger = logging.getLogger(__name__)


class ActionManager:
    """Manages and coordinates available actions for the RL agent."""

    def __init__(self, kg_loader, sparql_generator, internal_kg, llm_client, mcp_manager=None):
        """Initialize action manager with required dependencies.

        Args:
            kg_loader: Knowledge graph loader instance
            sparql_generator: SPARQL query generator instance
            internal_kg: Internal knowledge graph instance
            llm_client: LLM client for text generation
            mcp_manager: MCP manager for biomedical queries (optional)
        """
        self.kg_loader = kg_loader
        self.internal_kg = internal_kg
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager

        # Initialize all actions - each always ends with user response
        self.actions: Dict[ActionType, BaseAction] = {
            ActionType.RESPOND_DIRECTLY: RespondDirectlyAction(llm_client),
            ActionType.QUERY_KG_THEN_RESPOND: QueryKGThenRespondAction(kg_loader, sparql_generator, llm_client),
            ActionType.PLAN_THEN_RESPOND: PlanThenRespondAction(llm_client),
            ActionType.ASK_CLARIFYING_QUESTION: AskClarifyingQuestionAction(llm_client),
            # ActionType.STORE_AND_RESPOND: StoreAndRespondAction(internal_kg, llm_client)  # DISABLED - has errors
        }
        
        # Add MCP action if manager is available
        if mcp_manager:
            self.actions[ActionType.QUERY_MCP_THEN_RESPOND] = QueryMCPThenRespondAction(mcp_manager, llm_client)

        logger.info(f"Initialized ActionManager with {len(self.actions)} actions")

    def get_applicable_actions(self, context: Dict[str, Any]) -> List[ActionType]:
        """Get list of actions that are applicable in the current context.

        Args:
            context: Current context including query, entities, etc.

        Returns:
            List of applicable action types
        """
        applicable = []
        for action_type, action in self.actions.items():
            if action.is_applicable(context):
                applicable.append(action_type)

        logger.debug(f"Found {len(applicable)} applicable actions: {applicable}")
        return applicable

    def execute_action(self, action_type: ActionType, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Execute a specific action.

        Args:
            action_type: Type of action to execute
            context: Current context
            **kwargs: Additional action-specific parameters

        Returns:
            ActionResult with execution outcome
        """
        if action_type not in self.actions:
            logger.error(f"Unknown action type: {action_type}")
            return ActionResult(
                success=False,
                response=f"Unknown action type: {action_type}",
                metadata={"error": "unknown_action"},
                confidence=0.0
            )

        action = self.actions[action_type]
        logger.info(f"Executing action: {action_type.name}")

        try:
            result = action.execute(context, **kwargs)
            logger.info(f"Action {action_type.name} completed with success={result.success}")
            return result
        except Exception as e:
            logger.error(f"Action {action_type.name} failed with exception: {e}")
            return ActionResult(
                success=False,
                response=f"Action failed with error: {str(e)}",
                metadata={"error": str(e), "action": action_type.name},
                confidence=0.0
            )

    def execute_action_sequence(self, action_sequence: List[Tuple[ActionType, Dict[str, Any]]],
                              context: Dict[str, Any]) -> List[ActionResult]:
        """Execute a sequence of actions.

        Args:
            action_sequence: List of (action_type, kwargs) tuples
            context: Initial context

        Returns:
            List of ActionResults from each action
        """
        results = []
        current_context = context.copy()

        for action_type, action_kwargs in action_sequence:
            result = self.execute_action(action_type, current_context, **action_kwargs)
            results.append(result)

            # Update context with results for next action
            if result.success:
                current_context["last_action_result"] = result
                current_context["last_response"] = result.response
                if result.entities_discovered:
                    current_context["entities"] = list(set(
                        current_context.get("entities", []) + result.entities_discovered
                    ))
                if result.relations_discovered:
                    current_context["relations"] = list(set(
                        current_context.get("relations", []) + result.relations_discovered
                    ))
            else:
                current_context["failed_actions"] = current_context.get("failed_actions", 0) + 1

        return results

    def get_action_recommendations(self, context: Dict[str, Any]) -> List[Tuple[ActionType, float]]:
        """Get recommended actions with confidence scores.

        Args:
            context: Current context

        Returns:
            List of (action_type, confidence) tuples sorted by confidence
        """
        applicable_actions = self.get_applicable_actions(context)
        recommendations = []

        for action_type in applicable_actions:
            confidence = self._calculate_action_confidence(action_type, context)
            recommendations.append((action_type, confidence))

        # Sort by confidence (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

    def _calculate_action_confidence(self, action_type: ActionType, context: Dict[str, Any]) -> float:
        """Calculate confidence score for an action in given context.

        Args:
            action_type: Action to evaluate
            context: Current context

        Returns:
            Confidence score between 0 and 1
        """
        query = context.get("query", "").lower()
        entities = context.get("entities", [])
        failed_actions = context.get("failed_actions", 0)

        # Base confidence scores for new action types
        base_confidence = {
            ActionType.RESPOND_DIRECTLY: 0.75,          # Reliable fallback - increased
            ActionType.QUERY_KG_THEN_RESPOND: 0.85,    # Good for factual questions - increased
            ActionType.PLAN_THEN_RESPOND: 0.65,        # Good for complex questions - decreased to prevent overuse
            ActionType.ASK_CLARIFYING_QUESTION: 0.65,  # Good for ambiguous questions - increased
            ActionType.QUERY_MCP_THEN_RESPOND: 0.80,   # Good for biomedical questions
            # ActionType.STORE_AND_RESPOND: 0.7          # Good when learning new info - DISABLED
        }

        confidence = base_confidence.get(action_type, 0.5)

        # Adjust based on context for new action types
        if action_type == ActionType.QUERY_KG_THEN_RESPOND:
            # Higher confidence if entities detected or factual question
            if entities:
                confidence += 0.15
            factual_keywords = ["what", "who", "when", "where", "how many", "list"]
            if any(keyword in query for keyword in factual_keywords):
                confidence += 0.1

        elif action_type == ActionType.ASK_CLARIFYING_QUESTION:
            # Higher confidence for ambiguous/short queries
            if len(query.split()) < 3:
                confidence += 0.25
            elif any(vague in query for vague in ["tell me about", "what about", "anything about"]):
                confidence += 0.2

        elif action_type == ActionType.PLAN_THEN_RESPOND:
            # Higher confidence for complex queries
            if len(query.split()) > 8:
                confidence += 0.15
            if any(word in query for word in ["explain", "analyze", "compare", "why", "how"]):
                confidence += 0.1

        # elif action_type == ActionType.STORE_AND_RESPOND:  # DISABLED
        #     # Higher confidence if we have meaningful new entities
        #     if entities and len(query.split()) > 3:
        #         confidence += 0.2
        #     # Lower confidence if we've been storing too much recently
        #     if failed_actions > 2:
        #         confidence -= 0.3

        elif action_type == ActionType.QUERY_MCP_THEN_RESPOND:
            # Higher confidence for biomedical/scientific queries
            biomedical_keywords = [
                "gene", "protein", "disease", "drug", "medicine", "clinical", "medical",
                "biology", "biochemistry", "pharmacology", "pathology", "anatomy",
                "pubmed", "medline", "literature", "research", "study", "paper"
            ]
            if any(keyword in query for keyword in biomedical_keywords):
                confidence += 0.15
            
            # Higher confidence for research/literature questions
            research_patterns = ["research on", "studies about", "papers about", "find papers"]
            if any(pattern in query for pattern in research_patterns):
                confidence += 0.10
            
            # Higher confidence for factual biomedical questions
            factual_keywords = ["what", "how", "why", "where", "when", "which"]
            if any(keyword in query for keyword in factual_keywords) and len(query.split()) > 3:
                confidence += 0.05
            
            # Lower confidence if MCP manager is not available
            if not self.mcp_manager:
                confidence = 0.0

        elif action_type == ActionType.RESPOND_DIRECTLY:
            # This is the fallback - gets higher confidence when others fail
            if failed_actions > 1:
                confidence += 0.2

        # Reduce confidence if this action type has failed recently
        if failed_actions > 0 and action_type == context.get("last_failed_action"):
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))

    def get_action_descriptions(self) -> Dict[ActionType, str]:
        """Get human-readable descriptions of all actions.

        Returns:
            Dictionary mapping action types to descriptions
        """
        return {
            action_type: action.get_description()
            for action_type, action in self.actions.items()
        }

    def update_context_with_internal_knowledge(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update context with relevant information from internal knowledge graph.

        Args:
            context: Current context

        Returns:
            Updated context with internal knowledge
        """
        query = context.get("query", "")
        entities = context.get("entities", [])

        # Get relevant nodes from internal KG
        relevant_nodes = self.internal_kg.get_relevant_nodes(entities)

        # Get similar past contexts
        similar_contexts = self.internal_kg.get_similar_contexts(query, limit=3)

        # Format internal knowledge for context
        internal_knowledge_parts = []

        if relevant_nodes:
            internal_knowledge_parts.append("Relevant entities from memory:")
            for node in relevant_nodes[:5]:  # Limit to 5 most relevant
                props = ", ".join([f"{k}: {v}" for k, v in node.properties.items()][:3])
                internal_knowledge_parts.append(f"- {node.id} ({node.type}): {props}")

        if similar_contexts:
            internal_knowledge_parts.append("\\nSimilar past interactions:")
            for ctx in similar_contexts:
                internal_knowledge_parts.append(f"- Q: {ctx.query[:100]}...")
                internal_knowledge_parts.append(f"  A: {ctx.response[:100]}...")

        updated_context = context.copy()
        updated_context["internal_knowledge"] = "\\n".join(internal_knowledge_parts)
        updated_context["relevant_nodes"] = [node.id for node in relevant_nodes]

        return updated_context