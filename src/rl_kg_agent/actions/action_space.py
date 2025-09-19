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
    QUERY_MCP_THEN_RESPOND = 4    # Query MCP server for biomedical info, then respond
    # STORE_AND_RESPOND = 5         # Store to internal KG, then respond with LLM (DISABLED - has errors)


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
            logging.info(f"🎯 ACTION: RESPOND_DIRECTLY - Processing query: '{query}'")
            
            # Create messages for LLM
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question directly and concisely."},
                {"role": "user", "content": query}
            ]

            # Add internal knowledge context if available
            if internal_knowledge:
                messages[0]["content"] += f"\\n\\nRelevant context from memory: {internal_knowledge}"
                logging.info(f"🎯 Added internal knowledge context ({len(internal_knowledge)} chars)")

            # Generate response
            response = self.llm_client.generate_response(messages)
            
            logging.info(f"🎯 RESPOND_DIRECTLY completed successfully")

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
            error_msg = f"Direct response failed: {e}"
            logging.error(f"🚨 RESPOND_DIRECTLY failed: {error_msg}")
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
            results = []

            logger.info(f"Starting KG query for user question: '{query}'")
            logger.info(f"Detected entities: {entities}")

            # Try to generate SPARQL query
            if hasattr(self.sparql_generator, 'generate_sparql_from_context'):
                # Use LLM-powered SPARQL generation
                logger.info("Using LLM-powered SPARQL generation")
                sparql_query = self.sparql_generator.generate_sparql_from_context(context)
            else:
                # Fallback to simple entity-based queries
                logger.info("Using simple entity-based SPARQL generation")
                sparql_query = self._generate_simple_query(query, entities)

            if sparql_query:
                logger.info(f"Generated SPARQL query:\n{sparql_query}")

                if self.kg_loader.validate_sparql_syntax(sparql_query):
                    logger.info("SPARQL query syntax is valid, executing...")
                    results = self.kg_loader.execute_sparql(sparql_query)
                    logger.info(f"KG query returned {len(results)} results")

                    if results:
                        kg_info = self._format_kg_results(results)
                        logger.info(f"Formatted KG results for LLM context:\n{kg_info}")
                    else:
                        logger.info("No results found from KG query")
                else:
                    logger.warning(f"Invalid SPARQL syntax, skipping KG query: {sparql_query}")
            else:
                logger.info("No SPARQL query generated, proceeding with direct response")

            # Now generate response with KG information
            enhanced_context = context.get("internal_knowledge", "")
            if kg_info:
                enhanced_context = f"{enhanced_context}\\n\\nKnowledge Graph Information: {kg_info}".strip()
                logger.info("KG results successfully added to LLM context for final response")

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to a knowledge base. Answer the user's question using the provided information."
                },
                {"role": "user", "content": query}
            ]

            if enhanced_context:
                messages[0]["content"] += f"\\n\\nAvailable information: {enhanced_context}"
                logger.info(f"Final LLM context includes: internal knowledge + KG results (total context length: {len(enhanced_context)} chars)")
            else:
                logger.info("No additional context provided to LLM, using direct response mode")

            response = self.llm_client.generate_response(messages)

            logger.info(f"Generated final response using {'KG-enhanced' if kg_info else 'direct'} mode")
            logger.info(f"Response length: {len(response)} characters")

            return ActionResult(
                success=True,
                response=response,
                metadata={
                    "action": "kg_query_response",
                    "sparql_query": sparql_query,
                    "kg_results_found": len(results),
                    "used_kg_info": bool(kg_info),
                    "context_length": len(enhanced_context) if enhanced_context else 0
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
            # Fallback: try to extract key terms from the query for broad search
            query_words = [word.strip('.,!?;:"()[]').lower() for word in query.split() if len(word.strip('.,!?;:"()[]')) > 3]
            if not query_words:
                return None

            # Create a broad search query using key terms
            search_term = query_words[0].capitalize()  # Use the first meaningful word
            logger.info(f"No entities detected, using fallback search for term: '{search_term}'")

            return f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?s ?p ?o WHERE {{
                {{
                    ?s rdfs:label ?label .
                    FILTER(CONTAINS(LCASE(?label), "{search_term.lower()}"))
                    ?s ?p ?o
                }}
                UNION
                {{
                    ?o rdfs:label ?label .
                    FILTER(CONTAINS(LCASE(?label), "{search_term.lower()}"))
                    ?s ?p ?o
                }}
            }} LIMIT 15
            """

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


class QueryMCPThenRespondAction(BaseAction):
    """Query MCP server for biomedical information, then respond with enhanced context."""

    def __init__(self, mcp_manager, llm_client):
        super().__init__(ActionType.QUERY_MCP_THEN_RESPOND)
        self.mcp_manager = mcp_manager
        self.llm_client = llm_client

    def execute(self, context: Dict[str, Any], **kwargs) -> ActionResult:
        """Query MCP server then provide enhanced response."""
        query = context.get("query", "")
        entities = context.get("entities", [])

        try:
            logging.info(f"🎯 ACTION: QUERY_MCP_THEN_RESPOND - Processing query: '{query}'")
            
            # Determine which MCP tool to use based on query type
            tool_name, arguments = self._select_mcp_tool(query, entities, context)
            
            if not tool_name:
                logging.warning(f"🎯 No suitable MCP tool found for query: '{query}'")
                return self._fallback_response(query, context, "No suitable MCP tool found for query")

            logging.info(f"🎯 Selected MCP tool: '{tool_name}' with args: {arguments}")
            
            # Call the MCP server asynchronously
            import asyncio
            mcp_response = asyncio.run(self.mcp_manager.call_tool(
                "unified_biomedical", tool_name, arguments
            ))
            
            if not mcp_response.success:
                error_msg = f"MCP query failed: {mcp_response.error}"
                logging.error(f"🚨 MCP tool call failed: {mcp_response.error}")
                return self._fallback_response(query, context, error_msg)
            
            # Generate response with MCP information
            mcp_info = mcp_response.result
            enhanced_context = context.get("internal_knowledge", "")
            
            if mcp_info:
                enhanced_context = f"{enhanced_context}\n\nBiomedical Information from MCP Server:\n{mcp_info}".strip()
                logging.info(f"🎯 MCP results added to context ({len(mcp_info)} chars)")
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a knowledgeable biomedical assistant with access to scientific literature and entity databases. Answer the user's question using the provided biomedical information."
                },
                {"role": "user", "content": query}
            ]
            
            if enhanced_context:
                messages[0]["content"] += f"\n\nAvailable information: {enhanced_context}"
            
            response = self.llm_client.generate_response(messages)
            
            # Extract entities from MCP response if possible
            entities_discovered = self._extract_entities_from_mcp_response(mcp_info)
            
            return ActionResult(
                success=True,
                response=response,
                metadata={
                    "action": "mcp_query_response",
                    "mcp_tool": tool_name,
                    "mcp_success": True,
                    "mcp_execution_time": mcp_response.execution_time,
                    "context_length": len(enhanced_context) if enhanced_context else 0,
                    "entities_found": len(entities_discovered)
                },
                entities_discovered=entities_discovered,
                confidence=0.9 if mcp_info else 0.6
            )

        except Exception as e:
            logger.error(f"MCP query and response failed: {e}")
            return self._fallback_response(query, context, str(e))

    def _select_mcp_tool(self, query: str, entities: List[str], context: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Select appropriate MCP tool and arguments based on query characteristics.
        
        Returns:
            Tuple of (tool_name, arguments) or (None, None) if no suitable tool
        """
        query_lower = query.lower()
        
        # Check for question-answering patterns
        qa_keywords = ["what", "how", "why", "explain", "describe", "tell me about"]
        is_question = any(keyword in query_lower for keyword in qa_keywords)
        
        # Check for search/lookup patterns  
        search_keywords = ["find", "search", "look up", "information about", "papers about"]
        is_search = any(keyword in query_lower for keyword in search_keywords)
        
        # Check for entity lookup patterns
        entity_keywords = ["entity", "id:", "details about", "information on"]
        is_entity_lookup = any(keyword in query_lower for keyword in entity_keywords)
        
        # Select tool based on query type
        if is_entity_lookup and entities:
            # Use entity details tool for specific entity lookups
            return "get_entity_details", {"entity_id": entities[0]}
        
        elif is_question and len(query.split()) > 3:
            # Use RAG answer for complex questions
            return "get_rag_answer", {
                "question": query,
                "candidates": 20,
                "include_conversation_details": True
            }
        
        elif is_search or entities or len(query.split()) >= 2:
            # Use semantic search for general searches
            return "medline_semantic_search", {"query": query}
        
        else:
            # No suitable tool found
            return None, None

    def _extract_entities_from_mcp_response(self, mcp_response: str) -> List[str]:
        """Extract entity names from MCP response text.
        
        Args:
            mcp_response: Text response from MCP server
            
        Returns:
            List of entity names found in the response
        """
        if not mcp_response:
            return []
        
        entities = []
        
        # Look for patterns like "1. Entity Name" or "• Entity Name"
        import re
        entity_patterns = [
            r'^\d+\.\s+([^(]+?)(?:\s*\(|$)',  # Numbered list: "1. Entity Name (ID)"
            r'^•\s+([^(]+?)(?:\s*\(|$)',      # Bullet list: "• Entity Name (ID)"
            r'Entity.*?:\s*([^(]+?)(?:\s*\(|$)'  # "Entity: Name (ID)"
        ]
        
        for line in mcp_response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            for pattern in entity_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    entity_name = match.group(1).strip()
                    if entity_name and len(entity_name) > 1:
                        entities.append(entity_name)
                    break
        
        return list(set(entities))  # Remove duplicates

    def _fallback_response(self, query: str, context: Dict[str, Any], error_msg: str) -> ActionResult:
        """Generate fallback response when MCP query fails."""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question directly."},
                {"role": "user", "content": query}
            ]
            
            # Add internal knowledge if available
            internal_knowledge = context.get("internal_knowledge", "")
            if internal_knowledge:
                messages[0]["content"] += f"\n\nRelevant context from memory: {internal_knowledge}"
            
            response = self.llm_client.generate_response(messages)
            
            return ActionResult(
                success=True,
                response=response,
                metadata={
                    "action": "mcp_query_response",
                    "mcp_success": False,
                    "fallback": True,
                    "error": error_msg,
                    "used_internal_knowledge": bool(internal_knowledge)
                },
                confidence=0.5
            )
            
        except Exception as fallback_error:
            logger.error(f"Fallback response also failed: {fallback_error}")
            return ActionResult(
                success=False,
                response="I'm sorry, I couldn't find information to answer your question.",
                metadata={
                    "action": "mcp_query_response",
                    "mcp_success": False,
                    "fallback_error": str(fallback_error),
                    "original_error": error_msg
                },
                confidence=0.0
            )

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Applicable for biomedical queries, scientific questions, or entity lookups."""
        query = context.get("query", "").lower()
        entities = context.get("entities", [])
        
        # Biomedical/scientific keywords
        biomedical_keywords = [
            "gene", "protein", "disease", "drug", "medicine", "clinical", "medical",
            "biology", "biochemistry", "pharmacology", "pathology", "anatomy",
            "pubmed", "medline", "literature", "research", "study", "paper",
            "molecule", "compound", "treatment", "therapy", "diagnosis",
            "biomarker", "pathway", "mechanism", "interaction"
        ]
        
        # Question patterns that benefit from literature search
        question_patterns = [
            "what is", "what are", "how does", "why does", "what causes",
            "research on", "studies about", "papers about", "information about",
            "find papers", "scientific evidence", "clinical trials"
        ]
        
        # Check for biomedical content
        has_biomedical_content = any(keyword in query for keyword in biomedical_keywords)
        
        # Check for research/literature patterns
        has_research_pattern = any(pattern in query for pattern in question_patterns)
        
        # Check for scientific/factual questions
        factual_keywords = ["what", "how", "why", "where", "when", "which"]
        is_factual_question = any(keyword in query for keyword in factual_keywords)
        
        # More likely to be applicable if:
        # - Contains biomedical terms
        # - Is a research/literature query
        # - Is a factual question with some complexity
        # - Has entities that might be biomedical
        
        return (has_biomedical_content or 
                has_research_pattern or 
                (is_factual_question and len(query.split()) > 3) or
                bool(entities))

    def get_description(self) -> str:
        return "Query biomedical MCP server for literature search and entity information, then provide informed response"