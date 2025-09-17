"""TorchRL Action Executor that enhances discrete action execution with tool capabilities."""

import warnings
from typing import Dict, Any, Optional, List
from enum import IntEnum

try:
    import torch
    from tensordict import TensorDict
    TORCH_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"PyTorch/TorchRL dependencies not available: {e}")
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class TensorDict:
        pass

from ..actions.action_space import ActionType, ActionResult


class TorchRLActionExecutor:
    """
    Enhanced action executor that maintains discrete action space compatibility
    while adding TorchRL tool integration capabilities.
    
    This executor serves as a bridge between:
    - PPO's discrete action selection (0-4)
    - Enhanced tool-enabled action execution
    - TorchRL's tensor-based state management
    """
    
    def __init__(
        self,
        action_manager,
        kg_loader=None,
        internal_kg=None,
        llm_client=None,
        enable_tool_enhancement: bool = True,
        tool_timeout: float = 30.0
    ):
        """
        Initialize TorchRL action executor.
        
        Args:
            action_manager: Existing action manager for base functionality
            kg_loader: Knowledge graph loader for enhanced SPARQL operations
            internal_kg: Internal knowledge graph for enhanced storage
            llm_client: LLM client for enhanced response generation
            enable_tool_enhancement: Whether to enable tool enhancements
            tool_timeout: Timeout for tool operations
        """
        self.action_manager = action_manager
        self.kg_loader = kg_loader
        self.internal_kg = internal_kg
        self.llm_client = llm_client
        self.enable_tool_enhancement = enable_tool_enhancement
        self.tool_timeout = tool_timeout
        
        # Execution history for analysis
        self.execution_history = []
        
        # Performance metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "tool_enhanced_executions": 0,
            "average_execution_time": 0.0
        }
    
    def execute_action(
        self, 
        action_id: int, 
        context: Dict[str, Any],
        tensordict: Optional[TensorDict] = None
    ) -> ActionResult:
        """
        Execute action with optional TorchRL tool enhancement.
        
        Args:
            action_id: Discrete action ID (0-4)
            context: Execution context
            tensordict: Optional TensorDict for enhanced state management
            
        Returns:
            Enhanced ActionResult with tool integration
        """
        import time
        start_time = time.time()
        
        # Validate action ID
        if not 0 <= action_id <= 4:
            raise ValueError(f"Invalid action ID {action_id}. Must be in range [0, 4]")
        
        # Convert to ActionType
        action_type = ActionType(action_id)
        
        # Track execution
        self.metrics["total_executions"] += 1
        
        try:
            # Execute based on action type with possible tool enhancement
            if action_type == ActionType.RESPOND_DIRECTLY:
                result = self._execute_respond_directly(context, tensordict)
            elif action_type == ActionType.QUERY_KG_THEN_RESPOND:
                result = self._execute_query_kg_then_respond(context, tensordict)
            elif action_type == ActionType.PLAN_THEN_RESPOND:
                result = self._execute_plan_then_respond(context, tensordict)
            elif action_type == ActionType.ASK_CLARIFYING_QUESTION:
                result = self._execute_ask_clarifying_question(context, tensordict)
            elif action_type == ActionType.STORE_AND_RESPOND:
                result = self._execute_store_and_respond(context, tensordict)
            else:
                # Fallback to basic action manager
                result = self.action_manager.execute_action(action_type, context)
            
            # Update metrics
            if result.success:
                self.metrics["successful_executions"] += 1
            
            # Track tool enhancement usage
            if result.metadata.get("tool_enhanced", False):
                self.metrics["tool_enhanced_executions"] += 1
            
            # Update timing
            execution_time = time.time() - start_time
            self.metrics["average_execution_time"] = (
                (self.metrics["average_execution_time"] * (self.metrics["total_executions"] - 1) + execution_time) /
                self.metrics["total_executions"]
            )
            
            # Store execution history
            self.execution_history.append({
                "action_type": action_type,
                "success": result.success,
                "execution_time": execution_time,
                "tool_enhanced": result.metadata.get("tool_enhanced", False),
                "timestamp": start_time
            })
            
            # Limit history size
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]
            
            return result
            
        except Exception as e:
            # Handle execution errors gracefully
            error_result = ActionResult(
                success=False,
                response=f"Action execution failed: {str(e)}",
                metadata={
                    "action_type": action_type.name,
                    "error": str(e),
                    "tool_enhanced": False
                },
                confidence=0.0
            )
            
            execution_time = time.time() - start_time
            self.execution_history.append({
                "action_type": action_type,
                "success": False,
                "execution_time": execution_time,
                "error": str(e),
                "timestamp": start_time
            })
            
            return error_result
    
    def _execute_respond_directly(self, context: Dict[str, Any], tensordict: Optional[TensorDict]) -> ActionResult:
        """Execute direct response with possible tool enhancement."""
        # Try enhanced execution first
        if self.enable_tool_enhancement and self.llm_client:
            try:
                # Enhanced LLM response with tool context
                query = context.get("query", "")
                internal_knowledge = context.get("internal_knowledge", "")
                
                # Check if we can enhance with web search or other tools
                if self._should_enhance_with_tools(query, context):
                    return self._execute_tool_enhanced_response(query, context, tensordict)
                
                # Enhanced LLM response with better context
                messages = self._build_enhanced_messages(query, internal_knowledge, context)
                response = self.llm_client.generate_response(messages)
                
                return ActionResult(
                    success=True,
                    response=response,
                    metadata={
                        "action": "respond_directly",
                        "tool_enhanced": True,
                        "enhancement_type": "context_enhanced"
                    },
                    confidence=0.85
                )
                
            except Exception as e:
                warnings.warn(f"Enhanced response failed, falling back to basic: {e}")
        
        # Fallback to basic action manager
        return self.action_manager.execute_action(ActionType.RESPOND_DIRECTLY, context)
    
    def _execute_query_kg_then_respond(self, context: Dict[str, Any], tensordict: Optional[TensorDict]) -> ActionResult:
        """Execute KG query with enhanced SPARQL capabilities."""
        if self.enable_tool_enhancement and self.kg_loader:
            try:
                # Enhanced KG querying with better SPARQL generation
                query = context.get("query", "")
                entities = context.get("entities", [])
                
                # Generate enhanced SPARQL query
                sparql_query = self._generate_enhanced_sparql(query, entities, context)
                
                if sparql_query:
                    # Execute with enhanced error handling and caching
                    results = self._execute_enhanced_sparql(sparql_query)
                    
                    if results:
                        # Enhanced response generation with KG results
                        response = self._generate_enhanced_kg_response(query, results, context)
                        
                        return ActionResult(
                            success=True,
                            response=response,
                            metadata={
                                "action": "query_kg_then_respond",
                                "tool_enhanced": True,
                                "sparql_query": sparql_query,
                                "results_count": len(results),
                                "enhancement_type": "enhanced_sparql"
                            },
                            entities_discovered=self._extract_entities_from_results(results),
                            confidence=0.9
                        )
                
            except Exception as e:
                warnings.warn(f"Enhanced KG query failed, falling back to basic: {e}")
        
        # Fallback to basic action manager
        return self.action_manager.execute_action(ActionType.QUERY_KG_THEN_RESPOND, context)
    
    def _execute_plan_then_respond(self, context: Dict[str, Any], tensordict: Optional[TensorDict]) -> ActionResult:
        """Execute planning with enhanced reasoning capabilities."""
        if self.enable_tool_enhancement and self.llm_client:
            try:
                # Enhanced planning with tool awareness
                query = context.get("query", "")
                
                # Generate tool-aware plan
                plan = self._generate_enhanced_plan(query, context)
                
                # Execute plan with tool integration
                response = self._execute_plan_with_tools(plan, query, context)
                
                return ActionResult(
                    success=True,
                    response=response,
                    metadata={
                        "action": "plan_then_respond",
                        "tool_enhanced": True,
                        "plan": plan,
                        "enhancement_type": "tool_aware_planning"
                    },
                    confidence=0.88
                )
                
            except Exception as e:
                warnings.warn(f"Enhanced planning failed, falling back to basic: {e}")
        
        # Fallback to basic action manager
        return self.action_manager.execute_action(ActionType.PLAN_THEN_RESPOND, context)
    
    def _execute_ask_clarifying_question(self, context: Dict[str, Any], tensordict: Optional[TensorDict]) -> ActionResult:
        """Execute clarifying question with enhanced context awareness."""
        if self.enable_tool_enhancement and self.llm_client:
            try:
                # Enhanced clarifying questions with context awareness
                query = context.get("query", "")
                history = context.get("history", [])
                
                # Generate context-aware clarifying question
                clarifying_question = self._generate_enhanced_clarifying_question(query, history, context)
                
                return ActionResult(
                    success=True,
                    response=clarifying_question,
                    metadata={
                        "action": "ask_clarifying_question",
                        "tool_enhanced": True,
                        "enhancement_type": "context_aware"
                    },
                    confidence=0.75
                )
                
            except Exception as e:
                warnings.warn(f"Enhanced clarifying question failed, falling back to basic: {e}")
        
        # Fallback to basic action manager
        return self.action_manager.execute_action(ActionType.ASK_CLARIFYING_QUESTION, context)
    
    def _execute_store_and_respond(self, context: Dict[str, Any], tensordict: Optional[TensorDict]) -> ActionResult:
        """Execute storage with enhanced knowledge organization."""
        if self.enable_tool_enhancement and self.internal_kg:
            try:
                # Enhanced storage with better knowledge organization
                query = context.get("query", "")
                entities = context.get("entities", [])
                
                # Enhanced knowledge extraction and storage
                knowledge_items = self._extract_enhanced_knowledge(query, context)
                storage_results = self._store_enhanced_knowledge(knowledge_items)
                
                # Generate response with storage confirmation
                response = self._generate_storage_response(query, storage_results, context)
                
                return ActionResult(
                    success=True,
                    response=response,
                    metadata={
                        "action": "store_and_respond",
                        "tool_enhanced": True,
                        "items_stored": len(storage_results),
                        "enhancement_type": "enhanced_storage"
                    },
                    entities_discovered=entities,
                    confidence=0.8
                )
                
            except Exception as e:
                warnings.warn(f"Enhanced storage failed, falling back to basic: {e}")
        
        # Fallback to basic action manager
        return self.action_manager.execute_action(ActionType.STORE_AND_RESPOND, context)
    
    def _should_enhance_with_tools(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if query should be enhanced with external tools."""
        # Simple heuristics for tool enhancement
        query_lower = query.lower()
        
        # Current/recent information needs
        current_indicators = ["current", "recent", "latest", "now", "today", "this year"]
        if any(indicator in query_lower for indicator in current_indicators):
            return True
        
        # Research/academic queries
        research_indicators = ["research", "study", "publication", "paper", "article"]
        if any(indicator in query_lower for indicator in research_indicators):
            return True
        
        # Complex analytical queries
        if len(query.split()) > 10 and any(word in query_lower for word in ["analyze", "compare", "explain"]):
            return True
        
        return False
    
    def _build_enhanced_messages(self, query: str, internal_knowledge: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build enhanced message context for LLM."""
        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable assistant with access to structured knowledge and tools. Provide accurate, well-reasoned responses."
            }
        ]
        
        # Add context if available
        if internal_knowledge:
            messages[0]["content"] += f"\\n\\nRelevant context: {internal_knowledge}"
        
        # Add query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    # Placeholder methods for enhanced functionality
    # These would be fully implemented with actual tool integrations
    
    def _execute_tool_enhanced_response(self, query: str, context: Dict[str, Any], tensordict: Optional[TensorDict]) -> ActionResult:
        """Execute response with external tool enhancement (placeholder)."""
        # This would integrate with web search, APIs, etc.
        response = f"Enhanced response for: {query} (tool integration placeholder)"
        return ActionResult(
            success=True,
            response=response,
            metadata={"tool_enhanced": True, "enhancement_type": "external_tools"},
            confidence=0.7
        )
    
    def _generate_enhanced_sparql(self, query: str, entities: List[str], context: Dict[str, Any]) -> Optional[str]:
        """Generate enhanced SPARQL query (placeholder)."""
        # This would use improved SPARQL generation logic
        if entities:
            entity = entities[0]
            return f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?s ?p ?o WHERE {{
                ?s rdfs:label "{entity}" .
                ?s ?p ?o
            }} LIMIT 10
            """
        return None
    
    def _execute_enhanced_sparql(self, sparql_query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL with enhanced error handling."""
        if self.kg_loader and self.kg_loader.validate_sparql_syntax(sparql_query):
            return self.kg_loader.execute_sparql(sparql_query)
        return []
    
    def _generate_enhanced_kg_response(self, query: str, results: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Generate enhanced response from KG results."""
        if not results:
            return "No relevant information found in the knowledge graph."
        
        # Format results for response
        formatted_results = []
        for result in results[:3]:  # Limit to top 3 results
            items = [f"{k}: {v}" for k, v in result.items()]
            formatted_results.append(", ".join(items))
        
        return f"Based on the knowledge graph: {'; '.join(formatted_results)}"
    
    def _extract_entities_from_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract entities from SPARQL results."""
        entities = set()
        for result in results:
            for value in result.values():
                if isinstance(value, str) and len(value) > 2:
                    entities.add(value)
        return list(entities)[:10]  # Limit to 10 entities
    
    def _generate_enhanced_plan(self, query: str, context: Dict[str, Any]) -> str:
        """Generate enhanced planning (placeholder)."""
        return f"Enhanced plan for: {query}"
    
    def _execute_plan_with_tools(self, plan: str, query: str, context: Dict[str, Any]) -> str:
        """Execute plan with tool integration (placeholder)."""
        return f"Response based on plan: {plan}"
    
    def _generate_enhanced_clarifying_question(self, query: str, history: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Generate enhanced clarifying question."""
        if len(query.split()) < 3:
            return "Could you provide more details about what specifically you'd like to know?"
        return f"To better answer your question about '{query}', could you clarify what aspect interests you most?"
    
    def _extract_enhanced_knowledge(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract knowledge items for storage."""
        return [{"query": query, "context": context}]
    
    def _store_enhanced_knowledge(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Store knowledge items with enhanced organization."""
        results = []
        for item in knowledge_items:
            if self.internal_kg:
                context_id = self.internal_kg.add_context(
                    query=item.get("query", ""),
                    response="Stored knowledge",
                    entities=[],
                    success=True
                )
                results.append({"stored": True, "context_id": context_id})
        return results
    
    def _generate_storage_response(self, query: str, storage_results: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Generate response for storage action."""
        successful_stores = sum(1 for r in storage_results if r.get("stored", False))
        response = f"I've processed your question about '{query}' and stored {successful_stores} knowledge items for future reference."
        return response
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics."""
        if self.metrics["total_executions"] == 0:
            return {"message": "No executions recorded"}
        
        success_rate = self.metrics["successful_executions"] / self.metrics["total_executions"]
        tool_enhancement_rate = self.metrics["tool_enhanced_executions"] / self.metrics["total_executions"]
        
        return {
            "total_executions": self.metrics["total_executions"],
            "success_rate": success_rate,
            "tool_enhancement_rate": tool_enhancement_rate,
            "average_execution_time": self.metrics["average_execution_time"],
            "recent_performance": self._get_recent_performance()
        }
    
    def _get_recent_performance(self) -> Dict[str, Any]:
        """Get performance metrics for recent executions."""
        if len(self.execution_history) < 10:
            return {"message": "Insufficient execution history"}
        
        recent = self.execution_history[-50:]  # Last 50 executions
        recent_success_rate = sum(1 for e in recent if e["success"]) / len(recent)
        recent_tool_rate = sum(1 for e in recent if e.get("tool_enhanced", False)) / len(recent)
        recent_avg_time = sum(e["execution_time"] for e in recent) / len(recent)
        
        return {
            "recent_success_rate": recent_success_rate,
            "recent_tool_enhancement_rate": recent_tool_rate,
            "recent_average_time": recent_avg_time,
            "sample_size": len(recent)
        }
    
    def reset_metrics(self):
        """Reset execution metrics."""
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "tool_enhanced_executions": 0,
            "average_execution_time": 0.0
        }
        self.execution_history.clear()