"""Knowledge Graph Transform for TorchRL integration."""

import warnings
from typing import Dict, Any, Optional, List
import torch

try:
    from tensordict import TensorDict
    from torchrl.envs import Transform
    from torchrl.data import CompositeSpec, Unbounded
    TORCHRL_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"TorchRL dependencies not available: {e}")
    TORCHRL_AVAILABLE = False
    
    # Dummy base class for type hints
    class Transform:
        pass


class KnowledgeGraphTransform(Transform if TORCHRL_AVAILABLE else object):
    """
    Transform that adds Knowledge Graph operations as tools to TorchRL environment.
    
    This transform integrates SPARQL query capabilities and internal KG operations
    into the TorchRL tool framework while maintaining compatibility with existing
    RL-KG-Agent functionality.
    """
    
    def __init__(self, kg_loader, internal_kg=None, enable_sparql_tools: bool = True):
        """
        Initialize KG transform.
        
        Args:
            kg_loader: Knowledge graph loader for SPARQL operations
            internal_kg: Internal knowledge graph for dynamic storage
            enable_sparql_tools: Whether to enable SPARQL tool calls
        """
        if not TORCHRL_AVAILABLE:
            warnings.warn("TorchRL not available. Transform will operate in compatibility mode.")
            
        if TORCHRL_AVAILABLE:
            super().__init__()
            
        self.kg_loader = kg_loader
        self.internal_kg = internal_kg
        self.enable_sparql_tools = enable_sparql_tools
        
        # Tool execution history
        self.tool_history = []
    
    def _call(self, tensordict: TensorDict) -> TensorDict:
        """
        Process tensordict and handle KG tool execution.
        
        Args:
            tensordict: Input tensordict containing conversation state
            
        Returns:
            Enhanced tensordict with KG tool results
        """
        if not TORCHRL_AVAILABLE:
            return tensordict
            
        # Check if there are KG tool calls in the conversation
        if "history" in tensordict:
            history = tensordict["history"]
            
            # Look for tool calls in the latest interactions
            tool_calls = self._extract_kg_tool_calls(history)
            
            if tool_calls:
                # Execute KG tools and add results to tensordict
                tool_results = self._execute_kg_tools(tool_calls)
                tensordict = self._add_tool_results(tensordict, tool_results)
        
        return tensordict
    
    def _extract_kg_tool_calls(self, history) -> List[Dict[str, Any]]:
        """Extract KG-related tool calls from conversation history."""
        tool_calls = []
        
        # This would be enhanced to parse actual tool calls from LLM responses
        # For now, we provide a basic structure
        
        if hasattr(history, 'prompt') and history.prompt is not None:
            latest_content = str(history.prompt[-1].content) if len(history.prompt) > 0 else ""
            
            # Look for SPARQL tool patterns
            if "<tool>sparql" in latest_content.lower():
                tool_calls.append({
                    "type": "sparql_query",
                    "content": latest_content
                })
            
            # Look for internal KG tool patterns  
            if "<tool>store_knowledge" in latest_content.lower():
                tool_calls.append({
                    "type": "store_knowledge",
                    "content": latest_content
                })
                
        return tool_calls
    
    def _execute_kg_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute KG tool calls and return results."""
        results = []
        
        for tool_call in tool_calls:
            try:
                if tool_call["type"] == "sparql_query":
                    result = self._execute_sparql_tool(tool_call)
                elif tool_call["type"] == "store_knowledge":
                    result = self._execute_storage_tool(tool_call)
                else:
                    result = {
                        "success": False,
                        "error": f"Unknown tool type: {tool_call['type']}"
                    }
                
                results.append(result)
                
                # Add to tool history
                self.tool_history.append({
                    "tool_call": tool_call,
                    "result": result,
                    "timestamp": torch.time.time() if hasattr(torch, 'time') else None
                })
                
            except Exception as e:
                error_result = {
                    "success": False,
                    "error": str(e),
                    "tool_type": tool_call["type"]
                }
                results.append(error_result)
        
        return results
    
    def _execute_sparql_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SPARQL query tool."""
        if not self.enable_sparql_tools or not self.kg_loader:
            return {
                "success": False,
                "error": "SPARQL tools not enabled or KG loader not available"
            }
        
        content = tool_call["content"]
        
        # Extract SPARQL query from tool call (simplified parsing)
        # In a full implementation, this would use proper JSON parsing
        query_start = content.find("SELECT")
        if query_start == -1:
            query_start = content.find("ASK")
        if query_start == -1:
            query_start = content.find("CONSTRUCT")
            
        if query_start == -1:
            return {
                "success": False,
                "error": "No valid SPARQL query found in tool call"
            }
        
        # Extract query (simplified)
        query_end = content.find("}", query_start)
        if query_end != -1:
            sparql_query = content[query_start:query_end + 1].strip()
        else:
            sparql_query = content[query_start:].strip()
        
        try:
            # Validate and execute SPARQL
            if self.kg_loader.validate_sparql_syntax(sparql_query):
                results = self.kg_loader.execute_sparql(sparql_query)
                
                return {
                    "success": True,
                    "query": sparql_query,
                    "results": results,
                    "result_count": len(results),
                    "tool_type": "sparql_query"
                }
            else:
                return {
                    "success": False,
                    "error": "Invalid SPARQL syntax",
                    "query": sparql_query
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"SPARQL execution failed: {str(e)}",
                "query": sparql_query
            }
    
    def _execute_storage_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute knowledge storage tool."""
        if not self.internal_kg:
            return {
                "success": False,
                "error": "Internal KG not available"
            }
        
        try:
            # Extract information to store (simplified)
            content = tool_call["content"]
            
            # This would be enhanced with proper parsing
            # For now, store the tool call content as context
            context_id = self.internal_kg.add_context(
                query="Tool-based storage",
                response=content,
                entities=[],
                success=True
            )
            
            return {
                "success": True,
                "context_id": context_id,
                "tool_type": "store_knowledge"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Storage failed: {str(e)}"
            }
    
    def _add_tool_results(self, tensordict: TensorDict, tool_results: List[Dict[str, Any]]) -> TensorDict:
        """Add tool execution results to tensordict."""
        if not tool_results:
            return tensordict
        
        # Add tool results as metadata
        if "tool_results" not in tensordict:
            tensordict["tool_results"] = []
        
        # Convert to tensor-compatible format
        for result in tool_results:
            if TORCHRL_AVAILABLE:
                # Store as structured data
                tensordict["tool_results"].append(result)
        
        # Add summary metrics
        successful_tools = sum(1 for r in tool_results if r.get("success", False))
        tensordict["tool_success_count"] = torch.tensor([successful_tools], dtype=torch.long)
        tensordict["tool_total_count"] = torch.tensor([len(tool_results)], dtype=torch.long)
        
        return tensordict
    
    def transform_observation_spec(self, observation_spec: CompositeSpec) -> CompositeSpec:
        """Transform observation spec to include tool context."""
        if not TORCHRL_AVAILABLE:
            return observation_spec
            
        # Add tool-related observations
        observation_spec["tool_success_count"] = Unbounded(shape=(1,), dtype=torch.long)
        observation_spec["tool_total_count"] = Unbounded(shape=(1,), dtype=torch.long)
        
        return observation_spec
    
    def get_tool_history(self) -> List[Dict[str, Any]]:
        """Get history of tool executions."""
        return self.tool_history.copy()
    
    def clear_tool_history(self):
        """Clear tool execution history."""
        self.tool_history.clear()


# Compatibility function for non-TorchRL usage
def create_kg_tool_wrapper(kg_loader, internal_kg=None):
    """
    Create a simple wrapper for KG operations when TorchRL is not available.
    
    Args:
        kg_loader: Knowledge graph loader
        internal_kg: Internal knowledge graph
        
    Returns:
        Simple KG operations wrapper
    """
    class SimpleKGWrapper:
        def __init__(self, kg_loader, internal_kg):
            self.kg_loader = kg_loader
            self.internal_kg = internal_kg
        
        def execute_sparql(self, query: str) -> Dict[str, Any]:
            """Execute SPARQL query."""
            try:
                if self.kg_loader.validate_sparql_syntax(query):
                    results = self.kg_loader.execute_sparql(query)
                    return {"success": True, "results": results}
                else:
                    return {"success": False, "error": "Invalid SPARQL syntax"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        def store_context(self, query: str, response: str, entities: List[str] = None) -> Dict[str, Any]:
            """Store context in internal KG."""
            try:
                if self.internal_kg:
                    context_id = self.internal_kg.add_context(
                        query=query,
                        response=response,
                        entities=entities or [],
                        success=True
                    )
                    return {"success": True, "context_id": context_id}
                else:
                    return {"success": False, "error": "Internal KG not available"}
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    return SimpleKGWrapper(kg_loader, internal_kg)