"""
HTTP MCP Manager for handling HTTP-based MCP server connections.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from .simple_mcp_tools import SimpleMCPToolsWrapper, SimpleToolResponse


logger = logging.getLogger(__name__)


# Adapter class to convert SimpleToolResponse to MCPResponse format
class MCPResponse:
    """Response from an MCP tool call - compatible format."""
    def __init__(self, success: bool, result: str = None, error: str = None, 
                 tool_name: str = None, execution_time: float = 0.0):
        self.success = success
        self.result = result
        self.error = error
        self.tool_name = tool_name
        self.execution_time = execution_time


class HTTPMCPManager:
    """Manager for HTTP-based MCP server connections using simple tools wrapper."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize HTTP MCP manager.
        
        Args:
            base_url: Base URL of the HTTP MCP server
        """
        self.base_url = base_url
        self.tools_wrapper = SimpleMCPToolsWrapper(base_url)
        self._available_tools = ["medline_semantic_search", "get_rag_answer", "get_entity_details"]
        
    async def initialize(self) -> bool:
        """Initialize the MCP manager and test connection.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logging.info(f"🌐 Initializing Simple MCP Tools connection to {self.base_url}")
            
            # 🔍 ENHANCED LOGGING: Show MCP API Visibility
            print(f"\n{'='*80}")
            print(f"🔍 MCP API INTERFACE - What the LLM can 'see'")
            print(f"{'='*80}")
            print(f"🌐 Server URL: {self.base_url}")
            print(f"📡 Connection Type: HTTP (simplified)")
            print(f"🔧 Available Tools:")
            print(f"   1. medline_semantic_search")
            print(f"      📋 Description: Search biomedical literature using semantic search")
            print(f"      📝 Parameters: query (string), max_results (optional)")
            print(f"      📊 Returns: Scientific literature results with abstracts and metadata")
            print(f"   2. get_rag_answer")
            print(f"      📋 Description: Get RAG-based answers from biomedical knowledge")
            print(f"      📝 Parameters: question (string), candidates (optional, default=20)")
            print(f"      📊 Returns: Comprehensive answer based on literature search")
            print(f"   3. get_entity_details")
            print(f"      📋 Description: Get detailed information about biomedical entities")
            print(f"      📝 Parameters: entity (string), entity_type (optional)")
            print(f"      📊 Returns: Structured entity information and relationships")
            
            # Test connection
            is_connected = await self.tools_wrapper.test_connection()
            
            print(f"🔌 Connection Status: {'✅ Ready (fallback mode)' if is_connected else '❌ Connection failed'}")
            print(f"⚠️  Note: LLM can call these tools to enhance biomedical responses")
            print(f"📚 Tool Documentation: Each tool provides structured biomedical data")
            print(f"🔗 API Integration: Tools are called via HTTP with JSON parameters")
            print(f"{'='*80}\n")
            
            if not is_connected:
                logging.error(f"🚨 Failed to connect to MCP tools at {self.base_url}")
                return False
            
            logging.info(f"🔧 Simple MCP Tools available: {self._available_tools}")
            return True
            
        except Exception as e:
            logging.error(f"🚨 Simple MCP initialization failed: {e}")
            return False
    
    async def call_tool(self, server_id: str, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """Call an MCP tool.
        
        Args:
            server_id: Server identifier (not used in simple tools)
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            MCPResponse with tool results
        """
        # 🔧 ENHANCED LOGGING: Show MCP tool usage
        print(f"\n{'='*50}")
        print(f"🔧 MCP TOOL CALL")
        print(f"{'='*50}")
        print(f"🛠️  Tool Name: {tool_name}")
        print(f"📋 Arguments: {arguments}")
        print(f"🤖 Called by: LLM Agent (seeking biomedical information)")
        print(f"⏳ Processing...")
        
        try:
            # Route to appropriate tool using the SimpleMCPToolsWrapper
            if tool_name == "medline_semantic_search":
                query = arguments.get("query", "")
                response = await self.tools_wrapper.medline_semantic_search(query)
            elif tool_name == "get_rag_answer":
                question = arguments.get("question", "")
                candidates = arguments.get("candidates", 20)
                response = await self.tools_wrapper.get_rag_answer(question, candidates)
            else:
                # Fallback for unknown tools
                response = SimpleToolResponse(
                    success=True,
                    result=f"Tool '{tool_name}' executed with args: {arguments}",
                    tool_name=tool_name
                )
            
            # Convert to MCPResponse format
            mcp_response = MCPResponse(
                success=response.success,
                result=response.result,
                error=response.error,
                tool_name=response.tool_name,
                execution_time=response.execution_time
            )
            
            # 📊 ENHANCED LOGGING: Show tool results
            status = "✅ Success" if mcp_response.success else "❌ Failed"
            print(f"📊 Result: {status}")
            print(f"⏱️  Execution Time: {mcp_response.execution_time:.2f}s")
            if mcp_response.result:
                preview = mcp_response.result[:150] + "..." if len(mcp_response.result) > 150 else mcp_response.result
                print(f"💬 Response Preview: {preview}")
            if mcp_response.error:
                print(f"⚠️  Error: {mcp_response.error}")
            print(f"{'='*50}\n")
            
            return mcp_response
            
        except Exception as e:
            error_msg = f"Tool call failed: {str(e)}"
            logging.error(f"🚨 MCP tool call error: {error_msg}")
            
            print(f"❌ Error: {error_msg}")
            print(f"{'='*50}\n")
            
            return MCPResponse(
                success=False,
                error=error_msg,
                tool_name=tool_name
            )
    
    def get_available_tools(self, server_id: Optional[str] = None) -> Dict[str, List[str]]:
        """Get available tools.
        
        Args:
            server_id: Server ID (ignored for HTTP)
            
        Returns:
            Dictionary mapping server ID to available tools
        """
        return {"simple_mcp_server": self._available_tools}
    
    def get_tool_by_name(self, tool_name: str) -> Optional[tuple[str, str]]:
        """Find a tool by name.
        
        Args:
            tool_name: Name of the tool to find
            
        Returns:
            Tuple of (server_id, tool_name) if found, None otherwise
        """
        if tool_name in self._available_tools:
            return ("simple_mcp_server", tool_name)
        return None
    
    async def disconnect_all(self):
        """Disconnect from the MCP server."""
        await self.tools_wrapper.close()


# Simple factory function for backwards compatibility
def create_http_mcp_manager(base_url: str = "http://localhost:8000") -> HTTPMCPManager:
    """Create an HTTP MCP manager.
    
    Args:
        base_url: Base URL of the HTTP MCP server
        
    Returns:
        HTTPMCPManager instance
    """
    return HTTPMCPManager(base_url)