"""
HTTP MCP Client for connecting to MCP servers running in HTTP mode.
"""

import asyncio
import json
import logging
import httpx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class MCPResponse:
    """Response from an MCP tool call."""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    tool_name: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class HTTPMCPClient:
    """HTTP-based MCP client for connecting to servers running in HTTP mode."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize HTTP MCP client.
        
        Args:
            base_url: Base URL of the MCP server
        """
        self.base_url = base_url.rstrip('/')
        self._request_id = 0
        self._session = None
        
    async def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            self._session = httpx.AsyncClient(timeout=30.0)
        return self._session
    
    def _get_next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """Call a tool on the HTTP MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            MCPResponse with the result
        """
        import time
        start_time = time.time()
        
        try:
            logging.info(f"🔧 HTTP MCP Tool Call: {tool_name} with args: {arguments}")
            
            session = await self._get_session()
            
            # Prepare MCP JSON-RPC request
            request_data = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Send request to HTTP MCP server
            response = await session.post(
                f"{self.base_url}/mcp",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            execution_time = time.time() - start_time
            
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logging.error(f"🚨 HTTP MCP Error: {error_msg}")
                return MCPResponse(
                    success=False,
                    error=error_msg,
                    tool_name=tool_name,
                    execution_time=execution_time
                )
            
            response_data = response.json()
            
            if "error" in response_data:
                error_msg = f"MCP tool error: {response_data['error']}"
                logging.error(f"🚨 HTTP MCP Tool Error: {tool_name} failed with: {error_msg}")
                return MCPResponse(
                    success=False,
                    error=error_msg,
                    tool_name=tool_name,
                    execution_time=execution_time
                )
            
            result = response_data.get("result", {})
            
            # Extract content from MCP response
            if isinstance(result, dict):
                content = result.get("content", [])
                if isinstance(content, list) and content:
                    # Extract text from content blocks
                    text_content = content[0].get("text", "") if content else ""
                else:
                    text_content = str(result)
            else:
                text_content = str(result)
            
            logging.info(f"🔧 HTTP MCP Tool Success: {tool_name} returned {len(text_content)} chars in {execution_time:.2f}s")
            
            return MCPResponse(
                success=True,
                result=text_content,
                tool_name=tool_name,
                execution_time=execution_time,
                metadata=result.get("_meta", {}) if isinstance(result, dict) else {}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"HTTP MCP request failed: {str(e)}"
            logging.error(f"🚨 HTTP MCP Exception: {tool_name} - {error_msg}")
            return MCPResponse(
                success=False,
                error=error_msg,
                tool_name=tool_name,
                execution_time=execution_time
            )
    
    async def list_tools(self) -> List[str]:
        """List available tools from the HTTP MCP server.
        
        Returns:
            List of tool names
        """
        try:
            session = await self._get_session()
            
            request_data = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/list"
            }
            
            response = await session.post(
                f"{self.base_url}/mcp",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logging.error(f"Failed to list tools: HTTP {response.status_code}")
                return []
            
            response_data = response.json()
            
            if "error" in response_data:
                logging.error(f"Failed to list tools: {response_data['error']}")
                return []
            
            tools = response_data.get("result", {}).get("tools", [])
            tool_names = [tool.get("name", "") for tool in tools]
            
            logging.info(f"📋 Available HTTP MCP tools: {tool_names}")
            return tool_names
            
        except Exception as e:
            logging.error(f"Error listing HTTP MCP tools: {e}")
            return []
    
    async def test_connection(self) -> bool:
        """Test connection to the HTTP MCP server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            tools = await self.list_tools()
            return len(tools) > 0
        except Exception as e:
            logging.error(f"HTTP MCP connection test failed: {e}")
            return False
    
    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.aclose()
            self._session = None