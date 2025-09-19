"""
MCP Client utilities for communicating with Model Context Protocol servers.
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class MCPToolInfo:
    """Information about an available MCP tool."""
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class MCPServerInfo:
    """Information about an MCP server configuration."""
    name: str
    description: str
    transport: Dict[str, Any]
    tools: List[MCPToolInfo]
    enabled: bool
    timeout: int
    retry_attempts: int


@dataclass
class MCPResponse:
    """Response from an MCP tool call."""
    success: bool
    result: Any = None
    error: str = None
    tool_name: str = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MCPClientError(Exception):
    """Exception raised by MCP client operations."""
    pass


class MCPClient:
    """Client for communicating with an MCP server via stdio transport."""
    
    def __init__(self, server_config: MCPServerInfo):
        """Initialize MCP client.
        
        Args:
            server_config: Configuration for the MCP server
        """
        self.server_config = server_config
        self.process = None
        self._connected = False
        self._request_id = 0
        self._shared_process = None  # For reusing existing server process
        
    @classmethod
    def get_shared_process(cls, server_config: MCPServerInfo):
        """Get or create a shared server process for stdio transport."""
        # Use a class variable to store shared processes by server name
        if not hasattr(cls, '_shared_processes'):
            cls._shared_processes = {}
        
        server_name = server_config.name
        if server_name not in cls._shared_processes or cls._shared_processes[server_name] is None:
            logger.info(f"Creating new shared MCP server process for {server_name}")
            cls._shared_processes[server_name] = None  # Will be set in connect()
        
        return cls._shared_processes.get(server_name)
    
    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self._connected:
            return
            
        transport = self.server_config.transport
        
        if transport["type"] != "stdio":
            raise MCPClientError(f"Unsupported transport type: {transport['type']}")
        
        # Try to reuse existing shared process first
        shared_process = self.get_shared_process(self.server_config)
        if shared_process and shared_process.returncode is None:
            logger.info(f"Reusing existing MCP server process for {self.server_config.name}")
            self.process = shared_process
            self._connected = True
            return
        
        # Start a new server process only if no shared process exists
        cmd = [transport["command"]] + transport.get("args", [])
        working_dir = transport.get("working_directory")
        
        logger.info(f"Starting new MCP server: {' '.join(cmd)}")
        
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir
        )
        
        # Store as shared process
        self.__class__._shared_processes[self.server_config.name] = self.process
        
        # Initialize the connection only for new processes
        await self._initialize_connection()
        
    async def _initialize_connection(self):
        """Initialize the MCP connection with handshake."""
        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "rl-kg-agent",
                    "version": "0.1.0"
                }
            }
        }
        
        await self._send_request(init_request)
        response = await self._receive_response()
        
        if response.get("error"):
            raise MCPClientError(f"Initialization failed: {response['error']}")
        
        # Send initialized notification
        initialized_notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        await self._send_notification(initialized_notif)
        
        self._connected = True
        logger.info(f"Successfully connected to MCP server: {self.server_config.name}")

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except Exception as e:
                logger.warning(f"Error terminating MCP server process: {e}")
            finally:
                self.process = None
                self._connected = False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            MCPResponse with the result
        """
        if not self._connected:
            return MCPResponse(
                success=False,
                error="Not connected to MCP server",
                tool_name=tool_name
            )
        
        start_time = time.time()
        
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            logger.debug(f"Calling MCP tool: {tool_name} with args: {arguments}")
            
            await self._send_request(request)
            response = await self._receive_response()
            
            execution_time = time.time() - start_time
            
            if response.get("error"):
                error_msg = str(response["error"])
                logging.error(f"🚨 MCP Tool Error: {tool_name} failed with: {error_msg}")
                return MCPResponse(
                    success=False,
                    error=error_msg,
                    tool_name=tool_name,
                    execution_time=execution_time
                )
            
            result = response.get("result", {})
            tool_result = result.get("content", [])
            
            # Extract text content from the result
            if isinstance(tool_result, list) and tool_result:
                content = tool_result[0].get("text", "")
            else:
                content = str(tool_result)
            
            logging.info(f"🔧 MCP Tool Success: {tool_name} returned {len(content)} chars in {execution_time:.2f}s")
            
            return MCPResponse(
                success=True,
                result=content,
                tool_name=tool_name,
                execution_time=execution_time,
                metadata={
                    "isError": result.get("isError", False),
                    "_meta": result.get("_meta", {})
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool call failed: {e}"
            logging.error(f"🚨 MCP Exception: {tool_name} - {error_msg}")
            return MCPResponse(
                success=False,
                error=str(e),
                tool_name=tool_name,
                execution_time=execution_time
            )
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server.
        
        Returns:
            List of tool definitions
        """
        if not self._connected:
            return []
        
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/list"
            }
            
            await self._send_request(request)
            response = await self._receive_response()
            
            if response.get("error"):
                logger.error(f"Failed to list tools: {response['error']}")
                return []
            
            return response.get("result", {}).get("tools", [])
            
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    def _get_next_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id
    
    async def _send_request(self, request: Dict[str, Any]):
        """Send a JSON-RPC request to the server."""
        if not self.process or not self.process.stdin:
            raise MCPClientError("Process not available for sending request")
        
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()
    
    async def _send_notification(self, notification: Dict[str, Any]):
        """Send a JSON-RPC notification to the server."""
        if not self.process or not self.process.stdin:
            raise MCPClientError("Process not available for sending notification")
        
        notification_str = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_str.encode())
        await self.process.stdin.drain()
    
    async def _receive_response(self) -> Dict[str, Any]:
        """Receive a JSON-RPC response from the server."""
        if not self.process or not self.process.stdout:
            raise MCPClientError("Process not available for receiving response")
        
        try:
            # Read line with timeout
            line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=self.server_config.timeout
            )
            
            if not line:
                raise MCPClientError("No response received from server")
            
            response = json.loads(line.decode().strip())
            return response
            
        except asyncio.TimeoutError:
            raise MCPClientError(f"Timeout waiting for response after {self.server_config.timeout}s")
        except json.JSONDecodeError as e:
            raise MCPClientError(f"Invalid JSON response: {e}")


class MCPManager:
    """Manager for multiple MCP clients and configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP manager with configuration.
        
        Args:
            config_path: Path to MCP configuration file
        """
        self.config_path = config_path
        self.servers: Dict[str, MCPServerInfo] = {}
        self.clients: Dict[str, MCPClient] = {}
        self.load_config()
    
    def load_config(self):
        """Load MCP server configurations from file."""
        if not self.config_path:
            # Look for default config file
            for possible_path in [
                "configs/mcp_config.json",
                "mcp_config.json",
                "../configs/mcp_config.json"
            ]:
                if Path(possible_path).exists():
                    self.config_path = possible_path
                    break
        
        if not self.config_path or not Path(self.config_path).exists():
            logger.warning("No MCP configuration file found")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.servers.clear()
            
            for server_id, server_config in config.get("mcp_servers", {}).items():
                if not server_config.get("enabled", True):
                    continue
                
                # Convert tool list to MCPToolInfo objects
                tools = []
                for tool_config in server_config.get("tools", []):
                    tool = MCPToolInfo(
                        name=tool_config["name"],
                        description=tool_config["description"],
                        parameters=tool_config.get("parameters", {})
                    )
                    tools.append(tool)
                
                server_info = MCPServerInfo(
                    name=server_config["name"],
                    description=server_config["description"],
                    transport=server_config["transport"],
                    tools=tools,
                    enabled=server_config.get("enabled", True),
                    timeout=server_config.get("timeout", 30),
                    retry_attempts=server_config.get("retry_attempts", 3)
                )
                
                self.servers[server_id] = server_info
            
            logger.info(f"Loaded {len(self.servers)} MCP server configurations")
            
        except Exception as e:
            logger.error(f"Failed to load MCP configuration: {e}")
    
    async def connect_server(self, server_id: str) -> bool:
        """Connect to a specific MCP server.
        
        Args:
            server_id: ID of the server to connect to
            
        Returns:
            True if connection successful
        """
        if server_id not in self.servers:
            logger.error(f"Server {server_id} not found in configuration")
            return False
        
        if server_id in self.clients:
            # Already connected
            return True
        
        server_config = self.servers[server_id]
        client = MCPClient(server_config)
        
        if await client.connect():
            self.clients[server_id] = client
            return True
        else:
            return False
    
    async def disconnect_server(self, server_id: str):
        """Disconnect from a specific MCP server."""
        if server_id in self.clients:
            await self.clients[server_id].disconnect()
            del self.clients[server_id]
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for server_id in list(self.clients.keys()):
            await self.disconnect_server(server_id)
    
    async def call_tool(self, server_id: str, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """Call a tool on a specific MCP server.
        
        Args:
            server_id: ID of the server to call
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            MCPResponse with the result
        """
        # Ensure we're connected to the server
        if not await self.connect_server(server_id):
            return MCPResponse(
                success=False,
                error=f"Failed to connect to server {server_id}",
                tool_name=tool_name
            )
        
        client = self.clients[server_id]
        return await client.call_tool(tool_name, arguments)
    
    def get_available_tools(self, server_id: Optional[str] = None) -> Dict[str, List[MCPToolInfo]]:
        """Get available tools from MCP servers.
        
        Args:
            server_id: Specific server ID, or None for all servers
            
        Returns:
            Dictionary mapping server IDs to their available tools
        """
        result = {}
        
        if server_id:
            if server_id in self.servers:
                result[server_id] = self.servers[server_id].tools
        else:
            for sid, server in self.servers.items():
                result[sid] = server.tools
        
        return result
    
    def get_tool_by_name(self, tool_name: str) -> Optional[tuple[str, MCPToolInfo]]:
        """Find a tool by name across all servers.
        
        Args:
            tool_name: Name of the tool to find
            
        Returns:
            Tuple of (server_id, tool_info) if found, None otherwise
        """
        for server_id, server in self.servers.items():
            for tool in server.tools:
                if tool.name == tool_name:
                    return server_id, tool
        
        return None