"""Enhanced LLM Client with Azure OpenAI and MCP Integration"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from dotenv import load_dotenv

from .langchain_framework import LangChainAgent
from .mcp_client import MCPClient

# Load environment variables
load_dotenv()

# Ensure SSL certificates are properly set
ssl_cert_file = os.getenv('SSL_CERT_FILE', '/Users/markstreer/certs/combined-certs.pem')
if ssl_cert_file:
    os.environ['SSL_CERT_FILE'] = ssl_cert_file
    os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_file
    os.environ['CURL_CA_BUNDLE'] = ssl_cert_file
    logging.info(f"SSL certificates configured: {ssl_cert_file}")

class MCPTool:
    """Tool wrapper for MCP biomedical queries"""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
    
    def __call__(self, query: str) -> str:
        """Execute biomedical query through MCP"""
        try:
            # Use asyncio to run the async query_biomedical method
            result = asyncio.run(self.mcp_client.query_biomedical(query))
            if isinstance(result, dict) and 'text' in result:
                return result['text']
            elif isinstance(result, str):
                return result
            else:
                return str(result)
        except Exception as e:
            logging.error(f"MCP query error: {e}")
            return f"Error executing biomedical query: {str(e)}"


class EnhancedLLMClient:
    """Enhanced LLM Client using Azure OpenAI with LangChain and MCP integration"""
    
    def __init__(self, 
                 mcp_server_path: Optional[str] = None,
                 use_mcp: bool = True,
                 verbose: bool = True):
        """Initialize the enhanced LLM client
        
        Args:
            mcp_server_path: Path to MCP server (if not provided, uses default)
            use_mcp: Whether to enable MCP biomedical tools
            verbose: Whether to enable verbose logging
        """
        self.use_mcp = use_mcp
        self.verbose = verbose
        self.mcp_client = None
        self.langchain_agent = None
        self.azure_available = True
        
        # Load Azure OpenAI configuration from environment
        self.azure_config = {
            'azure_deployment_name': os.getenv('AZURE_DEPLOYMENT_NAME', 'gpt-4'),
            'api_key': os.getenv('API_KEY'),
            'azure_endpoint': os.getenv('AZURE_ENDPOINT'),
            'api_version': os.getenv('API_VERSION', '2025-01-01-preview'),
            'temperature': float(os.getenv('TEMPERATURE', '0.1'))
        }
        
        # Validate Azure configuration
        required_keys = ['api_key', 'azure_endpoint']
        missing_keys = [key for key in required_keys if not self.azure_config[key]]
        if missing_keys:
            logging.warning(f"Missing Azure OpenAI environment variables: {missing_keys}")
            self.azure_available = False
        else:
            logging.info(f"🔌 Azure OpenAI configured: {self.azure_config['azure_endpoint']}")
        
        # Initialize MCP if requested
        if self.use_mcp:
            try:
                # For now, let's skip MCP initialization in enhanced client
                # and rely on the existing MCP manager in the action manager
                logging.info("🔬 MCP integration will be handled by action manager")
                self.use_mcp = True
                self.mcp_client = None
            except Exception as e:
                logging.warning(f"Failed to initialize MCP client: {e}")
                self.use_mcp = False
                self.mcp_client = None
        
        # Initialize LangChain agent or fallback
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with tools"""
        tools = []
        
        # Add MCP biomedical tool if available
        if self.use_mcp and self.mcp_client:
            from langchain_community.tools import tool
            
            @tool
            def biomedical_query(query: str) -> str:
                """Query biomedical databases and knowledge sources for information about diseases, drugs, genes, and biological processes."""
                mcp_tool = MCPTool(self.mcp_client)
                return mcp_tool(query)
            
            tools.append(biomedical_query)
        
        # Define system prompt for biomedical agent
        system_prompt = """You are an intelligent biomedical research assistant with access to comprehensive biomedical databases and knowledge sources.

Your capabilities include:
- Searching biomedical literature and databases
- Providing information about diseases, drugs, genes, and biological processes
- Answering questions about medical terminology and concepts
- Helping with biomedical research queries

When responding to biomedical questions:
1. Use the biomedical_query tool to search for relevant information
2. Provide accurate, evidence-based responses
3. Cite relevant sources when possible
4. Be clear about any limitations or uncertainties

You should be helpful, accurate, and professional in all interactions."""
        
        # Initialize LangChain agent
        self.langchain_agent = LangChainAgent(
            tools=tools,
            prompt=system_prompt,
            verbose=self.verbose,
            **self.azure_config
        )
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[str] = None,
                         max_tokens: int = 500) -> str:
        """Generate a response using Azure OpenAI with optional MCP tools
        
        Args:
            prompt: The input prompt/question
            context: Optional context to include
            max_tokens: Maximum tokens for response (not directly used with LangChain but kept for compatibility)
            
        Returns:
            Generated response string
        """
        try:
            # Prepare the full input
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
            
            # 🤖 ENHANCED LOGGING: Show LLM interaction details
            print(f"\n{'='*80}")
            print(f"🤖 LLM INTERACTION - Internal Processing")
            print(f"{'='*80}")
            print(f"💭 Input Prompt: {full_prompt[:300]}{'...' if len(full_prompt) > 300 else ''}")
            print(f"📚 Context Provided: {'Yes' if context else 'No'}")
            print(f"🔧 Azure OpenAI Model: gpt-4")
            
            logging.info(f"🤖 LLM Request: {full_prompt[:200]}{'...' if len(full_prompt) > 200 else ''}")
            
            # Create and run the agent
            agent_executor = self.langchain_agent.make_langchain_agent()
            
            print(f"⚙️  Processing... (This is INTERNAL LLM reasoning)")
            print(f"{'='*80}")
            
            # Execute the agent
            result = agent_executor.invoke({"input": full_prompt})
            
            # Extract the output
            if isinstance(result, dict) and 'output' in result:
                response = result['output']
            else:
                response = str(result)
            
            # 🗣️ ENHANCED LOGGING: Show user-facing response
            print(f"\n{'='*80}")
            print(f"🗣️  USER-FACING RESPONSE")
            print(f"{'='*80}")
            print(f"💬 Final Answer: {response}")
            print(f"📏 Response Length: {len(response)} characters")
            print(f"{'='*80}\n")
            
            logging.info(f"🤖 LLM Response: {response[:200]}{'...' if len(response) > 200 else ''}")
            return response
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logging.error(f"🚨 LLM Error: {error_msg}")
            return error_msg
    
    async def agenerate_response(self, 
                                prompt: str, 
                                context: Optional[str] = None,
                                max_tokens: int = 500) -> str:
        """Async version of generate_response"""
        # For now, run synchronously - can be enhanced later for true async
        return self.generate_response(prompt, context, max_tokens)
    
    def query_biomedical(self, query: str) -> str:
        """Direct biomedical query using MCP tools"""
        if not self.use_mcp or not self.mcp_client:
            logging.warning("🔬 Biomedical query tools are not available")
            return "Biomedical query tools are not available"
        
        try:
            logging.info(f"🔬 MCP Biomedical Query: {query}")
            mcp_tool = MCPTool(self.mcp_client)
            result = mcp_tool(query)
            logging.info(f"🔬 MCP Response: {result[:200]}{'...' if len(result) > 200 else ''}")
            return result
        except Exception as e:
            error_msg = f"Error executing biomedical query: {str(e)}"
            logging.error(f"🚨 MCP Error: {error_msg}")
            return error_msg
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get information about client capabilities"""
        return {
            'model': self.azure_config['azure_deployment_name'],
            'provider': 'Azure OpenAI',
            'framework': 'LangChain',
            'mcp_enabled': self.use_mcp,
            'tools': ['biomedical_query'] if self.use_mcp else [],
            'azure_endpoint': self.azure_config['azure_endpoint']
        }
    
    def __del__(self):
        """Cleanup when client is destroyed"""
        if self.mcp_client:
            try:
                # Close MCP client if it has cleanup methods
                if hasattr(self.mcp_client, 'close'):
                    self.mcp_client.close()
            except Exception as e:
                logging.warning(f"Error closing MCP client: {e}")


# Create a default instance for backwards compatibility
def create_enhanced_llm_client(mcp_server_path: Optional[str] = None, 
                             use_mcp: bool = True,
                             verbose: bool = True) -> EnhancedLLMClient:
    """Factory function to create an enhanced LLM client"""
    return EnhancedLLMClient(
        mcp_server_path=mcp_server_path,
        use_mcp=use_mcp,
        verbose=verbose
    )