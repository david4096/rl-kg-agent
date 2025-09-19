#!/usr/bin/env python3
"""
Integration test for Azure OpenAI + MCP tools
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rl_kg_agent.utils.enhanced_llm_client import EnhancedLLMClient
from rl_kg_agent.utils.http_mcp_manager import HTTPMCPManager

async def test_integration():
    """Test the full integration of Azure OpenAI and MCP tools"""
    
    print("🧪 Testing Azure OpenAI + MCP Integration...")
    
    # Test 1: Azure OpenAI Client
    print("\n1️⃣ Testing Azure OpenAI Client...")
    try:
        llm_client = EnhancedLLMClient()
        response = llm_client.generate_response(
            "What is the capital of France? Keep it brief."
        )
        print(f"✅ Azure OpenAI Response: {response}")
    except Exception as e:
        print(f"❌ Azure OpenAI Error: {e}")
        return False
    
    # Test 2: MCP Tools
    print("\n2️⃣ Testing MCP Tools...")
    try:
        mcp_manager = HTTPMCPManager()
        await mcp_manager.initialize()
        
        # Test biomedical search
        search_result = await mcp_manager.call_tool(
            "default",
            "medline_semantic_search",
            {"query": "diabetes treatment", "max_results": 3}
        )
        print(f"✅ MCP Search Result: {search_result.result[:200]}...")
        
        # Test RAG answer
        rag_result = await mcp_manager.call_tool(
            "default",
            "get_rag_answer", 
            {"question": "What is insulin?"}
        )
        print(f"✅ MCP RAG Answer: {rag_result.result[:200]}...")
        
        # Close connection if available
        if hasattr(mcp_manager, 'close'):
            await mcp_manager.close()
        
    except Exception as e:
        print(f"❌ MCP Tools Error: {e}")
        return False
    
    # Test 3: Combined Usage
    print("\n3️⃣ Testing Combined Usage...")
    try:
        llm_client = EnhancedLLMClient()
        response = llm_client.generate_response(
            "Tell me about machine learning in healthcare. Keep it concise."
        )
        print(f"✅ Combined Response: {response[:300]}...")
    except Exception as e:
        print(f"❌ Combined Usage Error: {e}")
        return False
    
    print("\n🎉 All integration tests passed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_integration())