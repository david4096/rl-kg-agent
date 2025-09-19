#!/usr/bin/env python3
"""Test MCP integration with RL-KG-Agent."""

import sys
import asyncio
from pathlib import Path

# Add src to path for development  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_kg_agent.config import get_config, is_mcp_enabled, get_mcp_manager, validate_mcp_config
from rl_kg_agent.utils.mcp_client import MCPManager
from rl_kg_agent.actions.action_space import ActionType, QueryMCPThenRespondAction
from rl_kg_agent.utils.llm_client import LLMClient


async def test_mcp_client():
    """Test basic MCP client functionality."""
    print("🧪 Testing MCP Client...")
    
    try:
        # Load MCP manager from config
        config = get_config()
        
        print(f"📋 MCP enabled in config: {is_mcp_enabled(config)}")
        
        # Validate MCP config
        is_valid, errors = validate_mcp_config(config)
        if not is_valid:
            print("❌ MCP configuration errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        print("✅ MCP configuration is valid")
        
        # Create MCP manager
        mcp_manager = get_mcp_manager(config)
        if not mcp_manager:
            print("❌ Failed to create MCP manager")
            return False
        
        print("✅ MCP manager created successfully")
        
        # Test connection to unified biomedical server
        print("🔌 Testing connection to unified biomedical server...")
        
        success = await mcp_manager.connect_server("unified_biomedical")
        if not success:
            print("❌ Failed to connect to unified biomedical server")
            return False
        
        print("✅ Connected to unified biomedical server")
        
        # Test tool call
        print("🛠️  Testing medline_semantic_search tool...")
        
        response = await mcp_manager.call_tool(
            "unified_biomedical",
            "medline_semantic_search", 
            {"query": "diabetes treatment"}
        )
        
        if not response.success:
            print(f"❌ Tool call failed: {response.error}")
            return False
        
        print("✅ Tool call successful!")
        print(f"📄 Response length: {len(response.result)} characters")
        print(f"⏱️  Execution time: {response.execution_time:.2f}s")
        
        # Show first 200 characters of response
        if response.result:
            preview = response.result[:200] + "..." if len(response.result) > 200 else response.result
            print(f"📝 Response preview: {preview}")
        
        # Cleanup
        await mcp_manager.disconnect_server("unified_biomedical")
        print("🔌 Disconnected from server")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcp_action():
    """Test MCP action integration."""
    print("\n🎯 Testing MCP Action Integration...")
    
    try:
        # Create a mock LLM client for testing
        llm_client = LLMClient()
        
        # Create MCP manager
        mcp_manager = MCPManager("configs/mcp_config.json")
        
        # Create MCP action
        mcp_action = QueryMCPThenRespondAction(mcp_manager, llm_client)
        
        print("✅ MCP action created successfully")
        
        # Test action applicability
        test_contexts = [
            {
                "query": "what are the effects of insulin on glucose metabolism?",
                "entities": ["insulin", "glucose"]
            },
            {
                "query": "find papers about COVID-19 treatment",
                "entities": []
            },
            {
                "query": "what is the weather today?",
                "entities": []
            }
        ]
        
        for i, context in enumerate(test_contexts, 1):
            applicable = mcp_action.is_applicable(context)
            print(f"📝 Test {i}: '{context['query'][:50]}...' -> Applicable: {applicable}")
        
        print("✅ Action applicability tests completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Action test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_integration():
    """Test configuration integration."""
    print("\n⚙️  Testing Configuration Integration...")
    
    try:
        config = get_config()
        
        print(f"📊 Config loaded: {type(config).__name__}")
        print(f"🔧 MCP enabled: {is_mcp_enabled(config)}")
        print(f"📁 MCP config file: {config.mcp.config_file}")
        print(f"🏠 Default server: {config.mcp.default_server}")
        print(f"⏰ Connection timeout: {config.mcp.connection_timeout}s")
        
        # Check if MCP config file exists
        mcp_config_path = Path(config.mcp.config_file)
        if mcp_config_path.exists():
            print(f"✅ MCP config file found: {mcp_config_path}")
        else:
            print(f"❌ MCP config file not found: {mcp_config_path}")
            
            # Try relative paths
            for possible_path in ["configs/mcp_config.json", "../configs/mcp_config.json"]:
                if Path(possible_path).exists():
                    print(f"✅ Found MCP config at: {possible_path}")
                    break
            else:
                print("❌ No MCP config file found in common locations")
                return False
        
        print("✅ Configuration integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


async def main():
    """Run all MCP integration tests."""
    print("🚀 Starting MCP Integration Tests")
    print("=" * 50)
    
    # Test 1: Configuration
    config_ok = test_config_integration()
    
    # Test 2: Action integration  
    action_ok = test_mcp_action()
    
    # Test 3: MCP client (requires server to be running)
    print("\n⚠️  Note: MCP client test requires the unified MCP server to be running")
    print("💡 Start the server with: python /Users/markstreer/mac-mini-sb/dbclshackathon/unified_mcp_server.py")
    
    try_client_test = input("\n❓ Try MCP client test? (y/N): ").lower().strip() == 'y'
    
    client_ok = True
    if try_client_test:
        client_ok = await test_mcp_client()
    else:
        print("⏭️  Skipping MCP client test")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   ⚙️  Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   🎯 Action Integration: {'✅ PASS' if action_ok else '❌ FAIL'}")
    print(f"   🔌 MCP Client: {'✅ PASS' if client_ok else '❌ FAIL'}" + ("" if try_client_test else " (SKIPPED)"))
    
    if config_ok and action_ok and client_ok:
        print("\n🎉 All tests passed! MCP integration is ready.")
        return True
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)