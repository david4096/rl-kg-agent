#!/usr/bin/env python3
"""Example usage of RL-KG-Agent with MCP integration for biomedical queries."""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_kg_agent.knowledge.kg_loader import KnowledgeGraphLoader, SPARQLQueryGenerator
from rl_kg_agent.knowledge.internal_kg import InternalKnowledgeGraph
from rl_kg_agent.utils.llm_client import LLMClient
from rl_kg_agent.utils.reward_calculator import RewardCalculator
from rl_kg_agent.agents.ppo_agent import PPOKGAgent
from rl_kg_agent.config import get_config, is_mcp_enabled, get_mcp_manager
from rl_kg_agent.actions.action_manager import ActionManager


def create_agent_with_mcp():
    """Create RL-KG-Agent with MCP support enabled."""
    print("🚀 Initializing RL-KG-Agent with MCP Support")
    print("=" * 50)
    
    # Load configuration
    config = get_config()
    print(f"📋 MCP enabled: {is_mcp_enabled(config)}")
    
    # Initialize standard components
    print("📚 Loading knowledge graph...")
    kg_loader = KnowledgeGraphLoader("examples/simple_knowledge_graph.ttl")
    sparql_generator = SPARQLQueryGenerator()
    internal_kg = InternalKnowledgeGraph("mcp_demo_internal_kg.pkl")
    llm_client = LLMClient()
    reward_calculator = RewardCalculator()
    
    # Initialize MCP manager if enabled
    mcp_manager = None
    if is_mcp_enabled(config):
        print("🔌 Initializing MCP manager...")
        mcp_manager = get_mcp_manager(config)
        if mcp_manager:
            print("✅ MCP manager ready for biomedical queries")
        else:
            print("❌ Failed to create MCP manager")
    else:
        print("⏭️  MCP disabled in configuration")
    
    # Create action manager with MCP support
    print("⚙️  Creating action manager...")
    action_manager = ActionManager(
        kg_loader, sparql_generator, internal_kg, llm_client, mcp_manager
    )
    
    # Display available actions
    actions = action_manager.get_action_descriptions()
    print(f"\n🎯 Available Actions ({len(actions)}):")
    for action_type, description in actions.items():
        print(f"   {action_type.value}. {action_type.name}: {description}")
    
    # Create agent
    print("\n🤖 Creating PPO agent...")
    agent = PPOKGAgent(action_manager, reward_calculator, internal_kg)
    
    return agent, action_manager, mcp_manager


def demo_biomedical_queries(agent, action_manager):
    """Demonstrate biomedical queries with MCP integration."""
    print("\n🧬 Biomedical Query Demo")
    print("=" * 30)
    
    # Sample biomedical queries that should trigger MCP action
    biomedical_queries = [
        "What are the molecular mechanisms of diabetes?",
        "Find research papers about COVID-19 vaccines",
        "How does insulin regulate glucose metabolism?",
        "What are the side effects of metformin?",
        "Search for studies on Alzheimer's disease treatment"
    ]
    
    print("Testing biomedical queries that should trigger MCP action...\n")
    
    for i, query in enumerate(biomedical_queries, 1):
        print(f"🔬 Query {i}: {query}")
        
        # Get action recommendations
        context = {
            "query": query,
            "entities": [],  # Would normally be extracted by NER
            "internal_knowledge": "",
            "failed_actions": 0
        }
        
        # Update context with internal knowledge
        enhanced_context = action_manager.update_context_with_internal_knowledge(context)
        
        # Get recommended actions
        recommendations = action_manager.get_action_recommendations(enhanced_context)
        
        print("   📊 Action Recommendations:")
        for action_type, confidence in recommendations[:3]:  # Show top 3
            print(f"      {action_type.name}: {confidence:.3f}")
        
        # Predict action using agent
        try:
            action_idx, prediction_info = agent.predict_action(query)
            predicted_action = list(action_manager.actions.keys())[action_idx]
            print(f"   🎯 Agent prediction: {predicted_action.name}")
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
        
        print()


def demo_mixed_queries(agent, action_manager):
    """Demonstrate mixed queries to show action selection."""
    print("\n🔄 Mixed Query Demo")
    print("=" * 25)
    
    mixed_queries = [
        ("What is the capital of France?", "General knowledge"),
        ("Explain the pathophysiology of hypertension", "Biomedical"),
        ("How does machine learning work?", "Technical"),
        ("What are the latest treatments for cancer?", "Biomedical research"),
        ("Tell me about quantum computing", "General technical")
    ]
    
    print("Testing diverse queries to see action selection patterns...\n")
    
    for i, (query, category) in enumerate(mixed_queries, 1):
        print(f"❓ Query {i} ({category}): {query}")
        
        context = {
            "query": query,
            "entities": [],
            "internal_knowledge": "",
            "failed_actions": 0
        }
        
        # Get top recommendation
        recommendations = action_manager.get_action_recommendations(context)
        if recommendations:
            top_action, confidence = recommendations[0]
            print(f"   🏆 Top recommendation: {top_action.name} ({confidence:.3f})")
        
        print()


def interactive_demo():
    """Interactive demo for testing MCP integration."""
    print("\n💬 Interactive Demo")
    print("=" * 20)
    print("Enter biomedical queries to test MCP integration.")
    print("Type 'quit' to exit.\n")
    
    agent, action_manager, mcp_manager = create_agent_with_mcp()
    
    while True:
        try:
            query = input("🧬 Enter biomedical query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\n🔍 Processing: {query}")
            
            # Get action recommendations
            context = {
                "query": query,
                "entities": [],
                "internal_knowledge": "",
                "failed_actions": 0
            }
            
            enhanced_context = action_manager.update_context_with_internal_knowledge(context)
            recommendations = action_manager.get_action_recommendations(enhanced_context)
            
            print("📊 Action Recommendations:")
            for action_type, confidence in recommendations:
                print(f"   {action_type.name}: {confidence:.3f}")
            
            # Execute top recommendation (if available)
            if recommendations and mcp_manager:
                top_action, _ = recommendations[0]
                
                if top_action.name == "QUERY_MCP_THEN_RESPOND":
                    print(f"\n🚀 Executing {top_action.name}...")
                    try:
                        result = action_manager.execute_action(top_action, enhanced_context)
                        if result.success:
                            print("✅ Action successful!")
                            print(f"📝 Response: {result.response[:200]}...")
                        else:
                            print(f"❌ Action failed: {result.metadata.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"❌ Execution error: {e}")
                else:
                    print(f"ℹ️  Top action is {top_action.name}, not MCP action")
            
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Demo completed!")


def main():
    """Main demo function."""
    print("🧬 RL-KG-Agent MCP Integration Demo")
    print("=" * 40)
    
    try:
        # Create agent with MCP support
        agent, action_manager, mcp_manager = create_agent_with_mcp()
        
        # Run demos
        demo_biomedical_queries(agent, action_manager)
        demo_mixed_queries(agent, action_manager)
        
        # Ask if user wants interactive demo
        if mcp_manager:
            run_interactive = input("🤔 Run interactive demo? (y/N): ").lower().strip() == 'y'
            if run_interactive:
                interactive_demo()
        else:
            print("\n⚠️  Interactive demo requires MCP server to be running")
            print("💡 Start with: python /Users/markstreer/mac-mini-sb/dbclshackathon/unified_mcp_server.py")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()