#!/usr/bin/env python3
"""Basic usage example for RL-KG-Agent."""

import json
from pathlib import Path

# Add src to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_kg_agent.knowledge.kg_loader import KnowledgeGraphLoader, SPARQLQueryGenerator
from rl_kg_agent.knowledge.internal_kg import InternalKnowledgeGraph
from rl_kg_agent.actions.action_manager import ActionManager
from rl_kg_agent.utils.reward_calculator import RewardCalculator
from rl_kg_agent.utils.llm_client import LLMClient
from rl_kg_agent.data.dataset_loader import QADatasetLoader


def main():
    """Basic usage example."""
    print("🚀 RL-KG-Agent Basic Usage Example")

    # Set up paths
    examples_dir = Path(__file__).parent
    ttl_file = examples_dir / "simple_knowledge_graph.ttl"

    if not ttl_file.exists():
        print(f"❌ TTL file not found: {ttl_file}")
        print("Please run this from the examples directory or ensure the TTL file exists.")
        return

    try:
        # 1. Load Knowledge Graph
        print(f"📚 Loading knowledge graph from {ttl_file}")
        kg_loader = KnowledgeGraphLoader(str(ttl_file))
        sparql_generator = SPARQLQueryGenerator()

        # Show basic stats
        schema_info = kg_loader.get_schema_info()
        print(f"   Loaded {schema_info['total_triples']} triples")
        print(f"   Found {len(schema_info['classes'])} classes")
        print(f"   Found {len(schema_info['predicates'])} predicates")

        # 2. Initialize Internal KG
        print("🧠 Initializing internal knowledge graph")
        internal_kg = InternalKnowledgeGraph("example_internal_kg.pkl")

        # 3. Initialize LLM Client
        print("🤖 Initializing LLM client")
        llm_client = LLMClient()

        # 4. Create Action Manager
        print("⚙️  Setting up action manager")
        action_manager = ActionManager(kg_loader, sparql_generator, internal_kg, llm_client)

        # 5. Initialize Reward Calculator
        print("🎯 Initializing reward calculator")
        reward_calculator = RewardCalculator()

        # 6. Example Queries
        print("\n💬 Testing example queries...")

        example_queries = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "Tell me about Paris"
        ]

        for query in example_queries:
            print(f"\n❓ Query: {query}")

            # Prepare context
            context = {
                "query": query,
                "entities": _extract_simple_entities(query),
                "history": []
            }

            # Update with internal knowledge
            context = action_manager.update_context_with_internal_knowledge(context)

            # Get action recommendations
            recommendations = action_manager.get_action_recommendations(context)

            if recommendations:
                best_action, confidence = recommendations[0]
                print(f"   🎯 Best action: {best_action.name} (confidence: {confidence:.3f})")

                # Execute action
                result = action_manager.execute_action(best_action, context)

                if result.success:
                    print(f"   ✅ Success: {result.response}")

                    # Calculate reward
                    reward = reward_calculator.calculate_reward(context, result)
                    print(f"   📊 Reward: {reward.total:.3f}")

                    # Store to internal KG
                    if result.entities_discovered or result.relations_discovered:
                        store_result = action_manager.execute_action(
                            action_manager.actions[action_manager.actions.__iter__().__next__()].action_type.STORE_TO_INTERNAL_KG,
                            context,
                            response=result.response,
                            previous_action=best_action.name,
                            action_success=result.success
                        )
                        if store_result.success:
                            print(f"   💾 Stored to internal KG")
                else:
                    print(f"   ❌ Failed: {result.response}")
            else:
                print("   ⚠️  No applicable actions found")

        # 7. Show Internal KG Stats
        print(f"\n📈 Internal KG Statistics:")
        stats = internal_kg.get_stats()
        print(f"   Nodes: {stats['total_nodes']}")
        print(f"   Edges: {stats['total_edges']}")
        print(f"   Contexts: {stats['total_contexts']}")

        # 8. Example SPARQL Query
        print(f"\n🔍 Example SPARQL Query:")
        sparql_query = """
        SELECT ?subject ?predicate ?object WHERE {
            ?subject ?predicate ?object .
        } LIMIT 5
        """

        results = kg_loader.execute_sparql(sparql_query)
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results[:3], 1):
            print(f"   {i}. {result}")

        print("\n✅ Basic usage example completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def _extract_simple_entities(text):
    """Simple entity extraction for demo."""
    # Very basic - just look for capitalized words
    words = text.split()
    entities = [word.rstrip('?.,!') for word in words if word[0].isupper() and len(word) > 2]
    return entities[:5]  # Limit to 5


if __name__ == "__main__":
    main()