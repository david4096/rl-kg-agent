#!/usr/bin/env python3
"""Training example for RL-KG-Agent."""

import json
from pathlib import Path

# Add src to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_kg_agent.knowledge.kg_loader import KnowledgeGraphLoader, SPARQLQueryGenerator
from rl_kg_agent.knowledge.internal_kg import InternalKnowledgeGraph
from rl_kg_agent.actions.action_manager import ActionManager
from rl_kg_agent.agents.ppo_agent import PPOKGAgent
from rl_kg_agent.utils.reward_calculator import RewardCalculator
from rl_kg_agent.utils.llm_client import LLMClient
from rl_kg_agent.data.dataset_loader import QADatasetLoader


def main():
    """Training example."""
    print("🎯 RL-KG-Agent Training Example")

    # Set up paths
    examples_dir = Path(__file__).parent
    ttl_file = examples_dir / "sample_knowledge_graph.ttl"
    config_file = examples_dir.parent / "configs" / "training_config.json"

    if not ttl_file.exists():
        print(f"❌ TTL file not found: {ttl_file}")
        return

    try:
        # 1. Load Configuration
        print(f"⚙️  Loading configuration from {config_file}")
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            print("⚠️  Config file not found, using defaults")
            config = {}

        # 2. Initialize Components
        print("📚 Initializing components...")

        # Knowledge Graph
        kg_loader = KnowledgeGraphLoader(str(ttl_file))
        sparql_generator = SPARQLQueryGenerator()
        internal_kg = InternalKnowledgeGraph("training_internal_kg.pkl")
        llm_client = LLMClient()

        # Action Manager
        action_manager = ActionManager(kg_loader, sparql_generator, internal_kg, llm_client)

        # Reward Calculator
        reward_calculator = RewardCalculator()

        # Configure reward weights if specified
        if "reward_weights" in config:
            reward_calculator.update_weights(**config["reward_weights"])

        # 3. Load Training Dataset
        print("📊 Loading training dataset...")
        dataset_loader = QADatasetLoader()

        # Try to load SQuAD dataset (fallback to sample data if not available)
        try:
            train_dataset = dataset_loader.load_squad_dataset("2.0", "train")
            print(f"   Loaded SQuAD dataset with {len(train_dataset)} examples")
        except Exception as e:
            print(f"   ⚠️  Failed to load SQuAD: {e}")
            print("   Using fallback dataset")
            train_dataset = dataset_loader._create_fallback_dataset()

        # Limit dataset size for demo
        max_examples = config.get("dataset_config", {}).get("max_examples", 100)
        if len(train_dataset) > max_examples:
            train_dataset = train_dataset.select(range(max_examples))
            print(f"   Limited to {len(train_dataset)} examples for demo")

        # 4. Initialize PPO Agent
        print("🤖 Initializing PPO agent...")
        agent = PPOKGAgent(action_manager, reward_calculator, internal_kg)

        # Configure PPO parameters
        if "ppo_config" in config:
            agent.update_training_config(**config["ppo_config"])

        # 5. Training Setup
        total_timesteps = config.get("training_config", {}).get("total_timesteps", 1000)
        print(f"🏋️  Starting training for {total_timesteps} timesteps...")

        # 6. Simple Training Loop Demo
        print("📈 Running training demo...")

        # Get a few examples for testing
        examples = list(train_dataset.select(range(min(5, len(train_dataset)))))

        for i, example in enumerate(examples):
            print(f"\n🔄 Example {i+1}/{len(examples)}: {example['question']}")

            # Set up environment with this example
            agent.env.set_query(
                example['question'],
                example.get('entities', []),
                example.get('answer', '')
            )

            # Execute one episode
            try:
                result = agent.execute_query(
                    example['question'],
                    example.get('entities', []),
                    example.get('answer', ''),
                    max_steps=3
                )

                print(f"   Total Reward: {result['total_reward']:.3f}")
                print(f"   Steps Taken: {result['final_step']}")

                # Show action sequence
                for step in result['steps_taken']:
                    status = "✅" if step['success'] else "❌"
                    print(f"   {status} {step['action']} (reward: {step['reward']:.3f})")

            except Exception as e:
                print(f"   ❌ Episode failed: {e}")

        # 7. Show Statistics
        print(f"\n📊 Training Statistics:")
        internal_stats = internal_kg.get_stats()
        print(f"   Internal KG Nodes: {internal_stats['total_nodes']}")
        print(f"   Internal KG Edges: {internal_stats['total_edges']}")
        print(f"   Stored Contexts: {internal_stats['total_contexts']}")

        dataset_stats = dataset_loader.get_dataset_stats()
        if dataset_stats:
            for name, stats in dataset_stats.items():
                print(f"   {name} Dataset: {stats['total_examples']} examples")

        # 8. Save Model (demo)
        model_path = "demo_trained_model"
        print(f"💾 Saving model to {model_path}")
        try:
            agent.save_model(model_path)
            print("   ✅ Model saved successfully")
        except Exception as e:
            print(f"   ⚠️  Model save failed: {e}")

        print("\n✅ Training example completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()