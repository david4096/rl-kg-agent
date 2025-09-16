#!/usr/bin/env python3
"""Example of training with live visualization dashboard."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rl_kg_agent.knowledge.kg_loader import KnowledgeGraphLoader, SPARQLQueryGenerator
from rl_kg_agent.knowledge.internal_kg import InternalKnowledgeGraph
from rl_kg_agent.utils.llm_client import LLMClient
from rl_kg_agent.utils.reward_calculator import RewardCalculator
from rl_kg_agent.actions.action_manager import ActionManager
from rl_kg_agent.agents.ppo_agent import PPOKGAgent
from rl_kg_agent.utils.training_visualizer import TrainingDashboard, TrainingMonitor
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training_with_visualization():
    """Run training with live visualization."""
    print("🚀 Starting RL-KG-Agent Training with Visualization")
    print("=" * 60)

    try:
        # Initialize components
        print("📚 Loading knowledge graph...")
        kg_loader = KnowledgeGraphLoader("examples/simple_knowledge_graph.ttl")
        sparql_generator = SPARQLQueryGenerator()
        internal_kg = InternalKnowledgeGraph("training_internal_kg.pkl")
        llm_client = LLMClient()
        reward_calculator = RewardCalculator()

        print("🔧 Setting up action manager...")
        action_manager = ActionManager(kg_loader, sparql_generator, internal_kg, llm_client)

        print("📊 Starting visualization dashboard...")
        dashboard = TrainingDashboard()
        monitor = TrainingMonitor(dashboard, internal_kg)
        dashboard.start_dashboard()

        print("🤖 Initializing PPO agent...")
        agent = PPOKGAgent(action_manager, reward_calculator, internal_kg)

        # Set up some sample queries for training
        sample_queries = [
            ("What is the capital of France?", ["France"]),
            ("Who wrote Romeo and Juliet?", ["Romeo", "Juliet"]),
            ("What cities are in France?", ["France"]),
            ("Who is William Shakespeare?", ["William", "Shakespeare"]),
            ("What works did Shakespeare write?", ["Shakespeare"])
        ]

        print("🎯 Starting training with visualization...")
        print("Check the GUI window for live updates!")
        print("Training will run for 5000 timesteps...")

        # Run training episodes manually to demonstrate visualization
        total_timesteps = 5000
        episodes_run = 0

        try:
            # Use the built-in PPO training with our monitor
            agent.train(total_timesteps=total_timesteps, training_monitor=monitor)

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            logger.error(f"Training error: {e}")

        print("\n🎉 Training completed!")
        print("📊 Dashboard will remain open - close the GUI window when done")

        # Keep the script running so dashboard stays open
        try:
            while dashboard.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down dashboard...")
            dashboard.stop_dashboard()

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        logger.error(f"Setup error: {e}")
        return 1

    return 0


def demo_manual_training_loop():
    """Demo manual training loop with custom episodes."""
    print("🔄 Running demo with manual training episodes...")

    # This would be useful for custom training scenarios
    # where you want more control over the episode structure
    pass


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*60)
    print("TRAINING VISUALIZATION DEMO")
    print("="*60)
    print("This demo will:")
    print("• Load the simple knowledge graph")
    print("• Start a live training visualization dashboard")
    print("• Train the PPO agent with real-time monitoring")
    print("• Show live charts of rewards, losses, and KG growth")
    print("• Display internal KG statistics and recent queries")
    print("="*60)

    input("Press Enter to start the demo...")
    exit_code = run_training_with_visualization()
    sys.exit(exit_code)