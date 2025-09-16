"""Command line interface for the RL-KG-Agent."""

import click
import logging
import sys
from pathlib import Path
from typing import Optional
import json

from .knowledge.kg_loader import KnowledgeGraphLoader, SPARQLQueryGenerator
from .knowledge.internal_kg import InternalKnowledgeGraph
from .actions.action_manager import ActionManager
from .agents.ppo_agent import PPOKGAgent
from .utils.reward_calculator import RewardCalculator
from .data.dataset_loader import QADatasetLoader
from .utils.llm_client import LLMClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=str, help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """RL-KG-Agent: Reinforcement Learning Agent for Knowledge Graph Reasoning."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure context object exists
    ctx.ensure_object(dict)

    # Load configuration if provided
    if config and Path(config).exists():
        with open(config, 'r') as f:
            ctx.obj['config'] = json.load(f)
    else:
        ctx.obj['config'] = {}


@cli.command()
@click.option('--ttl-file', '-t', type=str, required=True, help='Path to TTL knowledge graph file')
@click.option('--model-path', '-m', type=str, help='Path to pre-trained model (optional)')
@click.option('--max-steps', default=5, help='Maximum steps per query')
@click.option('--save-interactions', is_flag=True, help='Save interactions to internal KG')
@click.pass_context
def interactive(ctx, ttl_file, model_path, max_steps, save_interactions):
    """Start interactive mode for querying the agent."""
    try:
        # Initialize components
        click.echo("🚀 Initializing RL-KG-Agent...")

        # Load knowledge graph
        click.echo(f"📚 Loading knowledge graph from {ttl_file}")
        kg_loader = KnowledgeGraphLoader(ttl_file)
        sparql_generator = SPARQLQueryGenerator()

        # Initialize internal KG
        internal_kg = InternalKnowledgeGraph("internal_kg.pkl")

        # Initialize LLM client (placeholder)
        llm_client = LLMClient()

        # Initialize action manager
        action_manager = ActionManager(kg_loader, sparql_generator, internal_kg, llm_client)

        # Initialize reward calculator
        reward_calculator = RewardCalculator()

        # Initialize PPO agent
        agent = PPOKGAgent(action_manager, reward_calculator, internal_kg)

        # Load pre-trained model if provided
        if model_path and Path(model_path).exists():
            click.echo(f"🧠 Loading pre-trained model from {model_path}")
            agent.load_model(model_path)
        else:
            click.echo("⚠️  No pre-trained model provided, using default policy")

        click.echo("✅ Initialization complete!")
        click.echo("💬 Enter your questions (type 'quit' to exit, 'help' for commands)")

        # Interactive loop
        while True:
            try:
                # Get user input
                query = click.prompt("❓ Question", type=str)

                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'help':
                    _show_help()
                    continue
                elif query.lower() == 'stats':
                    _show_stats(internal_kg, kg_loader)
                    continue
                elif query.lower().startswith('config'):
                    _handle_config_command(query, agent)
                    continue

                # Process query
                click.echo("🤖 Processing your question...")

                # Extract entities (simple approach)
                entities = _extract_entities_simple(query)

                if model_path:
                    # Use trained agent
                    result = agent.execute_query(query, entities, max_steps=max_steps)
                    _display_agent_result(result)
                else:
                    # Use action manager directly (fallback mode)
                    result = _execute_with_action_manager(
                        action_manager, query, entities, max_steps, save_interactions
                    )
                    _display_simple_result(result)

            except KeyboardInterrupt:
                break
            except EOFError:
                # Handle EOF (Ctrl+D or no input available)
                click.echo("\n👋 Goodbye!")
                break
            except Exception as e:
                if str(e):  # Only display error if it's not empty
                    click.echo(f"❌ Error: {e}", err=True)
                    logger.error(f"Interactive mode error: {e}")
                else:
                    # If empty error, likely a prompt issue, exit gracefully
                    click.echo("\n⚠️ Input error occurred, exiting...")
                    logger.error("Empty error in interactive mode, likely prompt issue")
                    break

        click.echo("👋 Goodbye!")

    except Exception as e:
        click.echo(f"❌ Failed to initialize: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--ttl-file', '-t', type=str, required=True, help='Path to TTL knowledge graph file')
@click.option('--dataset', '-d', type=str, default="squad", help='Dataset to use (squad, natural_questions, ms_marco)')
@click.option('--episodes', '-e', type=int, default=1000, help='Number of training episodes')
@click.option('--output-model', '-o', type=str, default="trained_model", help='Output path for trained model')
@click.option('--sample-size', type=int, help='Limit dataset size for faster training')
@click.option('--visualize', is_flag=True, help='Show live training visualization dashboard')
@click.pass_context
def train(ctx, ttl_file, dataset, episodes, output_model, sample_size, visualize):
    """Train the RL agent using QA datasets."""
    try:
        click.echo("🎯 Starting training...")

        # Initialize components
        click.echo("📚 Loading knowledge graph...")
        kg_loader = KnowledgeGraphLoader(ttl_file)
        sparql_generator = SPARQLQueryGenerator()
        internal_kg = InternalKnowledgeGraph("training_internal_kg.pkl")
        llm_client = LLMClient()

        action_manager = ActionManager(kg_loader, sparql_generator, internal_kg, llm_client)
        reward_calculator = RewardCalculator()

        # Load training dataset
        click.echo(f"📊 Loading {dataset} dataset...")
        dataset_loader = QADatasetLoader()

        if dataset == "squad":
            train_dataset = dataset_loader.load_squad_dataset("2.0", "train")
        elif dataset == "natural_questions":
            train_dataset = dataset_loader.load_natural_questions("train", sample_size)
        elif dataset == "ms_marco":
            train_dataset = dataset_loader.load_ms_marco("train", sample_size)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        click.echo(f"📈 Dataset loaded: {len(train_dataset)} examples")

        # Initialize training visualization if requested
        dashboard = None
        monitor = None
        if visualize:
            click.echo("📊 Starting training visualization dashboard...")
            try:
                from .utils.training_visualizer import TrainingDashboard, TrainingMonitor
                dashboard = TrainingDashboard()
                monitor = TrainingMonitor(dashboard, internal_kg)
                dashboard.start_dashboard()
                click.echo("✅ Dashboard started - check the GUI window")
            except ImportError:
                click.echo("⚠️  Visualization dependencies not available (matplotlib, tkinter)")
                visualize = False
            except Exception as e:
                click.echo(f"⚠️  Failed to start visualization: {e}")
                visualize = False

        # Initialize and train agent
        agent = PPOKGAgent(action_manager, reward_calculator, internal_kg)

        # Convert episodes to timesteps (rough approximation)
        total_timesteps = episodes * 5  # Assuming ~5 steps per episode

        click.echo(f"🏋️  Training for {total_timesteps} timesteps...")

        # Train with visualization support
        if monitor:
            agent.train(total_timesteps, output_model, monitor)
        else:
            # Custom training loop with dataset examples
            _train_with_dataset(agent, train_dataset, total_timesteps)

        # Save trained model
        click.echo(f"💾 Saving model to {output_model}")
        agent.save_model(output_model)

        # Stop dashboard if it was started
        if dashboard:
            try:
                dashboard.stop_dashboard()
            except Exception as e:
                logger.warning(f"Failed to stop dashboard cleanly: {e}")

        click.echo("✅ Training complete!")

    except Exception as e:
        click.echo(f"❌ Training failed: {e}", err=True)
        logger.error(f"Training error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--ttl-file', '-t', type=str, required=True, help='Path to TTL knowledge graph file')
@click.option('--test-file', type=str, help='JSON file with test questions')
@click.option('--model-path', '-m', type=str, required=True, help='Path to trained model')
@click.option('--output-file', '-o', type=str, help='Output file for results')
@click.pass_context
def evaluate(ctx, ttl_file, test_file, model_path, output_file):
    """Evaluate the trained agent on test questions."""
    try:
        click.echo("📊 Starting evaluation...")

        # Initialize components
        kg_loader = KnowledgeGraphLoader(ttl_file)
        sparql_generator = SPARQLQueryGenerator()
        internal_kg = InternalKnowledgeGraph("eval_internal_kg.pkl")
        llm_client = LLMClient()

        action_manager = ActionManager(kg_loader, sparql_generator, internal_kg, llm_client)
        reward_calculator = RewardCalculator()

        # Load trained agent
        agent = PPOKGAgent(action_manager, reward_calculator, internal_kg)
        agent.load_model(model_path)

        # Load test questions
        if test_file and Path(test_file).exists():
            with open(test_file, 'r') as f:
                test_questions = json.load(f)
        else:
            # Create sample test questions
            test_questions = [
                {"question": "What is the capital of France?", "expected_answer": "Paris"},
                {"question": "Who wrote Romeo and Juliet?", "expected_answer": "William Shakespeare"},
                {"question": "What is photosynthesis?", "expected_answer": "Process of converting light energy to chemical energy"}
            ]

        click.echo(f"🧪 Evaluating on {len(test_questions)} questions...")

        results = []
        total_reward = 0

        for i, test_case in enumerate(test_questions):
            question = test_case["question"]
            expected = test_case.get("expected_answer", "")

            click.echo(f"  [{i+1}/{len(test_questions)}] {question}")

            # Extract entities
            entities = _extract_entities_simple(question)

            # Execute query
            result = agent.execute_query(question, entities, expected)
            total_reward += result["total_reward"]

            results.append({
                "question": question,
                "expected_answer": expected,
                "agent_result": result,
                "reward": result["total_reward"]
            })

        # Calculate metrics
        avg_reward = total_reward / len(test_questions)
        success_rate = sum(1 for r in results if r["reward"] > 0.5) / len(results)

        click.echo(f"📈 Evaluation Results:")
        click.echo(f"   Average Reward: {avg_reward:.3f}")
        click.echo(f"   Success Rate: {success_rate:.1%}")

        # Save detailed results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    "summary": {
                        "avg_reward": avg_reward,
                        "success_rate": success_rate,
                        "total_questions": len(test_questions)
                    },
                    "results": results
                }, f, indent=2)
            click.echo(f"💾 Detailed results saved to {output_file}")

    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}", err=True)
        logger.error(f"Evaluation error: {e}")
        sys.exit(1)


def _show_help():
    """Show help for interactive commands."""
    help_text = """
Interactive Commands:
  help     - Show this help message
  stats    - Show knowledge graph statistics
  config   - Show/modify agent configuration
  quit/q   - Exit the program
    """
    click.echo(help_text)


def _show_stats(internal_kg, kg_loader):
    """Show knowledge graph statistics."""
    click.echo("📊 Knowledge Graph Statistics:")

    # Static KG stats
    schema_info = kg_loader.get_schema_info()
    click.echo(f"  Static KG: {schema_info['total_triples']} triples")
    click.echo(f"  Classes: {len(schema_info['classes'])}")
    click.echo(f"  Predicates: {len(schema_info['predicates'])}")

    # Internal KG stats
    internal_stats = internal_kg.get_stats()
    click.echo(f"  Internal KG: {internal_stats['total_nodes']} nodes, {internal_stats['total_edges']} edges")
    click.echo(f"  Contexts stored: {internal_stats['total_contexts']}")


def _handle_config_command(command, agent):
    """Handle configuration commands."""
    if command == "config":
        click.echo("Agent Configuration:")
        click.echo(f"  Training steps: {agent.training_steps}")
        click.echo(f"  Episodes completed: {agent.episodes_completed}")
    # Could add more config options here


def _extract_entities_simple(text):
    """Simple entity extraction (placeholder)."""
    # This is a very basic implementation
    words = text.split()
    entities = [word for word in words if word[0].isupper() and len(word) > 2]
    return entities[:5]


def _execute_with_action_manager(action_manager, query, entities, max_steps, save_interactions):
    """Execute query using action manager directly (fallback mode)."""
    context = {
        "query": query,
        "entities": entities,
        "history": []
    }

    # Update context with internal knowledge
    context = action_manager.update_context_with_internal_knowledge(context)

    # Get action recommendations
    recommendations = action_manager.get_action_recommendations(context)

    if not recommendations:
        return {"response": "No applicable actions found for this query.", "success": False}

    # Execute top recommended action
    best_action, confidence = recommendations[0]
    result = action_manager.execute_action(best_action, context)

    # Storage is now handled within actions themselves
    # No need for separate storage step

    return {
        "response": result.response,
        "success": result.success,
        "action_taken": best_action.name,
        "confidence": confidence
    }


def _display_agent_result(result):
    """Display result from trained agent."""
    click.echo(f"🎯 Total Reward: {result['total_reward']:.3f}")
    click.echo(f"🔢 Steps Taken: {result['final_step']}")

    click.echo("📋 Action Sequence:")
    for step in result['steps_taken']:
        status = "✅" if step['success'] else "❌"
        click.echo(f"  {status} Step {step['step'] + 1}: {step['action']} (reward: {step['reward']:.3f})")

    # Show final response if available
    if result['steps_taken']:
        last_step = result['steps_taken'][-1]
        if 'info' in last_step and 'response' in last_step['info']:
            click.echo(f"💬 Response: {last_step['info']['response']}")


def _display_simple_result(result):
    """Display result from simple action manager execution."""
    status = "✅" if result['success'] else "❌"
    click.echo(f"{status} Action: {result['action_taken']} (confidence: {result['confidence']:.3f})")
    click.echo(f"💬 Response: {result['response']}")


def _train_with_dataset(agent, dataset, total_timesteps):
    """Custom training loop using dataset examples."""
    # This is a simplified training approach
    # In practice, you'd want more sophisticated episode management

    examples = list(dataset)
    current_example_idx = 0

    # Set up first training example
    if examples:
        example = examples[current_example_idx]
        agent.env.set_query(
            example['question'],
            example.get('entities', []),
            example.get('answer', '')
        )

    # Train the agent
    def episode_callback():
        nonlocal current_example_idx
        # Move to next example for next episode
        current_example_idx = (current_example_idx + 1) % len(examples)
        if examples:
            example = examples[current_example_idx]
            agent.env.set_query(
                example['question'],
                example.get('entities', []),
                example.get('answer', '')
            )

    # Standard PPO training with periodic example updates
    agent.train(total_timesteps)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()