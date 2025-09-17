"""Command line interface for the RL-KG-Agent with TorchRL support."""

import click
import logging
import sys
from pathlib import Path
from typing import Optional
import json
import warnings

from .knowledge.kg_loader import KnowledgeGraphLoader, SPARQLQueryGenerator
from .knowledge.internal_kg import InternalKnowledgeGraph
from .actions.action_manager import ActionManager
from .agents.ppo_agent import PPOKGAgent, create_environment, create_action_executor
from .utils.reward_calculator import RewardCalculator
from .data.dataset_loader import QADatasetLoader
from .utils.llm_client import LLMClient
from .config import Config, ConfigManager, get_config, is_torchrl_enabled, validate_torchrl_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=str, help='Configuration file path')
@click.option('--use-torchrl', is_flag=True, help='Enable TorchRL features (requires torchrl[llm])')
@click.option('--torchrl-device', type=click.Choice(['auto', 'cpu', 'cuda']), default='auto', help='Device for TorchRL operations')
@click.pass_context
def cli(ctx, verbose, config, use_torchrl, torchrl_device):
    """RL-KG-Agent: Reinforcement Learning Agent for Knowledge Graph Reasoning with optional TorchRL support."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure context object exists
    ctx.ensure_object(dict)

    # Load configuration if provided
    if config and Path(config).exists():
        try:
            loaded_config = ConfigManager.load_config(config)
            ctx.obj['config'] = loaded_config
        except Exception as e:
            logger.error(f"Failed to load config from {config}: {e}")
            ctx.obj['config'] = Config()
    else:
        ctx.obj['config'] = get_config()

    # Override TorchRL settings from command line
    if use_torchrl:
        ctx.obj['config'].torchrl.enabled = True
        ctx.obj['config'].torchrl.device = torchrl_device
        
        # Validate TorchRL configuration
        is_valid, errors = validate_torchrl_config(ctx.obj['config'])
        if not is_valid:
            logger.warning("TorchRL configuration issues detected:")
            for error in errors:
                logger.warning(f"  - {error}")
            
            # Ask user if they want to continue
            if not click.confirm("Continue with potential issues?"):
                ctx.exit(1)
    
    # Store CLI flags
    ctx.obj['use_torchrl'] = use_torchrl
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option('--ttl-file', '-t', type=str, required=True, help='Path to TTL knowledge graph file')
@click.option('--model-path', '-m', type=str, help='Path to pre-trained model (optional)')
@click.option('--max-steps', default=5, help='Maximum steps per query')
@click.option('--save-interactions', is_flag=True, help='Save interactions to internal KG')
@click.option('--enable-tools', is_flag=True, help='Enable TorchRL tool enhancements for this session')
@click.pass_context
def interactive(ctx, ttl_file, model_path, max_steps, save_interactions, enable_tools):
    """Start interactive mode for querying the agent with optional TorchRL enhancements."""
    try:
        config = ctx.obj.get('config', Config())
        use_torchrl = ctx.obj.get('use_torchrl', False) or enable_tools
        
        # Show configuration status
        if use_torchrl:
            click.echo("🔧 TorchRL mode enabled - Enhanced tool capabilities active")
        else:
            click.echo("📊 Standard mode - Using stable PPO training")
        
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

        # Initialize action manager (potentially enhanced)
        action_manager = ActionManager(kg_loader, sparql_generator, internal_kg, llm_client)
        enhanced_action_manager = create_action_executor(
            action_manager=action_manager,
            kg_loader=kg_loader,
            internal_kg=internal_kg,
            llm_client=llm_client,
            config=config,
            use_torchrl=use_torchrl
        )

        # Initialize reward calculator
        reward_calculator = RewardCalculator()

        # Create environment (potentially TorchRL-enhanced)
        environment = create_environment(
            action_manager=enhanced_action_manager,
            reward_calculator=reward_calculator,
            internal_kg=internal_kg,
            kg_loader=kg_loader,
            config=config,
            use_torchrl=use_torchrl,
            max_steps=max_steps
        )

        # Initialize PPO agent with enhanced environment
        if use_torchrl:
            click.echo("✨ Using TorchRL-enhanced environment with tool capabilities")
        
        agent = PPOKGAgent(enhanced_action_manager, reward_calculator, internal_kg)

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
@click.option('--output', '-o', type=str, default='rl_kg_agent_config.json', help='Output configuration file path')
@click.option('--enable-torchrl', is_flag=True, help='Enable TorchRL features in generated config')
@click.option('--with-examples', is_flag=True, help='Include example settings and comments')
def init_config(output, enable_torchrl, with_examples):
    """Initialize a configuration file with default or example settings."""
    try:
        if with_examples:
            ConfigManager.create_example_config(output)
        else:
            config = Config()
            if enable_torchrl:
                config.torchrl.enabled = True
                config.torchrl.enable_tool_enhancement = True
                click.echo("✅ TorchRL features enabled in configuration")
            
            ConfigManager.save_config(config, output)
        
        click.echo(f"📝 Configuration file created: {output}")
        
        # Validate the created config
        config = ConfigManager.load_config(output)
        is_valid, errors = validate_torchrl_config(config)
        
        if not is_valid:
            click.echo("⚠️  Configuration validation warnings:")
            for error in errors:
                click.echo(f"   - {error}")
        else:
            click.echo("✅ Configuration is valid")
            
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        sys.exit(1)


@cli.command()
@click.option('--config-file', '-c', type=str, help='Configuration file to validate')
def validate_config(config_file):
    """Validate a configuration file."""
    try:
        if config_file:
            config = ConfigManager.load_config(config_file)
        else:
            config = get_config()
            config_file = "default configuration"
        
        click.echo(f"🔍 Validating {config_file}...")
        
        # General validation
        click.echo(f"✅ Basic configuration loaded successfully")
        
        # TorchRL validation
        is_valid, errors = validate_torchrl_config(config)
        
        if config.torchrl.enabled:
            if is_valid:
                click.echo("✅ TorchRL configuration is valid")
                
                # Check dependencies
                from .config import check_torchrl_dependencies
                if check_torchrl_dependencies():
                    click.echo("✅ TorchRL dependencies are available")
                else:
                    click.echo("⚠️  TorchRL dependencies not found. Install with: pip install 'torchrl[llm]'")
            else:
                click.echo("❌ TorchRL configuration has issues:")
                for error in errors:
                    click.echo(f"   - {error}")
        else:
            click.echo("ℹ️  TorchRL features are disabled")
        
        # Summary
        click.echo("\\n📊 Configuration Summary:")
        click.echo(f"   TorchRL enabled: {config.torchrl.enabled}")
        click.echo(f"   Tool enhancement: {config.torchrl.enable_tool_enhancement}")
        click.echo(f"   Device: {config.torchrl.device}")
        click.echo(f"   Max steps: {config.environment_config.get('max_steps', 10)}")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)


@cli.command()
def check_deps():
    """Check system dependencies for TorchRL features."""
    click.echo("🔍 Checking system dependencies...")
    
    # Check Python version
    import sys
    click.echo(f"✅ Python {sys.version}")
    
    # Check core dependencies
    deps_status = {}
    
    try:
        import torch
        deps_status['PyTorch'] = f"✅ {torch.__version__}"
    except ImportError:
        deps_status['PyTorch'] = "❌ Not installed"
    
    try:
        import torchrl
        deps_status['TorchRL'] = f"✅ {torchrl.__version__}"
    except ImportError:
        deps_status['TorchRL'] = "❌ Not installed"
    
    try:
        from tensordict import TensorDict
        deps_status['TensorDict'] = "✅ Available"
    except ImportError:
        deps_status['TensorDict'] = "❌ Not installed"
    
    try:
        from transformers import AutoTokenizer
        deps_status['Transformers'] = "✅ Available"
    except ImportError:
        deps_status['Transformers'] = "❌ Not installed"
    
    try:
        import playwright
        deps_status['Playwright'] = "✅ Available (for browser tools)"
    except ImportError:
        deps_status['Playwright'] = "⚠️  Not installed (optional for browser tools)"
    
    # Display results
    click.echo("\\n📦 Dependencies Status:")
    for dep, status in deps_status.items():
        click.echo(f"   {dep}: {status}")
    
    # Installation guidance
    click.echo("\\n📋 Installation Guide:")
    click.echo("   Core TorchRL features: pip install 'torchrl[llm]'")
    click.echo("   Browser tools (optional): pip install playwright && playwright install")
    
    # Overall status
    core_deps = ['PyTorch', 'TorchRL', 'TensorDict', 'Transformers']
    core_available = all('✅' in deps_status.get(dep, '') for dep in core_deps)
    
    if core_available:
        click.echo("\\n✅ All core dependencies available for TorchRL features!")
    else:
        click.echo("\\n⚠️  Some core dependencies missing. TorchRL features may not work.")


@cli.command()
@click.option('--ttl-file', '-t', type=str, required=True, help='Path to TTL knowledge graph file')
@click.option('--dataset', '-d', type=str, default="squad", help='Dataset to use (squad, natural_questions, ms_marco)')
@click.option('--episodes', '-e', type=int, default=1000, help='Number of training episodes')
@click.option('--output-model', '-o', type=str, default="trained_model", help='Output path for trained model')
@click.option('--sample-size', type=int, help='Limit dataset size for faster training')
@click.option('--visualize', is_flag=True, help='Show live training visualization dashboard')
@click.option('--use-torchrl-env', is_flag=True, help='Use TorchRL environment for training')
@click.pass_context
def train(ctx, ttl_file, dataset, episodes, output_model, sample_size, visualize, use_torchrl_env):
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
    # This is a basic implementation that looks for potential proper nouns
    words = text.split()
    entities = []

    # Look for capitalized words (original logic)
    for word in words:
        clean_word = word.strip('.,!?;:"()[]')
        if len(clean_word) > 2 and clean_word[0].isupper():
            entities.append(clean_word)

    # Also look for common geographical/person name patterns (case insensitive)
    text_lower = text.lower()
    common_entities = ['france', 'paris', 'london', 'england', 'shakespeare', 'einstein', 'newton']
    for entity in common_entities:
        if entity in text_lower:
            # Capitalize the first letter for KG lookup
            entities.append(entity.capitalize())

    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            unique_entities.append(entity)

    return unique_entities[:5]


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
    examples = list(dataset)
    if not examples:
        logger.error("No training examples found in dataset")
        return

    logger.info(f"Training with {len(examples)} examples from dataset")

    # Show first few examples for verification
    for i, example in enumerate(examples[:3]):
        logger.info(f"Example {i+1}: Question: '{example['question'][:80]}...', Answer: '{example.get('answer', 'N/A')[:50]}...'")

    # Give the environment access to all training examples
    agent.env.set_training_examples(examples)

    # Train with standard PPO - the environment will cycle through examples automatically
    try:
        from stable_baselines3.common.callbacks import BaseCallback

        class TrainingProgressCallback(BaseCallback):
            """Callback to show training progress with actual questions."""
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.episode_count = 0

            def _on_step(self) -> bool:
                # Each step is an episode since all actions are terminal
                self.episode_count += 1

                if self.episode_count % 200 == 0:
                    # Get current query from environment
                    try:
                        current_query = self.training_env.get_attr('current_context')[0].get('query', 'N/A')
                        logger.info(f"📊 Training progress: Episode {self.episode_count}, Current question: '{current_query[:60]}...'")
                    except Exception as e:
                        logger.debug(f"Could not get current query: {e}")

                return True

        progress_callback = TrainingProgressCallback()
        agent.model.learn(total_timesteps=total_timesteps, callback=progress_callback)
        logger.info("Dataset-aware PPO training completed successfully")
    except Exception as e:
        logger.error(f"Training with dataset failed: {e}")
        raise


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()