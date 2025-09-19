"""PPO agent for knowledge graph reasoning with memory capabilities and optional TorchRL support."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Any, Tuple, Optional
import logging
import warnings
from dataclasses import dataclass
from ..actions.action_space import ActionType
from ..config import Config, get_config, is_torchrl_enabled

# TorchRL imports with fallback
try:
    from ..envs.torchrl_kg_env import TorchRLKnowledgeGraphEnv, create_kg_environment
    from ..execution.torchrl_action_executor import TorchRLActionExecutor
    TORCHRL_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"TorchRL components not available: {e}")
    TORCHRL_AVAILABLE = False
    TorchRLKnowledgeGraphEnv = None
    TorchRLActionExecutor = None


logger = logging.getLogger(__name__)


@dataclass
class AgentMemory:
    """Memory structure for the RL agent."""
    states: List[Dict[str, Any]]
    actions: List[int]
    rewards: List[float]
    next_states: List[Dict[str, Any]]
    dones: List[bool]
    action_log_probs: List[float]
    state_values: List[float]


class KGReasoningEnvironment(gym.Env):
    """Custom environment for knowledge graph reasoning tasks."""

    def __init__(self, action_manager, reward_calculator, internal_kg, max_steps: int = 10,
                 training_monitor=None):
        """Initialize the KG reasoning environment.

        Args:
            action_manager: Manager for available actions
            reward_calculator: Calculator for reward computation
            internal_kg: Internal knowledge graph for memory
            max_steps: Maximum steps per episode
            training_monitor: Optional training monitor for logging
        """
        super().__init__()

        self.action_manager = action_manager
        self.reward_calculator = reward_calculator
        self.internal_kg = internal_kg
        self.max_steps = max_steps
        self.training_monitor = training_monitor

        # Dynamic action space based on available actions
        num_actions = len(self.action_manager.actions)
        self.action_space = gym.spaces.Discrete(num_actions)

        # Observation space: dictionary with various features
        self.observation_space = gym.spaces.Dict({
            "query_embedding": gym.spaces.Box(low=-1, high=1, shape=(384,), dtype=np.float32),
            "context_features": gym.spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32),
            "action_history": gym.spaces.Box(low=0, high=1, shape=(num_actions,), dtype=np.float32),
            "internal_kg_features": gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        })

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Initialize with a default context to prevent division by zero errors
        self.current_context = {
            "query": "",
            "entities": [],
            "expected_answer": None,
            "history": [],
            "failed_actions": 0,
            "internal_knowledge": "",
            "recent_actions": []  # Track action names for diversity calculation
        }

        # If we have training examples, set up the next one
        if hasattr(self, 'training_examples') and self.training_examples:
            self._setup_next_training_example()
        self.step_count = 0
        self.episode_reward = 0
        num_actions = len(self.action_manager.actions)
        self.action_history = [0] * num_actions  # Track recent actions dynamically
        self.conversation_history = []
        self.failed_actions = 0

        # Return initial observation
        return self._get_observation(), {}

    def step(self, action: int):
        """Execute one step in the environment.

        Args:
            action: Integer action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        action_type = ActionType(action)
        self.step_count += 1

        # 🎯 ENHANCED LOGGING: Show action selection decision
        print(f"\n{'='*80}")
        print(f"🎯 ACTION SELECTION - Step {self.step_count}")
        print(f"{'='*80}")
        print(f"📋 Query: '{self.current_context.get('query', 'N/A')}'")
        print(f"🎲 Action Chosen: {action_type.name} (ID: {action})")
        print(f"🔢 Available Actions: {[a.name for a in ActionType]}")
        
        # Show action recommendations if available
        try:
            recommendations = self.action_manager.get_action_recommendations(self.current_context)
            print(f"🎯 Action Confidence Scores:")
            for action_rec, confidence in recommendations:
                marker = "👉" if action_rec == action_type else "  "
                print(f"  {marker} {action_rec.name}: {confidence:.3f}")
        except:
            pass
        
        print(f"{'='*80}")

        # Update action history
        self.action_history = self.action_history[1:] + [action]

        # Execute action
        try:
            # Prepare context with internal knowledge
            context = self.action_manager.update_context_with_internal_knowledge(self.current_context)

            # Execute the action
            action_result = self.action_manager.execute_action(action_type, context)

            # Update recent actions for diversity tracking
            action_name = getattr(action_result, "metadata", {}).get("action", "")
            if action_name:
                context["recent_actions"] = context.get("recent_actions", []) + [action_name]
                # Keep only last 10 actions
                context["recent_actions"] = context["recent_actions"][-10:]

            # Calculate reward with updated context
            reward_components = self.reward_calculator.calculate_reward(context, action_result)
            reward = reward_components.total

            # 🏆 ENHANCED LOGGING: Show reward calculation and action results
            print(f"\n{'='*80}")
            print(f"🏆 ACTION RESULTS & REWARD CALCULATION")
            print(f"{'='*80}")
            print(f"✅ Action Executed: {action_type.name}")
            print(f"📈 Action Success: {'✅ Yes' if action_result.success else '❌ No'}")
            print(f"💭 Action Response: {action_result.response[:200]}{'...' if len(action_result.response) > 200 else ''}")
            print(f"📊 Reward Components:")
            print(f"   🎯 Semantic Similarity: {reward_components.semantic_similarity:.3f}")
            print(f"   ✅ Action Success: {reward_components.action_success:.3f}")
            print(f"   📚 Knowledge Gain: {reward_components.knowledge_gain:.3f}")
            print(f"   ⚡ Efficiency: {reward_components.efficiency:.3f}")
            print(f"   😊 User Satisfaction: {reward_components.user_satisfaction:.3f}")
            print(f"   🎲 Action Diversity: {reward_components.action_diversity:.3f}")
            print(f"🏆 Total Reward: {reward:.3f}")
            print(f"📊 Episode Stats: Step {self.step_count}/{self.max_steps}, Failed: {self.failed_actions}")
            print(f"{'='*80}\n")

            # Log detailed training information for debugging
            logger.info(f"TRAINING DEBUG - Action: {action_type.name}")
            logger.info(f"TRAINING DEBUG - Query: '{context.get('query', 'N/A')}'")
            logger.info(f"TRAINING DEBUG - Response: '{action_result.response[:100]}...'")
            logger.info(f"TRAINING DEBUG - Reward Components: semantic={reward_components.semantic_similarity:.3f}, "
                       f"success={reward_components.action_success:.3f}, diversity={reward_components.action_diversity:.3f}")
            logger.info(f"TRAINING DEBUG - Total Reward: {reward:.3f}")
            logger.info("---")

            # Update episode tracking
            self.episode_reward += reward
            if not action_result.success:
                self.failed_actions += 1

            # Log action result if monitor is available
            if self.training_monitor:
                self.training_monitor.log_action_result(action_result.success)

            # Storage is now handled within individual actions when appropriate
            # No need for automatic storage after each action

            # Update context for next step
            self.current_context.update({
                "last_action": action_type,
                "last_result": action_result,
                "failed_actions": self.failed_actions,
                "step_count": self.step_count
            })

            # Episode termination conditions - ALL actions now end the episode
            # since they all provide final user responses
            terminated = True  # Every action is now final
            truncated = self.step_count >= self.max_steps

            # No need for early termination logic since all actions are terminal

            info = {
                "action_type": action_type.name,
                "action_success": action_result.success,
                "action_response": action_result.response,
                "reward_components": reward_components,
                "step_count": self.step_count,
                "failed_actions": self.failed_actions
            }

            return self._get_observation(), reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"Environment step failed: {e}")

            # Return negative reward for errors
            reward = -0.5
            terminated = True
            info = {"error": str(e), "action_type": action_type.name}

            return self._get_observation(), reward, terminated, False, info

    def set_query(self, query: str, entities: Optional[List[str]] = None,
                  expected_answer: Optional[str] = None):
        """Set the query for the current episode.

        Args:
            query: Natural language query
            entities: Detected entities in the query
            expected_answer: Ground truth answer for evaluation
        """
        self.current_context = {
            "query": query,
            "entities": entities or [],
            "expected_answer": expected_answer,
            "history": self.conversation_history,
            "failed_actions": 0
        }

        # Log query if monitor is available
        if self.training_monitor and query:
            self.training_monitor.log_query(query)

    def set_training_examples(self, examples: List[Dict[str, Any]]):
        """Set training examples for the environment to cycle through.

        Args:
            examples: List of training examples with 'question', 'answer', etc.
        """
        self.training_examples = examples
        self.current_example_idx = 0
        logger.info(f"Set {len(examples)} training examples")

    def _setup_next_training_example(self):
        """Setup the next training example from the dataset."""
        if not hasattr(self, 'training_examples') or not self.training_examples:
            return

        example = self.training_examples[self.current_example_idx]
        self.set_query(
            example['question'],
            example.get('entities', []),
            example.get('answer', '')
        )

        # Move to next example for next episode
        self.current_example_idx = (self.current_example_idx + 1) % len(self.training_examples)

        logger.debug(f"Setup training example: '{example['question'][:50]}...'")

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation state.

        Returns:
            Dictionary containing observation features
        """
        # Get query embedding
        query = self.current_context.get("query", "")
        if query:
            try:
                query_embedding = self.reward_calculator.sentence_transformer.encode([query])[0]
            except:
                query_embedding = np.zeros(384)
        else:
            query_embedding = np.zeros(384)

        # Context features
        context_features = self._extract_context_features()

        # Action history (one-hot encoded recent actions)
        num_actions = len(self.action_manager.actions)
        action_hist = np.zeros(num_actions)
        for i, action in enumerate(self.action_history):
            if i < len(action_hist):
                action_hist[i] = action / num_actions  # Normalize by number of actions

        # Internal KG features
        internal_kg_features = self._extract_internal_kg_features()

        return {
            "query_embedding": query_embedding.astype(np.float32),
            "context_features": context_features.astype(np.float32),
            "action_history": action_hist.astype(np.float32),
            "internal_kg_features": internal_kg_features.astype(np.float32)
        }

    def _extract_context_features(self) -> np.ndarray:
        """Extract numerical features from current context."""
        features = np.zeros(20)

        query = self.current_context.get("query", "")
        entities = self.current_context.get("entities", [])

        # Query features - add safety checks for division by zero
        features[0] = min(len(query.split()) if query else 0, 50) / 50  # Query length (normalized)
        features[1] = len(entities) / 10  # Number of entities (normalized)
        features[2] = self.failed_actions / 5  # Failed actions ratio
        features[3] = self.step_count / max(self.max_steps, 1)  # Episode progress (prevent div by 0)

        # Query type indicators
        query_lower = query.lower()
        features[4] = 1.0 if "what" in query_lower else 0.0
        features[5] = 1.0 if "who" in query_lower else 0.0
        features[6] = 1.0 if "when" in query_lower else 0.0
        features[7] = 1.0 if "where" in query_lower else 0.0
        features[8] = 1.0 if "how" in query_lower else 0.0
        features[9] = 1.0 if "why" in query_lower else 0.0

        # Complexity indicators
        features[10] = 1.0 if "explain" in query_lower else 0.0
        features[11] = 1.0 if "compare" in query_lower else 0.0
        features[12] = 1.0 if "analyze" in query_lower else 0.0

        # Context availability
        features[13] = 1.0 if self.current_context.get("expected_answer") else 0.0
        features[14] = min(len(self.conversation_history), 10) / 10  # History length
        features[15] = 1.0 if self.current_context.get("internal_knowledge") else 0.0

        # Recent performance
        if hasattr(self, 'episode_reward'):
            features[16] = max(0, self.episode_reward) / 5  # Positive reward (normalized)
            features[17] = max(0, -self.episode_reward) / 5  # Negative reward (normalized)

        # Action applicability (simplified)
        applicable_actions = self.action_manager.get_applicable_actions(self.current_context)
        features[18] = len(applicable_actions) / len(ActionType)

        return features

    def _extract_internal_kg_features(self) -> np.ndarray:
        """Extract features from internal knowledge graph."""
        features = np.zeros(10)

        try:
            stats = self.internal_kg.get_stats()

            # Basic stats (normalized)
            features[0] = min(stats.get("total_nodes", 0), 1000) / 1000
            features[1] = min(stats.get("total_edges", 0), 2000) / 2000
            features[2] = min(stats.get("total_contexts", 0), 500) / 500

            # Average importance
            features[3] = max(0, min(stats.get("avg_node_importance", 0), 5)) / 5

            # Type diversity
            node_types = len(stats.get("node_types", {}))
            features[4] = min(node_types, 20) / 20

            relation_types = len(stats.get("relation_types", {}))
            features[5] = min(relation_types, 30) / 30

            # Query-relevant nodes
            entities = self.current_context.get("entities", [])
            if entities:
                relevant_nodes = self.internal_kg.get_relevant_nodes(entities, limit=5)
                features[6] = len(relevant_nodes) / 5

                # Similar contexts
                query = self.current_context.get("query", "")
                if query:  # Only search if query exists
                    similar_contexts = self.internal_kg.get_similar_contexts(query, limit=3)
                    features[7] = len(similar_contexts) / 3

        except Exception as e:
            logger.warning(f"Failed to extract internal KG features: {e}")

        return features

    def set_training_examples(self, examples: List[Dict[str, Any]]):
        """Set training examples for the environment to cycle through.

        Args:
            examples: List of training examples with 'question', 'answer', etc.
        """
        self.training_examples = examples
        self.current_example_idx = 0
        logger.info(f"Environment: Set {len(examples)} training examples")

    def _setup_next_training_example(self):
        """Setup the next training example from the dataset."""
        if not hasattr(self, 'training_examples') or not self.training_examples:
            return

        example = self.training_examples[self.current_example_idx]
        self.set_query(
            example['question'],
            example.get('entities', []),
            example.get('answer', '')
        )

        # Move to next example for next episode
        self.current_example_idx = (self.current_example_idx + 1) % len(self.training_examples)

        logger.debug(f"Setup training example: '{example['question'][:50]}...'")


def create_environment(
    action_manager,
    reward_calculator, 
    internal_kg,
    kg_loader=None,
    config: Optional[Config] = None,
    use_torchrl: Optional[bool] = None,
    **kwargs
):
    """
    Factory function to create KG environment with optional TorchRL support.
    
    This function maintains backward compatibility while allowing opt-in
    TorchRL features for enhanced capabilities.
    
    Args:
        action_manager: Action manager for executing actions
        reward_calculator: Reward calculator for environment feedback
        internal_kg: Internal knowledge graph for memory
        kg_loader: Knowledge graph loader (required for TorchRL features)
        config: Configuration object, loaded from file if not provided
        use_torchrl: Override config setting for TorchRL usage
        **kwargs: Additional arguments passed to environment
        
    Returns:
        Environment instance (TorchRL-enhanced or standard)
    """
    # Load config if not provided
    if config is None:
        config = get_config()
    
    # Determine if TorchRL should be used
    if use_torchrl is None:
        use_torchrl = is_torchrl_enabled(config)
    
    # Check TorchRL availability
    if use_torchrl and not TORCHRL_AVAILABLE:
        warnings.warn(
            "TorchRL requested but not available. "
            "Install with: pip install 'torchrl[llm]'. "
            "Falling back to standard environment."
        )
        use_torchrl = False
    
    if use_torchrl:
        logger.info("Creating TorchRL-enhanced environment")
        
        try:
            # Create TorchRL environment with enhanced features
            env = TorchRLKnowledgeGraphEnv(
                kg_loader=kg_loader,
                action_manager=action_manager,
                reward_calculator=reward_calculator,
                internal_kg=internal_kg,
                tokenizer_name=config.torchrl.tokenizer_name,
                max_steps=config.environment_config.get("max_steps", 10),
                device=config.torchrl.device,
                enable_tools=config.torchrl.enable_tool_enhancement,
                **kwargs
            )
            
            # Add transforms if enabled
            if config.torchrl.enable_tool_enhancement:
                from ..transforms.kg_transform import KnowledgeGraphTransform
                from ..transforms.hybrid_reward_transform import HybridRewardTransform
                
                # Add KG transform
                kg_transform = KnowledgeGraphTransform(
                    kg_loader=kg_loader,
                    internal_kg=internal_kg,
                    enable_sparql_tools=config.torchrl.enable_enhanced_sparql
                )
                env.append_transform(kg_transform)
                
                # Add hybrid reward transform
                reward_transform = HybridRewardTransform(
                    base_reward_calculator=reward_calculator,
                    tool_success_weight=config.torchrl.tool_success_weight,
                    knowledge_gain_weight=config.torchrl.knowledge_gain_weight,
                    efficiency_weight=config.torchrl.efficiency_weight,
                    enable_detailed_logging=config.torchrl.enable_detailed_logging
                )
                env.append_transform(reward_transform)
                
                logger.info("TorchRL transforms added successfully")
            
            return env
            
        except Exception as e:
            logger.error(f"Failed to create TorchRL environment: {e}")
            logger.info("Falling back to standard environment")
            use_torchrl = False
    
    if not use_torchrl:
        logger.info("Creating standard KG environment")
        
        # Create standard environment with backward compatibility
        return KGReasoningEnvironment(
            action_manager=action_manager,
            reward_calculator=reward_calculator,
            internal_kg=internal_kg,
            max_steps=config.environment_config.get("max_steps", 10),
            **kwargs
        )


def create_action_executor(
    action_manager,
    kg_loader=None,
    internal_kg=None,
    llm_client=None,
    config: Optional[Config] = None,
    use_torchrl: Optional[bool] = None
):
    """
    Factory function to create action executor with optional TorchRL enhancement.
    
    Args:
        action_manager: Base action manager
        kg_loader: Knowledge graph loader
        internal_kg: Internal knowledge graph
        llm_client: LLM client for enhanced responses
        config: Configuration object
        use_torchrl: Override config setting for TorchRL usage
        
    Returns:
        Action executor (enhanced or standard)
    """
    # Load config if not provided
    if config is None:
        config = get_config()
    
    # Determine if TorchRL should be used
    if use_torchrl is None:
        use_torchrl = is_torchrl_enabled(config) and config.torchrl.enable_tool_enhancement
    
    if use_torchrl and TORCHRL_AVAILABLE:
        logger.info("Creating TorchRL-enhanced action executor")
        
        return TorchRLActionExecutor(
            action_manager=action_manager,
            kg_loader=kg_loader,
            internal_kg=internal_kg,
            llm_client=llm_client,
            enable_tool_enhancement=config.torchrl.enable_tool_enhancement,
            tool_timeout=config.torchrl.tool_timeout
        )
    else:
        logger.info("Using standard action manager")
        return action_manager


class PPOKGAgent:
    """PPO agent specialized for knowledge graph reasoning."""

    def __init__(self, action_manager, reward_calculator, internal_kg,
                 learning_rate: float = 3e-4, n_steps: int = 2048,
                 batch_size: int = 64, n_epochs: int = 10,
                 clip_range: float = 0.2, ent_coef: float = 0.01,
                 env=None):
        """Initialize PPO agent.

        Args:
            action_manager: Action manager instance
            reward_calculator: Reward calculator instance
            internal_kg: Internal knowledge graph instance
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to collect before training
            batch_size: Batch size for training
            n_epochs: Number of training epochs per update
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient for exploration
            env: Optional environment to use (if None, creates standard environment)
        """
        self.action_manager = action_manager
        self.reward_calculator = reward_calculator
        self.internal_kg = internal_kg

        # Create environment (will be updated with training monitor when training starts)
        if env is not None:
            self.env = env
        else:
            self.env = KGReasoningEnvironment(action_manager, reward_calculator, internal_kg)
        
        # Set agent reference on environment for training coordination
        self.env.agent = self

        # PPO configuration
        self.ppo_config = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "verbose": 1,
            "device": "auto"
        }

        # Initialize PPO model
        self.model = None
        self._initialize_model()

        # Training tracking
        self.training_steps = 0
        self.episodes_completed = 0

    def set_training_examples(self, examples: List[Dict[str, Any]]):
        """Set training examples for both agent and environment.

        Args:
            examples: List of training examples with 'question', 'answer', etc.
        """
        self.training_examples = examples
        
        # Also set on environment
        if hasattr(self.env, 'set_training_examples'):
            self.env.set_training_examples(examples)
        
        logger.info(f"PPOKGAgent: Set {len(examples)} training examples")

    def _initialize_model(self):
        """Initialize the PPO model."""
        try:
            self.model = PPO(
                "MultiInputPolicy",
                self.env,
                **self.ppo_config
            )
            logger.info("PPO model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PPO model: {e}")
            raise

    def train(self, total_timesteps: int = 10000, save_path: Optional[str] = None,
             training_monitor=None):
        """Train the PPO agent.

        Args:
            total_timesteps: Total timesteps for training
            save_path: Optional path to save the model
            training_monitor: Optional training monitor for visualization
        """
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")

        try:
            # Update environment with training monitor
            if training_monitor:
                self.env.training_monitor = training_monitor

            # Custom callback for logging with visualization support
            callback = TrainingCallback(save_path, training_monitor)

            # Train the model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback
            )

            self.training_steps += total_timesteps
            logger.info(f"Training completed. Total steps: {self.training_steps}")

            # Save model if path provided
            if save_path:
                self.save_model(save_path)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def predict_action(self, query: str, entities: Optional[List[str]] = None,
                      expected_answer: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
        """Predict best action for a given query.

        Args:
            query: Natural language query
            entities: Detected entities
            expected_answer: Ground truth answer for evaluation

        Returns:
            Tuple of (action_index, prediction_info)
        """
        # Set up environment with query
        self.env.set_query(query, entities, expected_answer)
        obs, _ = self.env.reset()

        # Predict action
        action, _states = self.model.predict(obs, deterministic=True)

        # Get action probabilities for additional info
        try:
            obs_tensor = {k: torch.tensor([v]) for k, v in obs.items()}
            with torch.no_grad():
                logits = self.model.policy.get_distribution(obs_tensor).distribution.logits
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        except:
            probs = np.ones(len(ActionType)) / len(ActionType)

        prediction_info = {
            "action_probabilities": {ActionType(i).name: probs[i] for i in range(len(ActionType))},
            "predicted_action": ActionType(action).name,
            "observation": obs
        }

        return action, prediction_info

    def execute_query(self, query: str, entities: Optional[List[str]] = None,
                     expected_answer: Optional[str] = None, max_steps: int = 5) -> Dict[str, Any]:
        """Execute a full query using the trained agent.

        Args:
            query: Natural language query
            entities: Detected entities
            expected_answer: Ground truth for evaluation
            max_steps: Maximum steps to take

        Returns:
            Dictionary with execution results
        """
        # Set up environment
        self.env.set_query(query, entities, expected_answer)
        obs, _ = self.env.reset()

        steps_taken = []
        total_reward = 0
        terminated = False

        for step in range(max_steps):
            if terminated:
                break

            # Predict and execute action
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)

            total_reward += reward
            steps_taken.append({
                "step": step,
                "action": ActionType(action).name,
                "reward": reward,
                "success": info.get("action_success", False),
                "response": info.get("action_response", ""),
                "info": info
            })

            if terminated or truncated:
                break

        return {
            "query": query,
            "steps_taken": steps_taken,
            "total_reward": total_reward,
            "terminated": terminated,
            "final_step": len(steps_taken)
        }

    def save_model(self, path: str):
        """Save the trained model."""
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, path: str):
        """Load a trained model."""
        try:
            self.model = PPO.load(path, env=self.env)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def update_training_config(self, **kwargs):
        """Update training configuration."""
        self.ppo_config.update(kwargs)
        logger.info(f"Training config updated: {kwargs}")


class TrainingCallback(BaseCallback):
    """Callback for monitoring training progress."""

    def __init__(self, save_path: Optional[str] = None, training_monitor=None):
        super().__init__()
        self.save_path = save_path
        self.training_monitor = training_monitor
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if episode ended
        if 'episode_reward' in self.locals:
            episode_reward = self.locals['episode_reward']
            episode_length = self.locals.get('episode_length', 1)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_count += 1

            # Track success (positive reward)
            if episode_reward > 0:
                self.success_count += 1

            # Update training monitor if available
            if self.training_monitor:
                success_rate = self.success_count / max(self.episode_count, 1) * 100

                # Get policy metrics if available
                entropy = getattr(self.model, 'entropy_coef', 0.0)
                policy_loss = getattr(self.locals.get('policy_loss', 0), 'item', lambda: 0)()
                value_loss = getattr(self.locals.get('value_loss', 0), 'item', lambda: 0)()
                learning_rate = getattr(self.model, 'learning_rate', 3e-4)

                self.training_monitor.log_episode(
                    reward=episode_reward,
                    episode_length=episode_length,
                    success_rate=success_rate,
                    entropy=entropy,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    learning_rate=learning_rate
                )

        # Log training progress periodically
        if len(self.episode_rewards) > 0 and len(self.episode_rewards) % 100 == 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            logger.info(f"Episodes: {len(self.episode_rewards)}, "
                       f"Avg Reward: {avg_reward:.3f}, "
                       f"Avg Length: {avg_length:.1f}")

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if hasattr(self.locals, 'episode_reward'):
            self.episode_rewards.append(self.locals['episode_reward'])
        if hasattr(self.locals, 'episode_length'):
            self.episode_lengths.append(self.locals['episode_length'])