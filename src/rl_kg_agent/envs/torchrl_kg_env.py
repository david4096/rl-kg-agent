"""TorchRL-compatible Knowledge Graph Environment."""

import warnings
from typing import Dict, Any, Optional, Tuple, List
import gymnasium as gym
import torch
import numpy as np

try:
    from tensordict import TensorDict
    from torchrl.envs.llm import ChatEnv
    from torchrl.data import CompositeSpec, Unbounded, Bounded
    from transformers import AutoTokenizer
    TORCHRL_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"TorchRL dependencies not available: {e}. TorchRL features will be disabled.")
    TORCHRL_AVAILABLE = False
    # Create dummy classes for type hints
    class ChatEnv:
        pass
    class TensorDict:
        pass
    class CompositeSpec:
        pass

from ..actions.action_space import ActionType


class TorchRLKnowledgeGraphEnv(ChatEnv if TORCHRL_AVAILABLE else object):
    """
    TorchRL-compatible environment that maintains the existing 5-action discrete space
    while adding tool capabilities through transforms.
    
    This environment wraps the existing KGReasoningEnvironment to provide:
    - Backward compatibility with existing PPO training
    - TorchRL tool integration capabilities  
    - Enhanced observation and reward spaces
    - Seamless transform composition
    """
    
    def __init__(
        self,
        kg_loader,
        action_manager,
        reward_calculator,
        internal_kg,
        tokenizer_name: str = "microsoft/DialoGPT-medium",
        max_steps: int = 10,
        device: Optional[torch.device] = None,
        enable_tools: bool = True,
        **kwargs
    ):
        """
        Initialize TorchRL-compatible KG environment.
        
        Args:
            kg_loader: Knowledge graph loader
            action_manager: Action manager for executing actions
            reward_calculator: Reward calculator 
            internal_kg: Internal knowledge graph
            tokenizer_name: Tokenizer for conversation handling
            max_steps: Maximum steps per episode
            device: Device for tensor operations
            enable_tools: Whether to enable TorchRL tool capabilities
            **kwargs: Additional arguments for ChatEnv
        """
        if not TORCHRL_AVAILABLE:
            raise ImportError(
                "TorchRL dependencies not available. Install with: pip install 'torchrl[llm]'"
            )
            
        self.kg_loader = kg_loader
        self.action_manager = action_manager
        self.reward_calculator = reward_calculator
        self.internal_kg = internal_kg
        self.max_steps = max_steps
        self.enable_tools = enable_tools
        self.device = device or torch.device("cpu")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            warnings.warn(f"Could not load tokenizer {tokenizer_name}: {e}. Using dummy tokenizer.")
            self.tokenizer = None
        
        # Initialize ChatEnv with conversation capabilities
        super().__init__(
            batch_size=(1,),
            tokenizer=self.tokenizer,
            system_prompt=self._get_system_prompt(),
            device=self.device,
            **kwargs
        )
        
        # Set up action and observation spaces to maintain compatibility
        self._setup_spaces()
        
        # Initialize episode state
        self.current_step = 0
        self.episode_history = []
        self.current_query = ""
        self.current_context = {}
        
    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return (
            "You are a knowledgeable assistant with access to knowledge graphs and tools. "
            "Your goal is to provide accurate, helpful responses by leveraging available "
            "information sources and tools when appropriate."
        )
    
    def _setup_spaces(self):
        """Set up action and observation spaces maintaining PPO compatibility."""
        # CRITICAL: Use dynamic action space based on available actions
        num_actions = len(self.action_manager.actions) if hasattr(self.action_manager, 'actions') else 5
        self.action_space = gym.spaces.Discrete(num_actions)
        
        # Enhanced observation space that includes tool context
        self.observation_space = gym.spaces.Dict({
            "query_embedding": gym.spaces.Box(
                low=-1.0, high=1.0, shape=(768,), dtype=np.float32
            ),
            "kg_context_embedding": gym.spaces.Box(
                low=-1.0, high=1.0, shape=(512,), dtype=np.float32
            ),
            "conversation_history": gym.spaces.Box(
                low=0, high=1, shape=(10,), dtype=np.float32  # Binary flags for history
            ),
            "step_count": gym.spaces.Box(
                low=0, high=self.max_steps, shape=(1,), dtype=np.float32
            ),
            "internal_kg_size": gym.spaces.Box(
                low=0, high=10000, shape=(1,), dtype=np.float32
            )
        })
        
        # TorchRL specs for enhanced functionality
        if TORCHRL_AVAILABLE:
            self._setup_torchrl_specs()
    
    def _setup_torchrl_specs(self):
        """Set up TorchRL-specific specs."""
        # Get dynamic action count
        num_actions = len(self.action_manager.actions) if hasattr(self.action_manager, 'actions') else 5
        
        # Input spec includes query and context
        self.input_spec = CompositeSpec({
            "query": Unbounded(shape=(1,), dtype=torch.object),
            "context": Unbounded(shape=(1,), dtype=torch.object),
        })
        
        # Output spec includes response and metadata
        self.output_spec = CompositeSpec({
            "response": Unbounded(shape=(1,), dtype=torch.object),
            "action_taken": Bounded(low=0, high=num_actions-1, shape=(1,), dtype=torch.long),
            "success": Bounded(low=0, high=1, shape=(1,), dtype=torch.bool),
            "confidence": Bounded(low=0.0, high=1.0, shape=(1,), dtype=torch.float32),
        })
        
        # Reward spec
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
    
    def reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        """Reset environment and return initial observation."""
        # Reset episode state
        self.current_step = 0
        self.episode_history = []
        self.current_query = ""
        self.current_context = {}
        
        # Call agent's reset method to setup next training example
        if hasattr(self, 'agent') and self.agent:
            self.agent.reset()
            # Get the query from the agent after it sets up the next training example
            if hasattr(self.agent, 'current_context') and 'query' in self.agent.current_context:
                self.current_query = self.agent.current_context['query']
        
        # Extract query from tensordict if provided (this overrides agent query)
        if tensordict is not None and "query" in tensordict:
            self.current_query = tensordict["query"][0] if hasattr(tensordict["query"], '__getitem__') else str(tensordict["query"])
        
        # Create initial observation
        obs = self._get_observation()
        
        # Create TensorDict for return
        if TORCHRL_AVAILABLE:
            result = TensorDict({
                "observation": obs,
                "query": self.current_query,
                "step_count": torch.tensor([0], dtype=torch.long),
                "done": torch.tensor([False], dtype=torch.bool),
            }, batch_size=(1,))
        else:
            result = {"observation": obs}
            
        return result
    
    def step(self, action: torch.Tensor) -> Tuple[TensorDict, TensorDict]:
        """Execute action and return next state."""
        if isinstance(action, torch.Tensor):
            action_id = action.item()
        else:
            action_id = int(action)
            
        # Validate action
        if not 0 <= action_id < len(self.action_manager.actions):
            raise ValueError(f"Invalid action {action_id}. Must be in range [0, {len(self.action_manager.actions) - 1}]")
        
        # Map action ID to ActionType
        action_type = ActionType(action_id)
        
        # Execute action using existing action manager
        action_result = self._execute_action(action_type)
        
        # Calculate reward
        reward = self._calculate_reward(action_result)
        
        # Update episode state
        self.current_step += 1
        self.episode_history.append({
            "action": action_type,
            "result": action_result,
            "reward": reward
        })
        
        # Check if episode is done
        done = (
            self.current_step >= self.max_steps or
            action_result.is_final or
            not action_result.success
        )
        
        # Create next observation
        next_obs = self._get_observation()
        
        # Create TensorDict results
        if TORCHRL_AVAILABLE:
            # Current state
            current_td = TensorDict({
                "observation": self._get_observation(),
                "action": torch.tensor([action_id], dtype=torch.long),
                "step_count": torch.tensor([self.current_step - 1], dtype=torch.long),
            }, batch_size=(1,))
            
            # Next state  
            next_td = TensorDict({
                "observation": next_obs,
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done], dtype=torch.bool),
                "response": action_result.response,
                "action_taken": torch.tensor([action_id], dtype=torch.long),
                "success": torch.tensor([action_result.success], dtype=torch.bool),
                "confidence": torch.tensor([action_result.confidence], dtype=torch.float32),
                "step_count": torch.tensor([self.current_step], dtype=torch.long),
            }, batch_size=(1,))
        else:
            current_td = {"observation": self._get_observation()}
            next_td = {
                "observation": next_obs,
                "reward": reward,
                "done": done,
                "response": action_result.response
            }
        
        return current_td, next_td
    
    def _execute_action(self, action_type: ActionType) -> Any:
        """Execute action using the existing action manager."""
        # Prepare context for action execution
        context = {
            "query": self.current_query,
            "entities": self.current_context.get("entities", []),
            "internal_knowledge": self._get_relevant_internal_knowledge(),
            "step_count": self.current_step,
            "history": self.episode_history
        }
        
        # Execute action
        try:
            action_result = self.action_manager.execute_action(action_type, context)
            return action_result
        except Exception as e:
            # Return error result to maintain stability
            from ..actions.action_space import ActionResult
            return ActionResult(
                success=False,
                response=f"Action execution failed: {str(e)}",
                metadata={"error": str(e)},
                confidence=0.0
            )
    
    def _get_relevant_internal_knowledge(self) -> str:
        """Get relevant knowledge from internal KG."""
        try:
            if hasattr(self.internal_kg, 'query_relevant_context'):
                return self.internal_kg.query_relevant_context(self.current_query)
            elif hasattr(self.internal_kg, 'get_context'):
                contexts = self.internal_kg.get_context(limit=3)
                return "\\n".join([ctx.get("query", "") + ": " + ctx.get("response", "") for ctx in contexts])
            else:
                return ""
        except Exception:
            return ""
    
    def _calculate_reward(self, action_result) -> float:
        """Calculate reward using existing reward calculator."""
        try:
            # Use existing reward calculation
            reward_context = {
                "query": self.current_query,
                "response": action_result.response,
                "action_success": action_result.success,
                "confidence": action_result.confidence,
                "entities_found": len(action_result.entities_discovered),
                "metadata": action_result.metadata
            }
            
            reward = self.reward_calculator.calculate_reward(reward_context)
            
            # Add tool-specific bonuses if using TorchRL features
            if self.enable_tools and action_result.success:
                # Small bonus for successful tool usage
                if "tool" in action_result.metadata:
                    reward += 0.1
                # Bonus for discovering new entities
                if action_result.entities_discovered:
                    reward += 0.05 * len(action_result.entities_discovered)
            
            return float(reward)
            
        except Exception as e:
            # Fallback reward calculation
            if action_result.success:
                return action_result.confidence * 0.5
            else:
                return -0.1
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Create embeddings (simplified for now)
        query_embedding = self._embed_text(self.current_query)
        kg_context_embedding = self._embed_text(self._get_relevant_internal_knowledge())
        
        # Conversation history flags (last 10 actions)
        history_flags = np.zeros(10, dtype=np.float32)
        for i, entry in enumerate(self.episode_history[-10:]):
            history_flags[i] = 1.0
        
        obs = {
            "query_embedding": query_embedding,
            "kg_context_embedding": kg_context_embedding,
            "conversation_history": history_flags,
            "step_count": np.array([self.current_step], dtype=np.float32),
            "internal_kg_size": np.array([self._get_internal_kg_size()], dtype=np.float32)
        }
        
        return obs
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Create text embedding (simplified)."""
        if not text:
            return np.zeros(768 if "query" in str(text) else 512, dtype=np.float32)
        
        # Simple hash-based embedding for now
        hash_val = hash(text)
        size = 768 if len(text) > 50 else 512
        
        # Create deterministic embedding based on hash
        np.random.seed(abs(hash_val) % (2**31))
        embedding = np.random.normal(0, 0.1, size).astype(np.float32)
        np.random.seed()  # Reset seed
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _get_internal_kg_size(self) -> float:
        """Get size of internal knowledge graph."""
        try:
            if hasattr(self.internal_kg, 'size'):
                return float(self.internal_kg.size())
            elif hasattr(self.internal_kg, '__len__'):
                return float(len(self.internal_kg))
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def close(self):
        """Clean up environment."""
        if hasattr(super(), 'close'):
            super().close()


# Backward compatibility function
def create_kg_environment(kg_loader, action_manager, reward_calculator, internal_kg, 
                         use_torchrl: bool = False, **kwargs):
    """
    Factory function to create KG environment with optional TorchRL support.
    
    Args:
        kg_loader: Knowledge graph loader
        action_manager: Action manager
        reward_calculator: Reward calculator
        internal_kg: Internal knowledge graph
        use_torchrl: Whether to use TorchRL environment
        **kwargs: Additional arguments
        
    Returns:
        Environment instance (TorchRL or standard)
    """
    if use_torchrl and TORCHRL_AVAILABLE:
        return TorchRLKnowledgeGraphEnv(
            kg_loader=kg_loader,
            action_manager=action_manager,
            reward_calculator=reward_calculator,
            internal_kg=internal_kg,
            **kwargs
        )
    else:
        # Return existing environment for backward compatibility
        from ..agents.ppo_agent import KGReasoningEnvironment
        return KGReasoningEnvironment(
            action_manager=action_manager,
            reward_calculator=reward_calculator,
            internal_kg=internal_kg,
            **kwargs
        )