"""Hybrid Reward Transform combining semantic similarity with tool execution rewards."""

import warnings
from typing import Dict, Any, Optional
import torch

try:
    from tensordict import TensorDict
    from torchrl.envs import Transform
    from torchrl.data import CompositeSpec, Unbounded, Bounded
    TORCHRL_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"TorchRL dependencies not available: {e}")
    TORCHRL_AVAILABLE = False
    
    # Dummy base class for type hints
    class Transform:
        pass


class HybridRewardTransform(Transform if TORCHRL_AVAILABLE else object):
    """
    Transform that combines existing semantic similarity rewards with
    tool execution success rewards for enhanced RL training.
    
    This transform maintains backward compatibility with existing reward
    calculation while adding bonuses for successful tool usage.
    """
    
    def __init__(
        self,
        base_reward_calculator=None,
        tool_success_weight: float = 0.1,
        knowledge_gain_weight: float = 0.05,
        efficiency_weight: float = 0.02,
        enable_detailed_logging: bool = False
    ):
        """
        Initialize hybrid reward transform.
        
        Args:
            base_reward_calculator: Existing reward calculator to maintain compatibility
            tool_success_weight: Weight for tool execution success bonus
            knowledge_gain_weight: Weight for knowledge discovery bonus
            efficiency_weight: Weight for action efficiency bonus
            enable_detailed_logging: Whether to log detailed reward calculations
        """
        if TORCHRL_AVAILABLE:
            super().__init__()
            
        self.base_reward_calculator = base_reward_calculator
        self.tool_success_weight = tool_success_weight
        self.knowledge_gain_weight = knowledge_gain_weight
        self.efficiency_weight = efficiency_weight
        self.enable_detailed_logging = enable_detailed_logging
        
        # Reward history for analysis
        self.reward_history = []
    
    def _call(self, tensordict: TensorDict) -> TensorDict:
        """
        Calculate hybrid reward and add to tensordict.
        
        Args:
            tensordict: Input tensordict containing state and action information
            
        Returns:
            Tensordict with enhanced reward calculation
        """
        if not TORCHRL_AVAILABLE:
            return tensordict
            
        # Calculate base reward using existing system
        base_reward = self._calculate_base_reward(tensordict)
        
        # Calculate tool-specific bonuses
        tool_bonus = self._calculate_tool_bonus(tensordict)
        knowledge_bonus = self._calculate_knowledge_bonus(tensordict)
        efficiency_bonus = self._calculate_efficiency_bonus(tensordict)
        
        # Combine rewards
        total_reward = base_reward + tool_bonus + knowledge_bonus + efficiency_bonus
        
        # Ensure reward is within reasonable bounds
        total_reward = torch.clamp(total_reward, -2.0, 2.0)
        
        # Add detailed reward breakdown for analysis
        reward_breakdown = {
            "base_reward": float(base_reward),
            "tool_bonus": float(tool_bonus), 
            "knowledge_bonus": float(knowledge_bonus),
            "efficiency_bonus": float(efficiency_bonus),
            "total_reward": float(total_reward)
        }
        
        # Store in tensordict
        tensordict["reward"] = total_reward.unsqueeze(0) if total_reward.dim() == 0 else total_reward
        tensordict["reward_breakdown"] = reward_breakdown
        
        # Log if enabled
        if self.enable_detailed_logging:
            self.reward_history.append(reward_breakdown)
            if len(self.reward_history) % 100 == 0:  # Log every 100 steps
                self._log_reward_statistics()
        
        return tensordict
    
    def _calculate_base_reward(self, tensordict: TensorDict) -> torch.Tensor:
        """Calculate base reward using existing reward calculator."""
        
        # Extract necessary information from tensordict
        query = tensordict.get("query", "")
        response = tensordict.get("response", "")
        action_success = tensordict.get("success", torch.tensor(True))
        confidence = tensordict.get("confidence", torch.tensor(0.5))
        
        # If we have the base reward calculator, use it
        if self.base_reward_calculator:
            try:
                reward_context = {
                    "query": str(query),
                    "response": str(response),
                    "action_success": bool(action_success),
                    "confidence": float(confidence),
                    "entities_found": tensordict.get("entities_discovered", []),
                    "metadata": tensordict.get("metadata", {})
                }
                
                base_reward = self.base_reward_calculator.calculate_reward(reward_context)
                return torch.tensor(base_reward, dtype=torch.float32)
                
            except Exception as e:
                if self.enable_detailed_logging:
                    warnings.warn(f"Base reward calculation failed: {e}. Using fallback.")
        
        # Fallback reward calculation
        success_reward = 0.5 if bool(action_success) else -0.2
        confidence_reward = float(confidence) * 0.3
        
        return torch.tensor(success_reward + confidence_reward, dtype=torch.float32)
    
    def _calculate_tool_bonus(self, tensordict: TensorDict) -> torch.Tensor:
        """Calculate bonus for successful tool usage."""
        tool_bonus = 0.0
        
        # Check for tool execution results
        if "tool_results" in tensordict:
            tool_results = tensordict["tool_results"]
            if isinstance(tool_results, list):
                successful_tools = sum(1 for result in tool_results if result.get("success", False))
                tool_bonus = successful_tools * self.tool_success_weight
        
        # Check for tool success metrics
        if "tool_success_count" in tensordict:
            successful_tools = int(tensordict["tool_success_count"])
            tool_bonus = successful_tools * self.tool_success_weight
        
        # Check metadata for tool usage
        metadata = tensordict.get("metadata", {})
        if isinstance(metadata, dict):
            if metadata.get("used_kg_info", False):
                tool_bonus += self.tool_success_weight * 0.5
            if "tool" in metadata:
                tool_bonus += self.tool_success_weight * 0.3
        
        return torch.tensor(tool_bonus, dtype=torch.float32)
    
    def _calculate_knowledge_bonus(self, tensordict: TensorDict) -> torch.Tensor:
        """Calculate bonus for knowledge discovery and storage."""
        knowledge_bonus = 0.0
        
        # Bonus for discovering new entities
        entities_discovered = tensordict.get("entities_discovered", [])
        if isinstance(entities_discovered, list):
            knowledge_bonus += len(entities_discovered) * self.knowledge_gain_weight
        
        # Bonus for discovering new relations
        relations_discovered = tensordict.get("relations_discovered", [])
        if isinstance(relations_discovered, list):
            knowledge_bonus += len(relations_discovered) * self.knowledge_gain_weight
        
        # Bonus for successful knowledge storage
        metadata = tensordict.get("metadata", {})
        if isinstance(metadata, dict):
            if "stored_successfully" in metadata and metadata["stored_successfully"]:
                knowledge_bonus += self.knowledge_gain_weight * 2
            if "context_id" in metadata:  # Successful storage to internal KG
                knowledge_bonus += self.knowledge_gain_weight
        
        return torch.tensor(knowledge_bonus, dtype=torch.float32)
    
    def _calculate_efficiency_bonus(self, tensordict: TensorDict) -> torch.Tensor:
        """Calculate bonus for efficient action selection."""
        efficiency_bonus = 0.0
        
        # Bonus for quick successful resolution
        step_count = tensordict.get("step_count", torch.tensor(10))
        if isinstance(step_count, torch.Tensor):
            step_count = int(step_count)
        
        # Efficiency bonus decreases with step count
        if step_count <= 3:
            efficiency_bonus = self.efficiency_weight * 2
        elif step_count <= 5:
            efficiency_bonus = self.efficiency_weight
        
        # Penalty for excessive steps without progress
        if step_count > 8:
            efficiency_bonus = -self.efficiency_weight
        
        # Bonus for high confidence quick answers
        confidence = tensordict.get("confidence", torch.tensor(0.5))
        if float(confidence) > 0.8 and step_count <= 2:
            efficiency_bonus += self.efficiency_weight
        
        return torch.tensor(efficiency_bonus, dtype=torch.float32)
    
    def _log_reward_statistics(self):
        """Log reward statistics for analysis."""
        if not self.reward_history:
            return
            
        recent_rewards = self.reward_history[-100:]  # Last 100 rewards
        
        avg_base = sum(r["base_reward"] for r in recent_rewards) / len(recent_rewards)
        avg_tool = sum(r["tool_bonus"] for r in recent_rewards) / len(recent_rewards)
        avg_knowledge = sum(r["knowledge_bonus"] for r in recent_rewards) / len(recent_rewards)
        avg_efficiency = sum(r["efficiency_bonus"] for r in recent_rewards) / len(recent_rewards)
        avg_total = sum(r["total_reward"] for r in recent_rewards) / len(recent_rewards)
        
        print(f"Reward Statistics (last 100 steps):")
        print(f"  Average base reward: {avg_base:.4f}")
        print(f"  Average tool bonus: {avg_tool:.4f}")
        print(f"  Average knowledge bonus: {avg_knowledge:.4f}")
        print(f"  Average efficiency bonus: {avg_efficiency:.4f}")
        print(f"  Average total reward: {avg_total:.4f}")
    
    def transform_reward_spec(self, reward_spec: CompositeSpec) -> CompositeSpec:
        """Transform reward spec to include enhanced reward structure."""
        if not TORCHRL_AVAILABLE:
            return reward_spec
            
        # Update reward spec with enhanced bounds
        reward_spec["reward"] = Bounded(
            low=-2.0, high=2.0, shape=(1,), dtype=torch.float32
        )
        
        return reward_spec
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward statistics."""
        if not self.reward_history:
            return {"message": "No reward history available"}
        
        stats = {
            "total_episodes": len(self.reward_history),
            "average_rewards": {},
            "reward_trends": {},
        }
        
        # Calculate averages
        for component in ["base_reward", "tool_bonus", "knowledge_bonus", "efficiency_bonus", "total_reward"]:
            values = [r[component] for r in self.reward_history]
            stats["average_rewards"][component] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
        
        # Calculate trends (last 20% vs first 20%)
        if len(self.reward_history) >= 10:
            first_20_percent = self.reward_history[:len(self.reward_history)//5]
            last_20_percent = self.reward_history[-len(self.reward_history)//5:]
            
            for component in ["total_reward", "tool_bonus", "knowledge_bonus"]:
                first_avg = sum(r[component] for r in first_20_percent) / len(first_20_percent)
                last_avg = sum(r[component] for r in last_20_percent) / len(last_20_percent)
                stats["reward_trends"][component] = last_avg - first_avg
        
        return stats
    
    def reset_history(self):
        """Reset reward history."""
        self.reward_history.clear()


# Compatibility function for non-TorchRL usage
def create_hybrid_reward_calculator(base_calculator=None, **kwargs):
    """
    Create a hybrid reward calculator that works without TorchRL.
    
    Args:
        base_calculator: Existing reward calculator
        **kwargs: Additional configuration
        
    Returns:
        Enhanced reward calculator
    """
    class HybridRewardCalculator:
        def __init__(self, base_calculator, tool_weight=0.1, knowledge_weight=0.05):
            self.base_calculator = base_calculator
            self.tool_weight = tool_weight
            self.knowledge_weight = knowledge_weight
        
        def calculate_reward(self, context: Dict[str, Any]) -> float:
            """Calculate hybrid reward."""
            # Base reward
            if self.base_calculator:
                base_reward = self.base_calculator.calculate_reward(context)
            else:
                base_reward = 0.5 if context.get("action_success", False) else -0.2
            
            # Tool bonuses
            tool_bonus = 0.0
            if context.get("metadata", {}).get("used_kg_info", False):
                tool_bonus += self.tool_weight
            
            # Knowledge bonus
            knowledge_bonus = 0.0
            entities_found = context.get("entities_found", 0)
            knowledge_bonus += entities_found * self.knowledge_weight
            
            return base_reward + tool_bonus + knowledge_bonus
    
    return HybridRewardCalculator(base_calculator, **kwargs)