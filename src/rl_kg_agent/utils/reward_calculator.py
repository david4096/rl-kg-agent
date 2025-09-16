"""Reward calculation using sentence transformers for semantic similarity."""

from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Components that make up the total reward."""
    semantic_similarity: float
    action_success: float
    knowledge_gain: float
    efficiency: float
    user_satisfaction: float
    action_diversity: float
    total: float


class RewardCalculator:
    """Calculates rewards for RL training using sentence transformers and multiple criteria."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize reward calculator.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.sentence_transformer = None
        self._load_model()

        # Reward weights for different components
        self.weights = {
            "semantic_similarity": 0.3,
            "action_success": 0.2,
            "knowledge_gain": 0.15,
            "efficiency": 0.1,
            "user_satisfaction": 0.1,
            "action_diversity": 0.15
        }

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.sentence_transformer = SentenceTransformer(self.model_name)
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise

    def calculate_reward(self, context: Dict[str, Any], action_result: Any,
                        ground_truth: Optional[str] = None) -> RewardComponents:
        """Calculate comprehensive reward for an action.

        Args:
            context: Context including query, expected response, etc.
            action_result: Result from executing an action
            ground_truth: Optional ground truth answer for comparison

        Returns:
            RewardComponents with detailed reward breakdown
        """
        # Safe extraction with defaults to prevent division by zero
        query = context.get("query", "")
        response = getattr(action_result, "response", "") or ""
        expected_answer = context.get("expected_answer", ground_truth)

        # Calculate individual reward components
        semantic_sim = self._calculate_semantic_similarity(query, response, expected_answer)
        action_success = self._calculate_action_success_reward(action_result)
        knowledge_gain = self._calculate_knowledge_gain_reward(context, action_result)
        efficiency = self._calculate_efficiency_reward(context, action_result)
        user_satisfaction = self._calculate_user_satisfaction_reward(context, action_result)
        action_diversity = self._calculate_action_diversity_reward(context, action_result)

        # Calculate weighted total with safety checks
        try:
            total = (
                semantic_sim * self.weights["semantic_similarity"] +
                action_success * self.weights["action_success"] +
                knowledge_gain * self.weights["knowledge_gain"] +
                efficiency * self.weights["efficiency"] +
                user_satisfaction * self.weights["user_satisfaction"] +
                action_diversity * self.weights["action_diversity"]
            )
            # Ensure total is finite
            if not np.isfinite(total):
                logger.warning("Non-finite total reward calculated, using default")
                total = 0.0
        except Exception as e:
            logger.error(f"Error calculating total reward: {e}")
            total = 0.0

        return RewardComponents(
            semantic_similarity=semantic_sim,
            action_success=action_success,
            knowledge_gain=knowledge_gain,
            efficiency=efficiency,
            user_satisfaction=user_satisfaction,
            action_diversity=action_diversity,
            total=total
        )

    def _calculate_semantic_similarity(self, query: str, response: str,
                                     expected_answer: Optional[str] = None) -> float:
        """Calculate semantic similarity between response and expected answer.

        Args:
            query: Original query
            response: Agent's response
            expected_answer: Expected/ground truth answer

        Returns:
            Similarity score between 0 and 1
        """
        # Safety checks for empty inputs
        if not response or not response.strip():
            return 0.0

        if not expected_answer or not expected_answer.strip():
            # If no expected answer, use query-response relevance
            if not query or not query.strip():
                return 0.0
            return self._calculate_relevance_score(query, response)

        try:
            # Encode both responses
            embeddings = self.sentence_transformer.encode(
                [response, expected_answer],
                convert_to_tensor=True
            )

            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

            # Normalize to 0-1 range (cosine similarity can be -1 to 1)
            return max(0.0, (similarity + 1) / 2)

        except Exception as e:
            logger.error(f"Failed to calculate semantic similarity: {e}")
            return 0.0

    def _calculate_relevance_score(self, query: str, response: str) -> float:
        """Calculate how relevant the response is to the query.

        Args:
            query: Original query
            response: Agent's response

        Returns:
            Relevance score between 0 and 1
        """
        try:
            embeddings = self.sentence_transformer.encode(
                [query, response],
                convert_to_tensor=True
            )

            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            return max(0.0, (similarity + 1) / 2)

        except Exception as e:
            logger.error(f"Failed to calculate relevance score: {e}")
            return 0.0

    def _calculate_action_success_reward(self, action_result: Any) -> float:
        """Calculate reward based on action execution success.

        Args:
            action_result: Result from action execution

        Returns:
            Success reward between 0 and 1
        """
        if not hasattr(action_result, "success"):
            return 0.0

        base_reward = 1.0 if action_result.success else 0.0

        # Boost reward based on confidence if available
        if hasattr(action_result, "confidence"):
            confidence = getattr(action_result, "confidence", 1.0)
            base_reward *= confidence

        return base_reward

    def _calculate_knowledge_gain_reward(self, context: Dict[str, Any], action_result: Any) -> float:
        """Calculate reward based on knowledge gained from the action.

        Args:
            context: Current context
            action_result: Result from action execution

        Returns:
            Knowledge gain reward between 0 and 1
        """
        reward = 0.0

        # Reward for discovering new entities
        if hasattr(action_result, "entities_discovered"):
            entities = getattr(action_result, "entities_discovered", [])
            reward += min(len(entities) * 0.1, 0.5)

        # Reward for discovering new relations
        if hasattr(action_result, "relations_discovered"):
            relations = getattr(action_result, "relations_discovered", [])
            reward += min(len(relations) * 0.1, 0.3)

        # Reward for using internal knowledge effectively
        if context.get("internal_knowledge") and "internal_knowledge" in getattr(action_result, "metadata", {}):
            reward += 0.2

        return min(reward, 1.0)

    def _calculate_efficiency_reward(self, context: Dict[str, Any], action_result: Any) -> float:
        """Calculate reward based on action efficiency.

        Args:
            context: Current context
            action_result: Result from action execution

        Returns:
            Efficiency reward between 0 and 1
        """
        # Penalize if too many failed actions in context
        failed_actions = context.get("failed_actions", 0)
        efficiency_penalty = min(failed_actions * 0.2, 0.8)

        # Reward quick successful actions
        base_efficiency = 1.0 - efficiency_penalty

        # Bonus for first-try success
        if failed_actions == 0 and getattr(action_result, "success", False):
            base_efficiency += 0.2

        return min(max(base_efficiency, 0.0), 1.0)

    def _calculate_user_satisfaction_reward(self, context: Dict[str, Any], action_result: Any) -> float:
        """Calculate reward based on likely user satisfaction.

        Args:
            context: Current context
            action_result: Result from action execution

        Returns:
            User satisfaction reward between 0 and 1
        """
        response = getattr(action_result, "response", "") or ""

        # Simple heuristics for user satisfaction
        satisfaction = 0.5  # Base satisfaction

        if not response or not response.strip():
            return 0.0

        # Penalize very short responses (unless it's a direct answer)
        if len(response.split()) < 5:
            satisfaction -= 0.2

        # Reward informative responses
        if len(response.split()) > 20:
            satisfaction += 0.2

        # Penalize error messages
        error_indicators = ["error", "failed", "sorry", "unable", "can't", "cannot"]
        if any(indicator in response.lower() for indicator in error_indicators):
            satisfaction -= 0.3

        # Reward helpful patterns
        helpful_indicators = ["found", "discovered", "here", "shows", "indicates"]
        if any(indicator in response.lower() for indicator in helpful_indicators):
            satisfaction += 0.2

        return min(max(satisfaction, 0.0), 1.0)

    def _calculate_action_diversity_reward(self, context: Dict[str, Any], action_result: Any) -> float:
        """Calculate reward based on action diversity to discourage monotonic behavior.

        Args:
            context: Current context
            action_result: Result from action execution

        Returns:
            Action diversity reward between 0 and 1
        """
        # Get recent action history from metadata
        current_action = getattr(action_result, "metadata", {}).get("action", "")
        recent_actions = context.get("recent_actions", [])

        # Base diversity reward
        diversity_reward = 0.5

        if not current_action:
            return diversity_reward

        # Penalize if the same action is used repeatedly
        if len(recent_actions) >= 3:
            last_3_actions = recent_actions[-3:]
            if all(action == current_action for action in last_3_actions):
                diversity_reward -= 0.4  # Heavy penalty for 3+ consecutive same actions
                logger.info(f"Diversity penalty applied for repeated {current_action} action")
            elif recent_actions[-1] == current_action:
                diversity_reward -= 0.2  # Moderate penalty for immediate repetition

        # Reward using different action types
        unique_actions_used = len(set(recent_actions[-5:] + [current_action]))
        if unique_actions_used >= 3:
            diversity_reward += 0.3
        elif unique_actions_used >= 2:
            diversity_reward += 0.1

        # Special penalty for PLAN_THEN_RESPOND overuse (addressing the current issue)
        if current_action == "planned_response":
            plan_count = sum(1 for action in recent_actions[-5:] if action == "planned_response")
            if plan_count >= 2:
                diversity_reward -= 0.3
                logger.info("Special penalty for PLAN_THEN_RESPOND overuse")

        return min(max(diversity_reward, 0.0), 1.0)

    def calculate_batch_rewards(self, contexts: List[Dict[str, Any]],
                               action_results: List[Any],
                               ground_truths: Optional[List[str]] = None) -> List[RewardComponents]:
        """Calculate rewards for a batch of actions.

        Args:
            contexts: List of contexts
            action_results: List of action results
            ground_truths: Optional list of ground truth answers

        Returns:
            List of RewardComponents for each action
        """
        if ground_truths is None:
            ground_truths = [None] * len(contexts)

        rewards = []
        for context, result, ground_truth in zip(contexts, action_results, ground_truths):
            reward = self.calculate_reward(context, result, ground_truth)
            rewards.append(reward)

        return rewards

    def get_loss_function(self, predicted_rewards: torch.Tensor,
                         actual_rewards: torch.Tensor) -> torch.Tensor:
        """Calculate loss between predicted and actual rewards.

        Args:
            predicted_rewards: Predicted reward values
            actual_rewards: Calculated actual reward values

        Returns:
            Loss tensor
        """
        # Mean squared error loss
        loss = torch.nn.functional.mse_loss(predicted_rewards, actual_rewards)
        return loss

    def update_weights(self, semantic_weight: float = None, success_weight: float = None,
                      knowledge_weight: float = None, efficiency_weight: float = None,
                      satisfaction_weight: float = None, diversity_weight: float = None) -> None:
        """Update reward component weights.

        Args:
            semantic_weight: Weight for semantic similarity
            success_weight: Weight for action success
            knowledge_weight: Weight for knowledge gain
            efficiency_weight: Weight for efficiency
            satisfaction_weight: Weight for user satisfaction
            diversity_weight: Weight for action diversity
        """
        if semantic_weight is not None:
            self.weights["semantic_similarity"] = semantic_weight
        if success_weight is not None:
            self.weights["action_success"] = success_weight
        if knowledge_weight is not None:
            self.weights["knowledge_gain"] = knowledge_weight
        if efficiency_weight is not None:
            self.weights["efficiency"] = efficiency_weight
        if satisfaction_weight is not None:
            self.weights["user_satisfaction"] = satisfaction_weight
        if diversity_weight is not None:
            self.weights["action_diversity"] = diversity_weight

        # Ensure weights sum to approximately 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.1:
            logger.warning(f"Reward weights sum to {total_weight}, consider normalizing")

    def compute_semantic_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings
        """
        try:
            embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=False)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            return np.zeros((len(texts), 384))  # Default embedding size

    def find_most_similar(self, query_text: str, candidate_texts: List[str]) -> Tuple[int, float]:
        """Find the most similar text to a query from a list of candidates.

        Args:
            query_text: Query text
            candidate_texts: List of candidate texts

        Returns:
            Tuple of (best_index, similarity_score)
        """
        if not candidate_texts:
            return -1, 0.0

        try:
            query_embedding = self.sentence_transformer.encode([query_text], convert_to_tensor=True)
            candidate_embeddings = self.sentence_transformer.encode(candidate_texts, convert_to_tensor=True)

            similarities = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0]
            best_idx = torch.argmax(similarities).item()
            best_score = similarities[best_idx].item()

            # Normalize to 0-1 range
            normalized_score = max(0.0, (best_score + 1) / 2)

            return best_idx, normalized_score

        except Exception as e:
            logger.error(f"Failed to find most similar text: {e}")
            return 0, 0.0