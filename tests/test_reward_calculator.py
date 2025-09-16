"""Tests for reward calculator functionality."""

import pytest
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from rl_kg_agent.utils.reward_calculator import RewardCalculator, RewardComponents


@dataclass
class MockActionResult:
    """Mock action result for testing."""
    success: bool
    response: str
    metadata: dict
    entities_discovered: list = None
    relations_discovered: list = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.entities_discovered is None:
            self.entities_discovered = []
        if self.relations_discovered is None:
            self.relations_discovered = []


class TestRewardCalculator:
    """Test cases for RewardCalculator."""

    @pytest.fixture
    def reward_calculator(self):
        """Create a RewardCalculator instance for testing."""
        return RewardCalculator()

    @pytest.fixture
    def sample_context(self):
        """Sample context for testing."""
        return {
            "query": "What is the capital of France?",
            "expected_answer": "Paris",
            "entities": ["France", "Paris"],
            "internal_knowledge": "Some relevant knowledge",
            "failed_actions": 0
        }

    @pytest.fixture
    def successful_action_result(self):
        """Sample successful action result."""
        return MockActionResult(
            success=True,
            response="The capital of France is Paris.",
            metadata={"action": "llm_response"},
            entities_discovered=["Paris"],
            relations_discovered=["capital_of"],
            confidence=0.9
        )

    @pytest.fixture
    def failed_action_result(self):
        """Sample failed action result."""
        return MockActionResult(
            success=False,
            response="I encountered an error.",
            metadata={"action": "sparql_query", "error": "syntax error"},
            confidence=0.0
        )

    def test_init(self, reward_calculator):
        """Test initialization."""
        assert reward_calculator.model_name == "all-MiniLM-L6-v2"
        assert reward_calculator.sentence_transformer is not None
        assert len(reward_calculator.weights) == 5

    def test_calculate_reward_successful(self, reward_calculator, sample_context, successful_action_result):
        """Test reward calculation for successful action."""
        reward = reward_calculator.calculate_reward(
            sample_context,
            successful_action_result,
            ground_truth="Paris"
        )

        assert isinstance(reward, RewardComponents)
        assert 0 <= reward.total <= 1
        assert 0 <= reward.semantic_similarity <= 1
        assert reward.action_success > 0  # Should be positive for successful action
        assert reward.knowledge_gain > 0  # Should be positive for discovered entities

    def test_calculate_reward_failed(self, reward_calculator, sample_context, failed_action_result):
        """Test reward calculation for failed action."""
        reward = reward_calculator.calculate_reward(
            sample_context,
            failed_action_result
        )

        assert isinstance(reward, RewardComponents)
        assert reward.action_success == 0  # Should be 0 for failed action
        assert reward.total <= reward.semantic_similarity  # Action success weight should reduce total

    def test_semantic_similarity_with_ground_truth(self, reward_calculator):
        """Test semantic similarity calculation with ground truth."""
        query = "What is the capital of France?"
        response = "The capital of France is Paris."
        expected = "Paris"

        similarity = reward_calculator._calculate_semantic_similarity(query, response, expected)

        assert 0 <= similarity <= 1
        assert isinstance(similarity, float)

    def test_semantic_similarity_without_ground_truth(self, reward_calculator):
        """Test semantic similarity calculation without ground truth."""
        query = "What is the capital of France?"
        response = "The capital of France is Paris."

        similarity = reward_calculator._calculate_semantic_similarity(query, response)

        assert 0 <= similarity <= 1
        assert isinstance(similarity, float)

    def test_semantic_similarity_empty_response(self, reward_calculator):
        """Test semantic similarity with empty response."""
        query = "What is the capital of France?"
        response = ""
        expected = "Paris"

        similarity = reward_calculator._calculate_semantic_similarity(query, response, expected)

        assert similarity == 0.0

    def test_action_success_reward(self, reward_calculator, successful_action_result, failed_action_result):
        """Test action success reward calculation."""
        success_reward = reward_calculator._calculate_action_success_reward(successful_action_result)
        failure_reward = reward_calculator._calculate_action_success_reward(failed_action_result)

        assert success_reward > failure_reward
        assert success_reward > 0
        assert failure_reward == 0

    def test_knowledge_gain_reward(self, reward_calculator, sample_context, successful_action_result):
        """Test knowledge gain reward calculation."""
        # Action with discovered entities and relations
        reward = reward_calculator._calculate_knowledge_gain_reward(sample_context, successful_action_result)

        assert reward > 0
        assert reward <= 1.0

        # Action with no discoveries
        empty_result = MockActionResult(success=True, response="test", metadata={})
        empty_reward = reward_calculator._calculate_knowledge_gain_reward(sample_context, empty_result)

        assert empty_reward == 0

    def test_efficiency_reward(self, reward_calculator):
        """Test efficiency reward calculation."""
        # Context with no failed actions
        good_context = {"failed_actions": 0}
        mock_result = MockActionResult(success=True, response="test", metadata={})

        good_reward = reward_calculator._calculate_efficiency_reward(good_context, mock_result)

        # Context with failed actions
        bad_context = {"failed_actions": 3}
        bad_reward = reward_calculator._calculate_efficiency_reward(bad_context, mock_result)

        assert good_reward > bad_reward
        assert 0 <= good_reward <= 1
        assert 0 <= bad_reward <= 1

    def test_user_satisfaction_reward(self, reward_calculator, successful_action_result):
        """Test user satisfaction reward calculation."""
        context = {}

        # Test with informative response
        informative_result = MockActionResult(
            success=True,
            response="This is a detailed and informative response about the topic.",
            metadata={}
        )
        informative_reward = reward_calculator._calculate_user_satisfaction_reward(context, informative_result)

        # Test with error response
        error_result = MockActionResult(
            success=False,
            response="Sorry, I encountered an error.",
            metadata={}
        )
        error_reward = reward_calculator._calculate_user_satisfaction_reward(context, error_result)

        assert informative_reward > error_reward
        assert 0 <= informative_reward <= 1
        assert 0 <= error_reward <= 1

    def test_calculate_batch_rewards(self, reward_calculator, sample_context, successful_action_result):
        """Test batch reward calculation."""
        contexts = [sample_context, sample_context]
        results = [successful_action_result, successful_action_result]
        ground_truths = ["Paris", "Paris"]

        batch_rewards = reward_calculator.calculate_batch_rewards(contexts, results, ground_truths)

        assert len(batch_rewards) == 2
        assert all(isinstance(reward, RewardComponents) for reward in batch_rewards)

    def test_update_weights(self, reward_calculator):
        """Test weight updates."""
        original_semantic = reward_calculator.weights["semantic_similarity"]

        reward_calculator.update_weights(semantic_weight=0.6)

        assert reward_calculator.weights["semantic_similarity"] == 0.6
        assert reward_calculator.weights["semantic_similarity"] != original_semantic

    def test_compute_semantic_embeddings(self, reward_calculator):
        """Test semantic embeddings computation."""
        texts = ["What is the capital of France?", "Paris is the capital of France."]

        embeddings = reward_calculator.compute_semantic_embeddings(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0  # Should have some embedding dimension

    def test_find_most_similar(self, reward_calculator):
        """Test finding most similar text."""
        query = "What is the capital of France?"
        candidates = [
            "Paris is the capital of France.",
            "London is the capital of England.",
            "The weather is nice today."
        ]

        best_idx, best_score = reward_calculator.find_most_similar(query, candidates)

        assert 0 <= best_idx < len(candidates)
        assert 0 <= best_score <= 1
        assert isinstance(best_idx, int)
        assert isinstance(best_score, float)

        # The first candidate should be most similar
        assert best_idx == 0

    def test_find_most_similar_empty_candidates(self, reward_calculator):
        """Test finding most similar with empty candidates."""
        query = "What is the capital of France?"
        candidates = []

        best_idx, best_score = reward_calculator.find_most_similar(query, candidates)

        assert best_idx == -1
        assert best_score == 0.0


if __name__ == "__main__":
    pytest.main([__file__])