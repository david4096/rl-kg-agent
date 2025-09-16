"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def sample_ttl_content():
    """Sample TTL content for testing."""
    return """
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Paris rdfs:label "Paris" .
ex:Paris ex:isCapitalOf ex:France .
ex:France rdfs:label "France" .
ex:WilliamShakespeare rdfs:label "William Shakespeare" .
ex:RomeoAndJuliet rdfs:label "Romeo and Juliet" .
ex:RomeoAndJuliet ex:writtenBy ex:WilliamShakespeare .
"""


@pytest.fixture(scope="session")
def temp_ttl_file(sample_ttl_content):
    """Create a temporary TTL file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttl', delete=False) as f:
        f.write(sample_ttl_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "reward_weights": {
            "semantic_similarity": 0.4,
            "action_success": 0.25,
            "knowledge_gain": 0.15,
            "efficiency": 0.1,
            "user_satisfaction": 0.1
        },
        "ppo_config": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64
        },
        "sentence_transformer_config": {
            "model_name": "all-MiniLM-L6-v2"
        }
    }


@pytest.fixture
def sample_questions():
    """Sample questions for testing."""
    return [
        {
            "question": "What is the capital of France?",
            "expected_answer": "Paris",
            "question_type": "factual"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "expected_answer": "William Shakespeare",
            "question_type": "factual"
        }
    ]


# Skip tests that require models if they're not available
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring ML model (may be slow)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark tests that use sentence transformers as requiring models
        if "reward_calculator" in item.name or "sentence_transformer" in str(item.function):
            item.add_marker(pytest.mark.requires_model)

        # Mark integration tests
        if "integration" in item.name or "test_cli" in str(item.fspath):
            item.add_marker(pytest.mark.integration)