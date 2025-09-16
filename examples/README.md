# Examples

This directory contains example files and usage demonstrations for the RL-KG-Agent.

## Configuration Files

### `configs/default_config.json`
Default configuration with standard parameters for development and testing.

### `configs/training_config.json`
Training-optimized configuration with parameters tuned for model training.

## Knowledge Graphs

### `simple_knowledge_graph.ttl`
A minimal TTL file for basic testing with just a few triples:
- Paris is the capital of France
- William Shakespeare wrote Romeo and Juliet

### `sample_knowledge_graph.ttl`
A comprehensive example knowledge graph covering:
- Geography (countries, cities, landmarks)
- Literature (authors, books)
- Science (chemistry, physics, biology)
- History and culture

## Test Data

### `test_questions.json`
Basic test questions covering different question types:
- Factual questions (What, Who, When, Where)
- Explanatory questions (How, Why)
- Boolean questions (Is/Are)
- Different difficulty levels

### `evaluation_questions.json`
More comprehensive evaluation questions organized by:
- Subject categories (physics, chemistry, biology, etc.)
- Difficulty levels (easy, medium, hard)
- Question types with expected answers

## Usage Examples

### `basic_usage.py`
Demonstrates the core functionality:
- Loading knowledge graphs
- Setting up components
- Executing queries with the action manager
- Basic reward calculation
- Internal KG storage

Run with:
```bash
cd examples
uv run python basic_usage.py
```

### `training_example.py`
Shows how to set up training:
- Loading training datasets
- Configuring PPO agent
- Running training episodes
- Saving trained models

Run with:
```bash
cd examples
uv run python training_example.py
```

### `evaluation_example.py`
Demonstrates model evaluation:
- Loading test questions
- Running evaluation with trained/untrained models
- Calculating accuracy and reward metrics
- Generating evaluation reports

Run with:
```bash
cd examples
uv run python evaluation_example.py
```

## Quick Start

1. **Basic Testing**:
   ```bash
   # Test basic functionality
   uv run python examples/basic_usage.py

   # Or using the CLI
   uv run rl-kg-agent interactive --ttl-file examples/simple_knowledge_graph.ttl
   ```

2. **Training**:
   ```bash
   # Run training example
   uv run python examples/training_example.py

   # Or using the CLI
   uv run rl-kg-agent train --ttl-file examples/sample_knowledge_graph.ttl --config configs/training_config.json
   ```

3. **Evaluation**:
   ```bash
   # Run evaluation example
   uv run python examples/evaluation_example.py

   # Or using the CLI
   uv run rl-kg-agent evaluate --ttl-file examples/sample_knowledge_graph.ttl --test-file examples/evaluation_questions.json
   ```

## Using with Makefile

You can also use the provided Makefile commands:

```bash
# Interactive mode
make run-interactive TTL_FILE=examples/simple_knowledge_graph.ttl

# Training
make run-train TTL_FILE=examples/sample_knowledge_graph.ttl

# Evaluation
make run-eval TTL_FILE=examples/sample_knowledge_graph.ttl MODEL_PATH=demo_trained_model
```

## Customization

### Creating Your Own Knowledge Graph

Create a TTL file with your domain knowledge:

```turtle
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:YourEntity rdfs:label "Your Entity" .
ex:YourEntity ex:yourProperty ex:YourValue .
```

### Creating Custom Test Questions

Create a JSON file following the format in `test_questions.json`:

```json
[
  {
    "question": "Your question?",
    "expected_answer": "Your expected answer",
    "question_type": "factual",
    "difficulty": "easy",
    "category": "your_domain"
  }
]
```

### Custom Configuration

Modify the configuration files or create new ones to adjust:
- Reward weights
- PPO hyperparameters
- Model settings
- Dataset parameters

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the project root or using `uv run`
2. **Model loading fails**: The examples will fall back to untrained agents
3. **Dataset loading fails**: Examples include fallback sample data
4. **Memory issues**: Reduce `max_examples` in config for large datasets
5. **SPARQL queries return no results**: The current SPARQL query generation is basic and may not find all relevant data. The knowledge graph data is loaded correctly (as shown by manual SPARQL queries), but the automatic query generation could be improved. This is a known enhancement area.

### Debug Mode

Add `--verbose` flag or modify the logging level in configuration files to get more detailed output.

### Performance

- Use smaller knowledge graphs for faster testing
- Limit dataset sizes during development
- Use GPU if available for model training (configure in PPO settings)