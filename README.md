# RL-KG-Agent

A Reinforcement Learning agent for Knowledge Graph reasoning that combines static RDF knowledge graphs with learned internal knowledge representations and LLM capabilities.

## Features

- **Multi-modal Knowledge Access**: Integrates static RDF/TTL knowledge graphs with dynamic internal knowledge graph memory
- **Reinforcement Learning**: Uses PPO (Proximal Policy Optimization) for learning optimal question-answering strategies
- **Rich Action Space**: 5 core actions including SPARQL queries, LLM responses, knowledge storage, clarifying questions, and planning
- **Semantic Reward System**: Uses sentence transformers for semantic similarity-based rewards
- **Long-term Memory**: Builds and maintains an internal knowledge graph of learned information
- **HuggingFace Integration**: Supports training on popular QA datasets (SQuAD, Natural Questions, MS MARCO)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rl-kg-agent

# Install dependencies
pip install -e .

# For development
pip install -e .[dev]
```

## Quick Start

### 1. Interactive Mode

Start an interactive session with your knowledge graph:

```bash
rl-kg-agent interactive --ttl-file path/to/your/knowledge_graph.ttl
```

### 2. Training

Train the RL agent on a QA dataset:

```bash
rl-kg-agent train --ttl-file path/to/your/knowledge_graph.ttl --dataset squad --episodes 1000 --output-model trained_model
```

### 3. Evaluation

Evaluate a trained model:

```bash
rl-kg-agent evaluate --ttl-file path/to/your/knowledge_graph.ttl --model-path trained_model --test-file test_questions.json
```

## Architecture

### Core Components

1. **Knowledge Graph Loader** (`kg_loader.py`)
   - Loads RDF/TTL files using rdflib
   - Executes SPARQL queries
   - Provides schema introspection

2. **Internal Knowledge Graph** (`internal_kg.py`)
   - Dynamic knowledge storage with nodes and edges
   - Query context memory
   - Importance scoring and pruning

3. **Action Space** (`action_space.py`)
   - 5 discrete actions for the RL agent
   - Each action has applicability conditions and execution logic

4. **PPO Agent** (`ppo_agent.py`)
   - Custom Gymnasium environment
   - PPO implementation using stable-baselines3
   - Multi-input observation space

5. **Reward Calculator** (`reward_calculator.py`)
   - Sentence transformer-based semantic similarity
   - Multi-component reward system
   - Configurable reward weights

### Action Space

1. **RESPOND_WITH_LLM**: Generate response using language model
2. **QUERY_STATIC_KG**: Write and execute SPARQL queries
3. **STORE_TO_INTERNAL_KG**: Store learned information in memory
4. **ASK_REFINING_QUESTION**: Ask clarifying questions
5. **LLM_PLANNING_STAGE**: Engage in step-by-step reasoning

## Configuration

Create a configuration file to customize behavior:

```json
{
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
  }
}
```

## Dataset Support

The system supports several QA datasets:

- **SQuAD 1.1/2.0**: Stanford Question Answering Dataset
- **Natural Questions**: Google's natural language questions
- **MS MARCO**: Microsoft's machine reading comprehension
- **Custom datasets**: JSON/CSV format support

## Development

### Project Structure

```
src/rl_kg_agent/
├── agents/          # PPO agent implementation
├── actions/         # Action space and management
├── knowledge/       # KG loading and internal storage
├── utils/           # Reward calculation and utilities
├── data/           # Dataset loading and processing
└── cli.py          # Command line interface
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Formatting
black src/

# Linting
flake8 src/

# Type checking
mypy src/
```

## Examples

### Custom Knowledge Graph

Create a simple TTL file:

```turtle
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Paris rdfs:label "Paris" .
ex:Paris ex:isCapitalOf ex:France .
ex:France rdfs:label "France" .
```

### Test Questions

Create a JSON file with test questions:

```json
[
  {
    "question": "What is the capital of France?",
    "expected_answer": "Paris"
  },
  {
    "question": "Which city is the capital of France?",
    "expected_answer": "Paris"
  }
]
```

## License

Apache 2.0 - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Roadmap

- [ ] Advanced entity recognition integration
- [ ] Multi-hop reasoning capabilities
- [ ] Distributed training support
- [ ] Web interface
- [ ] Integration with more LLM APIs
- [ ] Advanced SPARQL query generation