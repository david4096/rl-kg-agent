# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Reinforcement Learning agent for Knowledge Graph reasoning that combines static RDF knowledge graphs with learned internal knowledge representations and LLM capabilities. The agent uses PPO (Proximal Policy Optimization) to learn optimal question-answering strategies across a 5-action space.

## Development Commands

### Installation and Setup (UV Recommended)
- `uv sync` - Install all dependencies
- `uv sync --dev` - Install with development dependencies
- `uv pip install -e .` - Install package in development mode
- `uv shell` - Activate virtual environment

### CLI Commands
- `uv run rl-kg-agent interactive --ttl-file <path>` - Start interactive mode
- `uv run rl-kg-agent train --ttl-file <path> --dataset squad --episodes 1000` - Train agent
- `uv run rl-kg-agent evaluate --ttl-file <path> --model-path <path>` - Evaluate trained model

### Code Quality
- `uv run ruff format src/` - Format code (fast)
- `uv run ruff check src/` - Lint code (replaces flake8)
- `uv run mypy src/` - Type checking
- `uv run pytest tests/` - Run tests
- `uv run pytest tests/ --cov=src/rl_kg_agent` - Run tests with coverage

### Makefile Commands (Recommended for Development)
- `make dev-setup` - Complete development setup
- `make format` - Format code with ruff
- `make lint` - Run linting
- `make lint-fix` - Run linting with fixes
- `make type-check` - Run mypy type checking
- `make test` - Run tests
- `make test-cov` - Run tests with coverage
- `make quality` - Run all quality checks
- `make ci` - Full CI pipeline
- `make clean` - Clean build artifacts
- `make help` - Show all available commands

## Architecture Overview

### Core Components
1. **Knowledge Graph Loader** (`src/rl_kg_agent/knowledge/kg_loader.py`) - Loads RDF/TTL files, executes SPARQL queries
2. **Internal Knowledge Graph** (`src/rl_kg_agent/knowledge/internal_kg.py`) - Dynamic knowledge storage with memory and importance scoring
3. **Action Space** (`src/rl_kg_agent/actions/action_space.py`) - 5 discrete actions: LLM response, SPARQL query, knowledge storage, clarifying questions, planning
4. **PPO Agent** (`src/rl_kg_agent/agents/ppo_agent.py`) - RL agent with custom Gymnasium environment
5. **Reward Calculator** (`src/rl_kg_agent/utils/reward_calculator.py`) - Sentence transformer-based semantic rewards
6. **Dataset Loader** (`src/rl_kg_agent/data/dataset_loader.py`) - HuggingFace integration for QA datasets

### Action Space
The agent can take 5 actions:
1. **RESPOND_WITH_LLM** - Generate response using language model
2. **QUERY_STATIC_KG** - Write and execute SPARQL queries against loaded RDF
3. **STORE_TO_INTERNAL_KG** - Store learned information in internal knowledge graph
4. **ASK_REFINING_QUESTION** - Ask clarifying questions to improve context
5. **LLM_PLANNING_STAGE** - Engage in step-by-step reasoning and planning

### Key Dependencies
- `torch` and `transformers` for ML components
- `sentence-transformers` for semantic similarity rewards
- `stable-baselines3` for PPO implementation
- `rdflib` for RDF/SPARQL handling
- `datasets` for HuggingFace dataset integration
- `gymnasium` for RL environment

## Development Notes

- The LLM client (`src/rl_kg_agent/utils/llm_client.py`) is currently a placeholder with rule-based responses
- Internal KG uses pickle for persistence (`internal_kg.pkl`)
- Reward system combines semantic similarity, action success, knowledge gain, efficiency, and user satisfaction
- Training uses QA datasets (SQuAD, Natural Questions, MS MARCO) for learning
- CLI provides interactive mode, training, and evaluation workflows
- uv run rl-kg-agent train --ttl-file examples/simple_knowledge_graph.ttl --dataset squad --episodes 5 --output-model kg-model-test --llm-model google/gemma-3-4b-it --visualize