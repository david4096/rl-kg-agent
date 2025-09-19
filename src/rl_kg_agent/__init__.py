"""RL-KG-Agent: Reinforcement Learning Agent for Knowledge Graph Reasoning."""

__version__ = "0.1.0"

# Load environment configuration on package import
from .utils.env_config import init_environment
init_environment()