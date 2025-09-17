"""TorchRL-compatible environments for RL-KG-Agent."""

try:
    from .torchrl_kg_env import TorchRLKnowledgeGraphEnv
    TORCHRL_AVAILABLE = True
except ImportError:
    TORCHRL_AVAILABLE = False
    TorchRLKnowledgeGraphEnv = None

__all__ = ['TorchRLKnowledgeGraphEnv', 'TORCHRL_AVAILABLE']