"""TorchRL transforms for RL-KG-Agent."""

try:
    from .kg_transform import KnowledgeGraphTransform
    from .hybrid_reward_transform import HybridRewardTransform
    TORCHRL_AVAILABLE = True
except ImportError:
    TORCHRL_AVAILABLE = False
    KnowledgeGraphTransform = None
    HybridRewardTransform = None

__all__ = [
    'KnowledgeGraphTransform', 
    'HybridRewardTransform', 
    'TORCHRL_AVAILABLE'
]