"""Configuration support for TorchRL integration."""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TorchRLConfig:
    """Configuration for TorchRL features."""
    
    # Core TorchRL settings
    enabled: bool = False
    tokenizer_name: str = "microsoft/DialoGPT-medium"
    device: str = "auto"  # "auto", "cpu", "cuda"
    batch_size: tuple = (1,)
    
    # Tool enhancement settings
    enable_tool_enhancement: bool = True
    tool_timeout: float = 30.0
    enable_browser_tools: bool = False  # Requires additional setup
    allowed_domains: list = None
    
    # Reward system settings
    tool_success_weight: float = 0.1
    knowledge_gain_weight: float = 0.05
    efficiency_weight: float = 0.02
    enable_detailed_logging: bool = False
    
    # Action executor settings
    enable_enhanced_sparql: bool = True
    enable_enhanced_planning: bool = True
    enable_context_awareness: bool = True
    
    # Performance settings
    max_history_size: int = 1000
    enable_caching: bool = True
    log_execution_metrics: bool = True
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.allowed_domains is None:
            self.allowed_domains = ["google.com", "github.com", "pubmed.ncbi.nlm.nih.gov"]


@dataclass
class MCPConfig:
    """Configuration for MCP (Model Context Protocol) integration."""
    
    # Core MCP settings
    enabled: bool = False
    config_file: str = "configs/mcp_config.json"
    default_server: str = "unified_biomedical"
    
    # Connection settings
    connection_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_concurrent_requests: int = 5
    
    # Tool preferences
    prefer_rag_for_questions: bool = True
    prefer_search_for_entities: bool = True
    enable_entity_lookup: bool = True
    default_candidates: int = 20
    
    # Integration settings
    enable_biomedical_enhancement: bool = True
    auto_detect_biomedical_queries: bool = True
    fallback_on_error: bool = True
    cache_responses: bool = True


@dataclass 
class Config:
    """Main configuration class for RL-KG-Agent with TorchRL support."""
    
    # Existing configuration (backward compatibility)
    reward_weights: Dict[str, float] = None
    ppo_config: Dict[str, Any] = None
    environment_config: Dict[str, Any] = None
    
    # TorchRL configuration
    torchrl: TorchRLConfig = None
    
    # MCP configuration
    mcp: MCPConfig = None
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.reward_weights is None:
            self.reward_weights = {
                "semantic_similarity": 0.4,
                "action_success": 0.25,
                "knowledge_gain": 0.15,
                "efficiency": 0.1,
                "user_satisfaction": 0.1
            }
        
        if self.ppo_config is None:
            self.ppo_config = {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95
            }
        
        if self.environment_config is None:
            self.environment_config = {
                "max_steps": 10,
                "timeout": 120
            }
            
        if self.torchrl is None:
            self.torchrl = TorchRLConfig()
            
        if self.mcp is None:
            self.mcp = MCPConfig()


class ConfigManager:
    """Manager for loading and saving configurations."""
    
    DEFAULT_CONFIG_FILE = "rl_kg_agent_config.json"
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Config:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file, defaults to current directory
            
        Returns:
            Configuration object
        """
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_FILE
        
        if not os.path.exists(config_path):
            # Create default config if none exists
            config = Config()
            cls.save_config(config, config_path)
            return config
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Handle nested TorchRL config
            torchrl_data = data.pop('torchrl', {})
            torchrl_config = TorchRLConfig(**torchrl_data)
            
            # Handle nested MCP config
            mcp_data = data.pop('mcp', {})
            mcp_config = MCPConfig(**mcp_data)
            
            config = Config(torchrl=torchrl_config, mcp=mcp_config, **data)
            return config
            
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration.")
            return Config()
    
    @classmethod
    def save_config(cls, config: Config, config_path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            config: Configuration object to save
            config_path: Path to save configuration file
        """
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_FILE
        
        try:
            # Convert to dictionary
            config_dict = asdict(config)
            
            # Ensure directory exists
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            print(f"Configuration saved to {config_path}")
            
        except Exception as e:
            print(f"Warning: Could not save config to {config_path}: {e}")
    
    @classmethod
    def create_example_config(cls, output_path: str = "example_config.json"):
        """Create an example configuration file with all options."""
        config = Config()
        
        # Set some example TorchRL settings
        config.torchrl.enabled = True
        config.torchrl.enable_tool_enhancement = True
        config.torchrl.enable_browser_tools = False  # Requires setup
        config.torchrl.enable_detailed_logging = True
        
        cls.save_config(config, output_path)
        print(f"Example configuration created at {output_path}")
    
    @classmethod
    def get_device(cls, device_str: str = "auto"):
        """
        Get appropriate device for TorchRL operations.
        
        Args:
            device_str: Device specification ("auto", "cpu", "cuda")
            
        Returns:
            Device string or torch.device if torch is available
        """
        if device_str == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return torch.device("cuda")
                else:
                    return torch.device("cpu")
            except ImportError:
                return "cpu"
        
        elif device_str == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return torch.device("cuda")
                else:
                    print("Warning: CUDA requested but not available. Using CPU.")
                    return torch.device("cpu")
            except ImportError:
                print("Warning: PyTorch not available. Using CPU string.")
                return "cpu"
        
        else:
            return device_str


def get_config(config_path: Optional[str] = None) -> Config:
    """Convenience function to get configuration."""
    return ConfigManager.load_config(config_path)


def is_torchrl_enabled(config: Optional[Config] = None) -> bool:
    """Check if TorchRL features are enabled."""
    if config is None:
        config = get_config()
    
    return config.torchrl.enabled


def check_torchrl_dependencies() -> bool:
    """Check if TorchRL dependencies are available."""
    try:
        import torch
        import torchrl
        from tensordict import TensorDict
        return True
    except ImportError:
        return False


def validate_torchrl_config(config: Config) -> tuple[bool, list[str]]:
    """
    Validate TorchRL configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if config.torchrl.enabled:
        # Check dependencies
        if not check_torchrl_dependencies():
            errors.append("TorchRL enabled but dependencies not available. Install with: pip install 'torchrl[llm]'")
        
        # Check tokenizer
        if config.torchrl.tokenizer_name:
            try:
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(config.torchrl.tokenizer_name)
            except Exception as e:
                errors.append(f"Could not load tokenizer '{config.torchrl.tokenizer_name}': {e}")
        
        # Check browser tools
        if config.torchrl.enable_browser_tools:
            try:
                import playwright
            except ImportError:
                errors.append("Browser tools enabled but playwright not available. Install with: pip install playwright")
        
        # Validate weights
        if not 0 <= config.torchrl.tool_success_weight <= 1:
            errors.append("tool_success_weight must be between 0 and 1")
        
        if not 0 <= config.torchrl.knowledge_gain_weight <= 1:
            errors.append("knowledge_gain_weight must be between 0 and 1")
    
    return len(errors) == 0, errors


def is_mcp_enabled(config: Optional[Config] = None) -> bool:
    """Check if MCP features are enabled."""
    if config is None:
        config = get_config()
    
    return config.mcp.enabled


def check_mcp_dependencies() -> bool:
    """Check if MCP dependencies are available."""
    try:
        import asyncio
        import json
        import subprocess
        return True
    except ImportError:
        return False


def validate_mcp_config(config: Config) -> tuple[bool, list[str]]:
    """
    Validate MCP configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if config.mcp.enabled:
        # Check dependencies
        if not check_mcp_dependencies():
            errors.append("MCP enabled but basic dependencies not available")
        
        # Check config file
        mcp_config_path = config.mcp.config_file
        if not Path(mcp_config_path).exists():
            # Try relative to current directory
            if not Path(mcp_config_path).is_absolute():
                for possible_path in [mcp_config_path, f"../{mcp_config_path}"]:
                    if Path(possible_path).exists():
                        break
                else:
                    errors.append(f"MCP config file not found: {mcp_config_path}")
        
        # Validate timeout settings
        if config.mcp.connection_timeout <= 0:
            errors.append("connection_timeout must be positive")
        
        if config.mcp.retry_attempts < 0:
            errors.append("retry_attempts must be non-negative")
        
        if config.mcp.default_candidates <= 0:
            errors.append("default_candidates must be positive")
    
    return len(errors) == 0, errors


def get_mcp_manager(config: Optional[Config] = None):
    """Get MCP manager instance if MCP is enabled."""
    if config is None:
        config = get_config()
    
    if not config.mcp.enabled:
        return None
    
    try:
        from .utils.mcp_client import MCPManager
        return MCPManager(config.mcp.config_file)
    except ImportError as e:
        print(f"Warning: Could not import MCP client: {e}")
        return None
    except Exception as e:
        print(f"Warning: Could not create MCP manager: {e}")
        return None


if __name__ == "__main__":
    # Create example configuration
    ConfigManager.create_example_config()
    
    # Test configuration loading
    config = get_config()
    print(f"TorchRL enabled: {config.torchrl.enabled}")
    print(f"MCP enabled: {config.mcp.enabled}")
    print(f"TorchRL dependencies available: {check_torchrl_dependencies()}")
    print(f"MCP dependencies available: {check_mcp_dependencies()}")
    
    # Validate TorchRL config
    is_valid, errors = validate_torchrl_config(config)
    if not is_valid:
        print("TorchRL configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("TorchRL configuration is valid!")
    
    # Validate MCP config
    is_valid, errors = validate_mcp_config(config)
    if not is_valid:
        print("MCP configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("MCP configuration is valid!")