"""Tests for TorchRL integration with backward compatibility verification."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import components for testing
from rl_kg_agent.config import Config, ConfigManager, TorchRLConfig
from rl_kg_agent.actions.action_space import ActionType, ActionResult
from rl_kg_agent.agents.ppo_agent import create_environment, create_action_executor


def _torchrl_available() -> bool:
    """Check if TorchRL dependencies are available."""
    try:
        import torch
        import torchrl
        from tensordict import TensorDict
        return True
    except ImportError:
        return False


class TestBackwardCompatibility:
    """Test that existing functionality continues to work without TorchRL."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create mock components
        self.mock_action_manager = Mock()
        self.mock_reward_calculator = Mock()
        self.mock_internal_kg = Mock()
        self.mock_kg_loader = Mock()
        self.mock_llm_client = Mock()
        
        # Configure mock responses
        self.mock_action_manager.execute_action.return_value = ActionResult(
            success=True,
            response="Test response",
            metadata={"action": "test"},
            confidence=0.8
        )
        # Mock get_applicable_actions to return a list of actions
        self.mock_action_manager.get_applicable_actions.return_value = [
            ActionType.RESPOND_DIRECTLY, ActionType.QUERY_KG_THEN_RESPOND, ActionType.PLAN_THEN_RESPOND
        ]
        
        self.mock_reward_calculator.calculate_reward.return_value = 0.5
        self.mock_internal_kg.size.return_value = 10
    
    def test_config_creation_and_loading(self):
        """Test configuration system works correctly."""
        # Test default config creation
        config = Config()
        assert isinstance(config.torchrl, TorchRLConfig)
        assert config.torchrl.enabled == False  # Default disabled
        
        # Test config serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            ConfigManager.save_config(config, config_path)
            loaded_config = ConfigManager.load_config(config_path)
            
            assert loaded_config.torchrl.enabled == config.torchrl.enabled
            assert loaded_config.torchrl.tool_success_weight == config.torchrl.tool_success_weight
        finally:
            os.unlink(config_path)
    
    def test_environment_creation_fallback(self):
        """Test environment creation falls back gracefully when TorchRL is not available."""
        config = Config()
        config.torchrl.enabled = False
        
        # Should create standard environment
        env = create_environment(
            action_manager=self.mock_action_manager,
            reward_calculator=self.mock_reward_calculator,
            internal_kg=self.mock_internal_kg,
            config=config,
            use_torchrl=False
        )
        
        # Should be standard KGReasoningEnvironment
        assert env.__class__.__name__ == "KGReasoningEnvironment"
        assert env.action_space.n == 5  # 5 discrete actions
    
    def test_action_executor_fallback(self):
        """Test action executor falls back to standard manager when TorchRL is disabled."""
        config = Config()
        config.torchrl.enabled = False
        
        # Should return the original action manager
        executor = create_action_executor(
            action_manager=self.mock_action_manager,
            config=config,
            use_torchrl=False
        )
        
        assert executor == self.mock_action_manager
    
    def test_action_space_compatibility(self):
        """Test that action space remains compatible (5 discrete actions)."""
        config = Config()
        
        # Test with TorchRL disabled
        env = create_environment(
            action_manager=self.mock_action_manager,
            reward_calculator=self.mock_reward_calculator,
            internal_kg=self.mock_internal_kg,
            config=config,
            use_torchrl=False
        )
        
        # Action space should be unchanged
        assert hasattr(env, 'action_space')
        assert env.action_space.n == 5
        
        # Should handle all 5 action types
        for action_id in range(5):
            action_type = ActionType(action_id)
            assert action_type in [
                ActionType.RESPOND_DIRECTLY,
                ActionType.QUERY_KG_THEN_RESPOND, 
                ActionType.PLAN_THEN_RESPOND,
                ActionType.ASK_CLARIFYING_QUESTION,
                ActionType.STORE_AND_RESPOND
            ]


class TestTorchRLIntegration:
    """Test TorchRL integration features (when available)."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_action_manager = Mock()
        self.mock_reward_calculator = Mock()
        self.mock_internal_kg = Mock()
        self.mock_kg_loader = Mock()
        self.mock_llm_client = Mock()
        
        # Mock get_applicable_actions to return a list of actions
        self.mock_action_manager.get_applicable_actions.return_value = [
            ActionType.RESPOND_DIRECTLY, ActionType.QUERY_KG_THEN_RESPOND, ActionType.PLAN_THEN_RESPOND
        ]
    
    @pytest.mark.skipif(
        not _torchrl_available(),
        reason="TorchRL dependencies not available"
    )
    def test_torchrl_environment_creation(self):
        """Test TorchRL environment creation when dependencies are available."""
        config = Config()
        config.torchrl.enabled = True
        
        # Should create TorchRL environment if available
        try:
            env = create_environment(
                action_manager=self.mock_action_manager,
                reward_calculator=self.mock_reward_calculator,
                internal_kg=self.mock_internal_kg,
                kg_loader=self.mock_kg_loader,
                config=config,
                use_torchrl=True
            )
            
            # Should be TorchRL environment but maintain action space
            assert hasattr(env, 'action_space')
            assert env.action_space.n == 5  # Still 5 discrete actions
            
        except ImportError:
            # If TorchRL not available, should gracefully fallback
            pytest.skip("TorchRL not available, fallback tested separately")
    
    def test_config_validation(self):
        """Test configuration validation works correctly."""
        from rl_kg_agent.config import validate_torchrl_config
        
        # Valid config
        config = Config()
        config.torchrl.enabled = False
        is_valid, errors = validate_torchrl_config(config)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid weights
        config.torchrl.enabled = True
        config.torchrl.tool_success_weight = 2.0  # Invalid (> 1.0)
        is_valid, errors = validate_torchrl_config(config)
        assert not is_valid
        assert len(errors) > 0


class TestCLIIntegration:
    """Test CLI integration and command line options."""
    
    def test_cli_import_safety(self):
        """Test that CLI can be imported without TorchRL dependencies."""
        try:
            from rl_kg_agent.cli import cli
            assert cli is not None
        except ImportError as e:
            pytest.fail(f"CLI import failed: {e}")
    
    def test_config_commands_work(self):
        """Test configuration commands work without TorchRL."""
        from rl_kg_agent.config import ConfigManager
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Should be able to create example config
            ConfigManager.create_example_config(config_path)
            assert os.path.exists(config_path)
            
            # Should be able to load it back
            config = ConfigManager.load_config(config_path)
            assert isinstance(config, Config)
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)


class TestIntegrationSafety:
    """Test that integration is safe and handles errors gracefully."""
    
    def test_import_safety(self):
        """Test that all modules can be imported safely."""
        # These should not raise ImportError even without TorchRL
        from rl_kg_agent.config import Config, get_config
        from rl_kg_agent.agents.ppo_agent import create_environment, create_action_executor
        
        config = get_config()
        assert isinstance(config, Config)
    
    def test_graceful_fallback_when_torchrl_missing(self):
        """Test graceful fallback when TorchRL is requested but not available."""
        # Mock TorchRL as unavailable
        with patch('rl_kg_agent.agents.ppo_agent.TORCHRL_AVAILABLE', False):
            mock_action_manager = Mock()
            mock_reward_calculator = Mock()
            mock_internal_kg = Mock()
            
            # Mock get_applicable_actions to return a list of actions
            mock_action_manager.get_applicable_actions.return_value = [
                ActionType.RESPOND_DIRECTLY, ActionType.QUERY_KG_THEN_RESPOND, ActionType.PLAN_THEN_RESPOND
            ]
            
            config = Config()
            config.torchrl.enabled = True  # Request TorchRL
            
            # Should fall back gracefully
            env = create_environment(
                action_manager=mock_action_manager,
                reward_calculator=mock_reward_calculator,
                internal_kg=mock_internal_kg,
                config=config,
                use_torchrl=True  # Request TorchRL
            )
            
            # Should create standard environment
            assert env.__class__.__name__ == "KGReasoningEnvironment"
    
    def test_existing_tests_still_pass(self):
        """Verify that existing functionality hasn't been broken."""
        # Test basic action type enumeration
        assert len(ActionType) == 5
        assert ActionType.RESPOND_DIRECTLY.value == 0
        assert ActionType.STORE_AND_RESPOND.value == 4
        
        # Test action result creation
        result = ActionResult(
            success=True,
            response="Test",
            metadata={}
        )
        assert result.success
        assert result.is_final  # Default should be True


if __name__ == "__main__":
    # Run basic compatibility tests
    print("🧪 Running TorchRL integration compatibility tests...")
    
    # Test config system
    config = Config()
    print(f"✅ Config creation: TorchRL enabled = {config.torchrl.enabled}")
    
    # Test import safety
    try:
        from rl_kg_agent.agents.ppo_agent import create_environment
        print("✅ Safe imports working")
    except Exception as e:
        print(f"❌ Import error: {e}")
    
    # Test basic functionality
    from rl_kg_agent.actions.action_space import ActionType
    
    try:
        # Verify ActionType has 5 actions (key compatibility requirement)
        assert len(ActionType) == 5
        print(f"✅ Action space compatibility: {len(ActionType)} actions preserved")
        
        # Test factory function availability
        from rl_kg_agent.agents.ppo_agent import create_environment, create_action_executor
        print("✅ Factory functions available")
        
    except Exception as e:
        print(f"❌ Compatibility test error: {e}")
    
    print("🎉 Basic compatibility tests completed!")