# TorchRL LLM Integration Proposal for RL-KG-Agent

## Executive Summary

This proposal outlines how to integrate the TorchRL LLM framework (demonstrated in `llm_browser.ipynb`) into the existing `rl-kg-agent` project. The integration will create a powerful hybrid system that combines:

1. **TorchRL's LLM Environment Composition** - Advanced tool-enabled environments with transforms
2. **Existing Knowledge Graph Reasoning** - Static RDF + dynamic internal KG with SPARQL
3. **Enhanced Action Space** - Tool-based actions alongside existing KG actions
4. **Unified Reward System** - Combining semantic similarity with tool execution success

## Current State Analysis

### RL-KG-Agent (Existing System)
- **Architecture**: PPO agent with 5-action space over knowledge graphs
- **Actions**: LLM response, SPARQL query, knowledge storage, clarifying questions, planning
- **Environment**: Custom Gymnasium environment with multi-input observation space
- **Rewards**: Sentence transformer-based semantic similarity + multi-component scoring
- **Memory**: Internal knowledge graph with importance scoring and pruning

### TorchRL LLM Framework (From llm_browser.ipynb)
- **Architecture**: ChatEnv with composable transforms (BrowserTransform, RewardTransform)
- **Tool System**: JSON-formatted tool calls with structured execution
- **Conversation Management**: History and ChatHistory objects for state management
- **Transforms**: Modular tool integration with transform chaining
- **Rewards**: Custom reward transforms based on tool execution outcomes

## Integration Strategy

### Phase 1: Foundation Layer (Core Integration)

#### 1.1 TorchRL-Compatible Environment Wrapper
```python
# New file: src/rl_kg_agent/envs/torchrl_kg_env.py
class TorchRLKnowledgeGraphEnv(ChatEnv):
    """
    Extends TorchRL's ChatEnv while maintaining PPO compatibility.
    
    Key Design:
    - Action space: Discrete(5) - same as existing system
    - Observation space: Enhanced with tool execution context
    - Reward system: Combines semantic similarity + tool success
    """
    
    def __init__(self, kg_loader, tokenizer, **kwargs):
        super().__init__(
            batch_size=(1,),
            tokenizer=tokenizer,
            system_prompt=self._get_system_prompt()
        )
        
        # Maintain existing 5-action discrete space for PPO
        self.action_space = gym.spaces.Discrete(5)
        
        # Add TorchRL transforms for tool capabilities
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup TorchRL transforms without changing action space."""
        # Add KG transform for SPARQL capabilities
        kg_transform = KnowledgeGraphTransform(self.kg_loader)
        self.append_transform(kg_transform)
        
        # Add browser transform for web search (when needed)
        browser_transform = BrowserTransform(
            allowed_domains=["google.com", "pubmed.ncbi.nlm.nih.gov"]
        )
        self.append_transform(browser_transform)
        
        # Add hybrid reward transform
        reward_transform = HybridRewardTransform()
        self.append_transform(reward_transform)
```

**Key Features:**
- **PPO Compatibility**: Maintains `Discrete(5)` action space
- **Tool Integration**: Transforms handle tool execution transparently
- **Backward Compatibility**: Existing reward calculation preserved
- **Enhanced Execution**: Actions can use tools without changing RL training

#### 1.2 Knowledge Graph Transform
```python
# New file: src/rl_kg_agent/transforms/kg_transform.py
class KnowledgeGraphTransform(Transform):
    """
    Transform that adds SPARQL query and internal KG operations as tools.
    Integrates existing kg_loader.py and internal_kg.py functionality.
    """
```

**Capabilities:**
- SPARQL query execution as a tool
- Internal KG storage/retrieval as tools  
- Schema introspection and query validation
- Seamless integration with existing KG components

#### 1.3 Enhanced Reward Transform
```python
# New file: src/rl_kg_agent/transforms/hybrid_reward_transform.py
class HybridRewardTransform(Transform):
    """
    Combines existing semantic similarity rewards with tool execution rewards.
    Maintains backward compatibility with current reward_calculator.py.
    """
```

**Reward Components:**
- Semantic similarity (existing)
- Tool execution success (new)
- Knowledge gain from tools (enhanced)
- Multi-step reasoning rewards (new)

### Phase 2: Action Space Integration (CORRECTED)

#### 2.1 Maintain Clean Discrete Action Space
**Key Insight**: Keep the action space simple for RL training, but enhance execution:

```python
class TorchRLCompatibleActionSpace:
    # SAME 5 discrete actions (for PPO compatibility)
    RESPOND_WITH_LLM = 0
    QUERY_STATIC_KG = 1  
    STORE_TO_INTERNAL_KG = 2  
    ASK_REFINING_QUESTION = 3
    LLM_PLANNING_STAGE = 4
    
    # Action space remains: Discrete(5) for PPO training
    # But execution is enhanced with TorchRL tools
```

#### 2.2 Two-Stage Action Execution
**Stage 1**: PPO selects discrete action (0-4)  
**Stage 2**: TorchRL transforms enhance the execution

```python
# New file: src/rl_kg_agent/execution/action_executor.py
class TorchRLActionExecutor:
    """
    Executes discrete RL actions using TorchRL's tool framework.
    
    Flow:
    1. PPO agent selects action_id ∈ {0,1,2,3,4}
    2. Action gets executed via TorchRL transforms
    3. Tools are used within the action execution context
    """
    
    def execute_action(self, action_id: int, context: TensorDict) -> TensorDict:
        if action_id == 0:  # RESPOND_WITH_LLM
            return self._execute_llm_with_tools(context)
        elif action_id == 1:  # QUERY_STATIC_KG  
            return self._execute_sparql_with_validation(context)
        # ... etc
```

#### 2.3 Tool-Enhanced Action Implementation
```python
def _execute_llm_with_tools(self, context: TensorDict) -> TensorDict:
    """
    LLM response action enhanced with TorchRL tool access.
    The LLM can decide to use tools during its response generation.
    """
    # LLM generates response that may include tool calls
    llm_response = self.llm_client.generate(context["query"])
    
    # If response contains tool calls, execute via TorchRL transforms
    if "<tool>" in llm_response:
        return self._execute_tools_in_response(llm_response, context)
    else:
        return self._standard_llm_response(llm_response, context)
```

### Phase 3: Conversation and State Management

#### 3.1 Enhanced Conversation History
Upgrade existing conversation handling to use TorchRL's History/ChatHistory system:

```python
# Enhanced: src/rl_kg_agent/agents/conversation_manager.py
class ConversationManager:
    """
    Manages conversation state using TorchRL's History objects.
    Integrates with existing action execution and memory systems.
    """
```

#### 3.2 Multi-Modal State Representation
Extend observation space to include:
- Text conversation history (existing)
- Knowledge graph embeddings (existing)
- Tool execution context (new)
- Multi-step reasoning state (new)

### Phase 4: Training Pipeline Integration

#### 4.1 Hybrid Training Environment
```python
# Enhanced: src/rl_kg_agent/agents/hybrid_ppo_agent.py
class HybridPPOAgent:
    """
    PPO agent that can train on both traditional KG tasks and tool-enabled tasks.
    Uses curriculum learning to gradually introduce tool complexity.
    """
```

#### 4.2 Curriculum Learning Strategy
1. **Stage 1**: Traditional KG reasoning (existing functionality)
2. **Stage 2**: Simple tool usage (browser navigation, API calls)
3. **Stage 3**: Complex multi-tool workflows
4. **Stage 4**: Advanced reasoning with tool chains

## Implementation Architecture

### Directory Structure (New Components)
```
src/rl_kg_agent/
├── envs/
│   ├── torchrl_kg_env.py          # TorchRL-compatible environment
│   └── hybrid_environment.py      # Unified environment wrapper
├── transforms/
│   ├── __init__.py
│   ├── kg_transform.py            # Knowledge graph as tools
│   ├── hybrid_reward_transform.py # Combined reward system
│   └── conversation_transform.py  # Conversation management
├── tools/
│   ├── __init__.py
│   ├── tool_registry.py           # Central tool management
│   ├── kg_tools.py                # SPARQL and internal KG tools
│   ├── browser_tools.py           # Web browsing capabilities
│   └── api_tools.py               # External API integration
├── agents/
│   ├── hybrid_ppo_agent.py        # Enhanced PPO with tools
│   └── conversation_manager.py    # TorchRL conversation handling
└── training/
    ├── curriculum_trainer.py      # Staged training pipeline
    └── hybrid_datasets.py         # Tool-enabled datasets
```

### Backward Compatibility Strategy

1. **Existing Interfaces Preserved**: All current CLI commands continue to work
2. **Gradual Migration**: New features are opt-in via configuration flags
3. **Fallback Mode**: System operates in legacy mode if TorchRL components fail
4. **Progressive Enhancement**: Existing models can be enhanced without retraining

## PPO Training Compatibility Analysis

### Why This Integration Properly Supports RL Training

#### 1. **Maintains Clean Action Space**
```python
# PPO-compatible action space (unchanged)
action_space = gym.spaces.Discrete(5)

# PPO agent training loop (unchanged)
for batch in data_collector:
    # Agent selects action_id ∈ {0, 1, 2, 3, 4}
    action_probs = policy_network(observations)
    actions = action_probs.sample()  # Still discrete sampling
    
    # Execute action via enhanced executor
    next_obs, rewards, dones = env.step(actions)
```

#### 2. **Tool Integration is Transparent to PPO**
- **PPO sees**: Simple discrete action selection
- **Environment handles**: Complex tool execution behind the scenes
- **Reward signal**: Enhanced but still single scalar per step

#### 3. **Observation Space Enhancement**
```python
observation_space = {
    "query_embedding": Box(...),           # Existing
    "kg_context": Box(...),                # Existing  
    "tool_execution_context": Box(...),    # New
    "conversation_history": Box(...)       # Enhanced
}
```

#### 4. **Training Loop Compatibility**
```python
# Standard TorchRL PPO training (no changes needed)
collector = SyncDataCollector(
    env=torchrl_kg_env,           # Our enhanced environment
    policy=policy_module,          # Standard PPO policy
    frames_per_batch=1000,
    total_frames=100_000
)

# PPO loss computation (unchanged)
loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=0.2
)
```

### Comparison with PyTorch RL Tutorial

| Component | PyTorch RL Tutorial | Our Integration |
|-----------|-------------------|-----------------|
| **Action Space** | `Continuous(1)` for pendulum | `Discrete(5)` for KG reasoning |
| **Policy Network** | Outputs (μ, σ) for TanhNormal | Outputs logits for Categorical |
| **Environment** | GymEnv("InvertedPendulum") | TorchRLKnowledgeGraphEnv |
| **Transforms** | ObservationNorm, DoubleToFloat | + KGTransform, BrowserTransform |
| **Reward** | Sparse task completion | Semantic similarity + tool success |
| **Training Loop** | Standard PPO with GAE | Same PPO, enhanced rewards |

### Key Advantages

1. **No Breaking Changes**: Existing PPO training code works unchanged
2. **Enhanced Capabilities**: Actions can now use web search, APIs, etc.
3. **Better Rewards**: Richer reward signals improve learning
4. **Modular Design**: Tools can be added/removed without retraining

### Example Training Flow

```python
# 1. PPO selects action (same as before)
action = policy_network(observation).sample()  # Returns 0, 1, 2, 3, or 4

# 2. Enhanced environment executes action with tools
if action == 0:  # RESPOND_WITH_LLM
    # LLM can now use browser tools during response generation
    response = llm_client.generate_with_tools(query, available_tools)
elif action == 1:  # QUERY_STATIC_KG
    # SPARQL execution enhanced with validation and caching
    result = kg_loader.execute_sparql_with_validation(query)

# 3. Reward computation includes tool success
reward = semantic_similarity(response, expected) + tool_success_bonus

# 4. PPO training continues normally
loss = ppo_loss(actions, rewards, advantages)
```

This approach **perfectly aligns** with the PyTorch RL tutorial's principles while adding powerful tool capabilities!

## Benefits of Integration

### 1. Enhanced Capabilities
- **Web Search Integration**: Agent can search for real-time information
- **API Tool Access**: Integration with external services and databases
- **Multi-Modal Reasoning**: Combine text, structured data, and tool outputs
- **Advanced Conversation Management**: Better context tracking and state management

### 2. Improved Training
- **Richer Action Space**: More diverse learning opportunities
- **Tool-Specific Rewards**: Better feedback for complex tasks
- **Transfer Learning**: Knowledge from web browsing can improve KG reasoning
- **Curriculum Learning**: Staged complexity introduction

### 3. Production Benefits
- **Real-World Applicability**: Tools enable practical task completion
- **Scalability**: Transform-based architecture supports easy extension
- **Maintainability**: Clear separation of concerns with modular design
- **Performance**: TorchRL's optimized tensor operations

## Migration Plan

### Phase 1: Foundation (Week 1-2)
- Implement `TorchRLKnowledgeGraphEnv` wrapper
- Create basic `KnowledgeGraphTransform`
- Ensure backward compatibility with existing tests

### Phase 2: Tool Integration (Week 3-4)  
- Add `BrowserTransform` integration
- Implement tool registry system
- Create enhanced action space

### Phase 3: Training Pipeline (Week 5-6)
- Develop curriculum learning strategy
- Implement hybrid reward system
- Create tool-enabled datasets

### Phase 4: Testing & Optimization (Week 7-8)
- Comprehensive testing of hybrid system
- Performance optimization
- Documentation and examples

## Risk Mitigation

### Technical Risks
1. **Event Loop Conflicts**: Use `nest_asyncio` for Jupyter compatibility
2. **Memory Usage**: Implement efficient tensor memory management
3. **Training Stability**: Gradual curriculum introduction prevents instability

### Compatibility Risks
1. **Breaking Changes**: Comprehensive backward compatibility testing
2. **Dependency Conflicts**: Careful version pinning and virtual environment management
3. **Model Migration**: Provide migration utilities for existing trained models

## Success Metrics

### Technical Metrics
- Backward compatibility: 100% of existing tests pass
- Performance: <10% overhead for traditional KG tasks
- Tool Success Rate: >80% successful tool executions

### Capability Metrics
- Enhanced Question Answering: 25% improvement on complex queries
- Tool Integration: Successful completion of multi-step tool workflows
- Training Efficiency: 30% faster convergence with curriculum learning

## Conclusion

This integration represents a significant enhancement to the rl-kg-agent project, combining the best of both worlds:

1. **Proven KG Reasoning**: Existing SPARQL and internal KG capabilities
2. **Modern Tool Framework**: TorchRL's composable, efficient tool system
3. **Enhanced Learning**: Richer action spaces and reward signals
4. **Production Ready**: Real-world tool integration for practical applications

The phased approach ensures minimal disruption while maximizing capability enhancement. The backward compatibility strategy protects existing investments while enabling future growth.

The result will be a state-of-the-art reinforcement learning agent that can reason over knowledge graphs AND interact with the broader digital ecosystem through tools - a truly hybrid intelligent system.