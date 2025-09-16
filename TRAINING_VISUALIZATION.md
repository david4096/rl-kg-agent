# Training Visualization Dashboard

This document explains how to use the live training visualization dashboard to monitor your RL-KG-Agent training in real-time.

## Installation

First, install the visualization dependencies:

```bash
# Install with visualization support
uv add --optional visualization

# Or install manually
uv add matplotlib
```

Note: `tkinter` is usually included with Python installations. If not available, install it using your system package manager:
- Ubuntu/Debian: `sudo apt-get install python3-tk`
- macOS: Usually included with Python
- Windows: Usually included with Python

## Quick Start

### Using the CLI

Train with live visualization using the `--visualize` flag:

```bash
# Basic training with visualization
uv run rl-kg-agent train --ttl-file examples/simple_knowledge_graph.ttl --visualize

# Advanced training with custom parameters
uv run rl-kg-agent train \
  --ttl-file examples/simple_knowledge_graph.ttl \
  --dataset squad \
  --episodes 2000 \
  --output-model my_model \
  --visualize
```

### Using the Python API

```python
from rl_kg_agent.utils.training_visualizer import TrainingDashboard, TrainingMonitor
from rl_kg_agent.agents.ppo_agent import PPOKGAgent
# ... other imports

# Set up components
dashboard = TrainingDashboard()
monitor = TrainingMonitor(dashboard, internal_kg)
dashboard.start_dashboard()

# Train with visualization
agent = PPOKGAgent(action_manager, reward_calculator, internal_kg)
agent.train(total_timesteps=10000, training_monitor=monitor)
```

### Running the Demo

Try the comprehensive demo:

```bash
uv run python examples/training_with_visualization.py
```

## Dashboard Features

The training dashboard provides real-time monitoring with the following visualizations:

### Training Metrics (Top Row)
1. **Episode Rewards**: Shows reward progression over episodes with moving average
2. **Episode Length**: Number of steps per episode
3. **Success Rate**: Percentage of successful episodes

### Training Diagnostics (Middle Row)
4. **Policy Loss**: PPO policy gradient loss
5. **Value Loss**: Value function estimation loss
6. **Policy Entropy**: Exploration vs exploitation balance

### Knowledge Graph Status (Bottom Row)
7. **KG Nodes**: Growth of internal knowledge graph nodes over time
8. **Stored Contexts**: Number of query-response contexts stored
9. **Action Distribution**: Success/failure rates by action type

### Status Panel
- **Current Episode**: Episode number
- **Runtime**: Total training time
- **Average Reward**: Recent average reward (last 100 episodes)
- **Success Rate**: Recent success percentage
- **KG Stats**: Current internal KG size
- **Recent Action**: Most recent action taken

### Recent Queries Panel
Shows the last 5 queries processed by the agent, helping you understand what the agent is learning.

## Dashboard Controls

- **Save Plots**: Export current visualizations as PNG
- **Clear Data**: Reset all visualization data
- **Export Data**: Save training metrics to CSV files

## Understanding the Visualizations

### Good Training Signs
- **Rewards**: Trending upward with decreasing volatility
- **Success Rate**: Increasing over time
- **Policy Loss**: Decreasing and stabilizing
- **KG Growth**: Steady increase in nodes/contexts

### Potential Issues
- **Flat Rewards**: May indicate learning plateau
- **High Value Loss**: Value function estimation problems
- **Low Entropy**: Insufficient exploration
- **No KG Growth**: Knowledge not being retained

## Advanced Usage

### Custom Metrics

You can log custom metrics to the dashboard:

```python
# Log episode results
monitor.log_episode(
    reward=episode_reward,
    episode_length=steps_taken,
    success_rate=success_percentage,
    entropy=policy_entropy,
    policy_loss=pg_loss,
    value_loss=vf_loss
)

# Log queries and actions
monitor.log_query("What is the capital of France?")
monitor.log_action_result(success=True)
```

### Dashboard Configuration

Customize the dashboard behavior:

```python
dashboard = TrainingDashboard(
    max_points=2000,      # Maximum data points to keep
    update_interval=50    # Update frequency (ms)
)
```

### Data Export

The dashboard can export data for further analysis:

- **CSV Export**: Training metrics and KG statistics
- **Plot Export**: High-resolution PNG images
- **Timestamped Files**: Automatic naming with timestamps

## Troubleshooting

### Common Issues

1. **Dashboard doesn't appear**
   - Check if `tkinter` is installed
   - Verify matplotlib installation
   - Try running from command line to see error messages

2. **Slow updates**
   - Reduce `update_interval` parameter
   - Decrease `max_points` to keep less history

3. **Memory usage**
   - Use "Clear Data" button periodically
   - Reduce `max_points` parameter
   - Close dashboard when not needed

### Performance Tips

- The dashboard runs in a separate thread to minimize training impact
- Visualization updates are batched for efficiency
- Data is automatically pruned to prevent memory issues

## Integration with Weights & Biases

You can also log metrics to W&B alongside the live dashboard:

```python
import wandb

# Initialize W&B
wandb.init(project="rl-kg-agent")

# In your training loop
wandb.log({
    "episode_reward": reward,
    "success_rate": success_rate,
    "kg_nodes": internal_kg.get_stats()["total_nodes"]
})
```

## Examples

See the `examples/` directory for:
- `training_with_visualization.py`: Comprehensive demo
- `basic_usage.py`: Simple training setup
- `custom_visualization.py`: Advanced customization

## FAQ

**Q: Can I use the dashboard with custom training loops?**
A: Yes! Create a `TrainingMonitor` and call `log_episode()` with your metrics.

**Q: Does visualization slow down training?**
A: Minimal impact - updates run in background threads with batched processing.

**Q: Can I customize the plots?**
A: Yes, modify the `TrainingDashboard` class or create custom visualizations using the exported data.

**Q: What if I don't have a GUI environment?**
A: The dashboard requires a GUI. For headless servers, use the data export features and create plots offline.