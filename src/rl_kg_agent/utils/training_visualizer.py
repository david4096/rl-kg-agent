"""Real-time training visualization dashboard."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Use non-GUI backend for macOS compatibility
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import threading
import time
import queue
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging
import os


logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    episode: int
    reward: float
    episode_length: int
    success_rate: float
    entropy: float
    policy_loss: float
    value_loss: float
    learning_rate: float
    timestamp: float


@dataclass
class KGMetrics:
    """Knowledge Graph metrics data structure."""
    total_nodes: int
    total_edges: int
    total_contexts: int
    avg_node_importance: float
    recent_queries: List[str]
    successful_actions: int
    failed_actions: int
    timestamp: float


class TrainingDashboard:
    """Real-time training visualization dashboard."""

    def __init__(self, max_points: int = 1000, update_interval: int = 100):
        """Initialize the training dashboard.

        Args:
            max_points: Maximum number of data points to keep in memory
            update_interval: Update interval in milliseconds
        """
        self.max_points = max_points
        self.update_interval = update_interval

        # Data queues for thread-safe communication
        self.training_queue = queue.Queue()
        self.kg_queue = queue.Queue()

        # Data storage
        self.training_data = deque(maxlen=max_points)
        self.kg_data = deque(maxlen=max_points)

        # GUI components
        self.root = None
        self.fig = None
        self.axes = {}
        self.canvas = None

        # Control variables
        self.running = False
        self.animation = None

        # Current status
        self.current_episode = 0
        self.start_time = time.time()

    def start_dashboard(self):
        """Start the dashboard GUI."""
        self.running = True
        
        # Create output directory for plots
        os.makedirs("training_plots", exist_ok=True)
        
        # Create plots using matplotlib with file output
        self._create_plots_file_based()
        
        # Start update thread
        update_thread = threading.Thread(target=self._update_loop, daemon=True)
        update_thread.start()

        logger.info("Training dashboard started (file-based plotting for macOS compatibility)")

    def stop_dashboard(self):
        """Stop the dashboard."""
        self.running = False
        logger.info("Training dashboard stopped")

    def update_training_metrics(self, metrics: TrainingMetrics):
        """Update training metrics."""
        try:
            self.training_queue.put(metrics, block=False)
        except queue.Full:
            # Remove old data if queue is full
            try:
                self.training_queue.get_nowait()
                self.training_queue.put(metrics, block=False)
            except queue.Empty:
                pass

    def update_kg_metrics(self, metrics: KGMetrics):
        """Update knowledge graph metrics."""
        try:
            self.kg_queue.put(metrics, block=False)
        except queue.Full:
            try:
                self.kg_queue.get_nowait()
                self.kg_queue.put(metrics, block=False)
            except queue.Empty:
                pass

    def _create_plots_file_based(self):
        """Create plots using file-based output for macOS compatibility."""
        self.fig, self.axes = plt.subplots(3, 3, figsize=(15, 12))
        self.fig.suptitle('RL-KG-Agent Training Dashboard', fontsize=16)
        
        # Initialize subplot titles
        subplot_titles = [
            'Episode Rewards', 'Policy Loss', 'Value Loss',
            'Episode Length', 'Success Rate', 'Learning Rate', 
            'Action Distribution', 'KG Growth', 'Performance Summary'
        ]
        
        for i, ax in enumerate(self.axes.flat):
            if i < len(subplot_titles):
                ax.set_title(subplot_titles[i])
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()

    def _update_loop(self):
        """Update loop for file-based plotting."""
        last_update = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Update every 5 seconds
            if current_time - last_update >= 5.0:
                self._process_queues()
                self._update_plots_file_based()
                last_update = current_time
            
            time.sleep(1.0)

    def _process_queues(self):
        """Process data from queues."""
        # Process training metrics
        while True:
            try:
                metrics = self.training_queue.get_nowait()
                self.training_data.append(metrics)
                self.current_episode = metrics.episode
            except queue.Empty:
                break
        
        # Process KG metrics
        while True:
            try:
                metrics = self.kg_queue.get_nowait()
                self.kg_data.append(metrics)
            except queue.Empty:
                break

    def _update_plots_file_based(self):
        """Update plots and save to file."""
        if not self.training_data:
            return
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Extract data
        episodes = [m.episode for m in self.training_data]
        rewards = [m.reward for m in self.training_data]
        policy_losses = [m.policy_loss for m in self.training_data]
        value_losses = [m.value_loss for m in self.training_data]
        episode_lengths = [m.episode_length for m in self.training_data]
        success_rates = [m.success_rate for m in self.training_data]
        learning_rates = [m.learning_rate for m in self.training_data]
        
        # Plot data
        self.axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.7)
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        self.axes[0, 1].plot(episodes, policy_losses, 'r-', alpha=0.7)
        self.axes[0, 1].set_title('Policy Loss')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Loss')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        self.axes[0, 2].plot(episodes, value_losses, 'g-', alpha=0.7)
        self.axes[0, 2].set_title('Value Loss')
        self.axes[0, 2].set_xlabel('Episode')
        self.axes[0, 2].set_ylabel('Loss')
        self.axes[0, 2].grid(True, alpha=0.3)
        
        self.axes[1, 0].plot(episodes, episode_lengths, 'm-', alpha=0.7)
        self.axes[1, 0].set_title('Episode Length')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].set_ylabel('Steps')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].plot(episodes, success_rates, 'c-', alpha=0.7)
        self.axes[1, 1].set_title('Success Rate')
        self.axes[1, 1].set_xlabel('Episode')
        self.axes[1, 1].set_ylabel('Rate')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        self.axes[1, 2].plot(episodes, learning_rates, 'orange', alpha=0.7)
        self.axes[1, 2].set_title('Learning Rate')
        self.axes[1, 2].set_xlabel('Episode')
        self.axes[1, 2].set_ylabel('Rate')
        self.axes[1, 2].grid(True, alpha=0.3)
        
        # Summary stats in bottom row
        if len(rewards) > 0:
            recent_rewards = rewards[-min(100, len(rewards)):]
            avg_reward = np.mean(recent_rewards)
            self.axes[2, 0].text(0.1, 0.8, f'Current Episode: {self.current_episode}', transform=self.axes[2, 0].transAxes)
            self.axes[2, 0].text(0.1, 0.6, f'Avg Reward (last 100): {avg_reward:.2f}', transform=self.axes[2, 0].transAxes)
            self.axes[2, 0].text(0.1, 0.4, f'Total Episodes: {len(self.training_data)}', transform=self.axes[2, 0].transAxes)
            self.axes[2, 0].text(0.1, 0.2, f'Runtime: {time.time() - self.start_time:.1f}s', transform=self.axes[2, 0].transAxes)
            self.axes[2, 0].set_title('Training Summary')
            self.axes[2, 0].set_xlim(0, 1)
            self.axes[2, 0].set_ylim(0, 1)
            self.axes[2, 0].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = "training_plots/training_dashboard.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        
        logger.info(f"Training plots updated: {plot_file} (Episode {self.current_episode})")

    def _create_gui(self):
        """Create the GUI interface."""
        self.root = tk.Tk()
        self.root.title("RL-KG-Agent Training Dashboard")
        self.root.geometry("1400x900")

        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create status panel
        self._create_status_panel(main_frame)

        # Create matplotlib figure
        self._create_plots(main_frame)

        # Create control panel
        self._create_control_panel(main_frame)

        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self._update_plots, interval=self.update_interval, blit=False
        )

        # Start GUI event loop
        self.root.mainloop()

    def _create_status_panel(self, parent):
        """Create status information panel."""
        status_frame = ttk.LabelFrame(parent, text="Training Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        # Create status variables
        self.status_vars = {
            'episode': tk.StringVar(value="Episode: 0"),
            'runtime': tk.StringVar(value="Runtime: 0:00:00"),
            'avg_reward': tk.StringVar(value="Avg Reward: 0.0"),
            'success_rate': tk.StringVar(value="Success Rate: 0%"),
            'kg_nodes': tk.StringVar(value="KG Nodes: 0"),
            'kg_edges': tk.StringVar(value="KG Edges: 0"),
            'recent_action': tk.StringVar(value="Recent Action: None")
        }

        # Create status labels
        for i, (key, var) in enumerate(self.status_vars.items()):
            row, col = divmod(i, 4)
            label = ttk.Label(status_frame, textvariable=var, font=('Arial', 10, 'bold'))
            label.grid(row=row, column=col, sticky=tk.W, padx=20, pady=5)

    def _create_plots(self, parent):
        """Create matplotlib plots."""
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure with subplots
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor('white')

        # Define subplot layout (3 rows, 3 columns)
        self.axes = {
            'rewards': self.fig.add_subplot(331),
            'episode_length': self.fig.add_subplot(332),
            'success_rate': self.fig.add_subplot(333),
            'policy_loss': self.fig.add_subplot(334),
            'value_loss': self.fig.add_subplot(335),
            'entropy': self.fig.add_subplot(336),
            'kg_nodes': self.fig.add_subplot(337),
            'kg_contexts': self.fig.add_subplot(338),
            'action_distribution': self.fig.add_subplot(339)
        }

        # Configure axes
        self._configure_axes()

        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tight layout
        plt.tight_layout()

    def _configure_axes(self):
        """Configure subplot axes."""
        configs = {
            'rewards': {'title': 'Episode Rewards', 'ylabel': 'Reward', 'color': 'blue'},
            'episode_length': {'title': 'Episode Length', 'ylabel': 'Steps', 'color': 'green'},
            'success_rate': {'title': 'Success Rate', 'ylabel': 'Rate (%)', 'color': 'orange'},
            'policy_loss': {'title': 'Policy Loss', 'ylabel': 'Loss', 'color': 'red'},
            'value_loss': {'title': 'Value Loss', 'ylabel': 'Loss', 'color': 'purple'},
            'entropy': {'title': 'Policy Entropy', 'ylabel': 'Entropy', 'color': 'brown'},
            'kg_nodes': {'title': 'Knowledge Graph Nodes', 'ylabel': 'Count', 'color': 'teal'},
            'kg_contexts': {'title': 'Stored Contexts', 'ylabel': 'Count', 'color': 'pink'},
            'action_distribution': {'title': 'Action Distribution', 'ylabel': 'Frequency', 'color': 'gray'}
        }

        for name, ax in self.axes.items():
            config = configs[name]
            ax.set_title(config['title'], fontsize=10, fontweight='bold')
            ax.set_ylabel(config['ylabel'], fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=8)

    def _create_control_panel(self, parent):
        """Create control panel."""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        # Control buttons
        ttk.Button(control_frame, text="Save Plots",
                  command=self._save_plots).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear Data",
                  command=self._clear_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Data",
                  command=self._export_data).pack(side=tk.LEFT, padx=5)

        # Recent queries display
        query_frame = ttk.LabelFrame(control_frame, text="Recent Queries", padding=5)
        query_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))

        self.query_text = tk.Text(query_frame, height=3, width=60, font=('Arial', 8))
        query_scroll = ttk.Scrollbar(query_frame, orient=tk.VERTICAL, command=self.query_text.yview)
        self.query_text.configure(yscrollcommand=query_scroll.set)

        self.query_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        query_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _update_plots(self, frame):
        """Update plots with new data."""
        if not self.running:
            return

        # Process queued data
        self._process_training_queue()
        self._process_kg_queue()

        # Update status
        self._update_status()

        # Update plots
        self._update_training_plots()
        self._update_kg_plots()
        self._update_action_distribution()

    def _process_training_queue(self):
        """Process training metrics from queue."""
        while True:
            try:
                metrics = self.training_queue.get_nowait()
                self.training_data.append(metrics)
                self.current_episode = metrics.episode
            except queue.Empty:
                break

    def _process_kg_queue(self):
        """Process KG metrics from queue."""
        while True:
            try:
                metrics = self.kg_queue.get_nowait()
                self.kg_data.append(metrics)
            except queue.Empty:
                break

    def _update_status(self):
        """Update status panel."""
        if not self.training_data:
            return

        latest = self.training_data[-1]
        runtime = time.time() - self.start_time
        hours, remainder = divmod(int(runtime), 3600)
        minutes, seconds = divmod(remainder, 60)

        # Calculate recent averages (last 100 episodes)
        recent_rewards = [d.reward for d in list(self.training_data)[-100:]]
        recent_success = [d.success_rate for d in list(self.training_data)[-100:]]

        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_success = np.mean(recent_success) if recent_success else 0

        # Update status variables
        self.status_vars['episode'].set(f"Episode: {self.current_episode}")
        self.status_vars['runtime'].set(f"Runtime: {hours}:{minutes:02d}:{seconds:02d}")
        self.status_vars['avg_reward'].set(f"Avg Reward: {avg_reward:.3f}")
        self.status_vars['success_rate'].set(f"Success Rate: {avg_success:.1f}%")

        if self.kg_data:
            latest_kg = self.kg_data[-1]
            self.status_vars['kg_nodes'].set(f"KG Nodes: {latest_kg.total_nodes}")
            self.status_vars['kg_edges'].set(f"KG Edges: {latest_kg.total_edges}")

    def _update_training_plots(self):
        """Update training-related plots."""
        if not self.training_data:
            return

        episodes = [d.episode for d in self.training_data]

        # Plot data
        plots = {
            'rewards': [d.reward for d in self.training_data],
            'episode_length': [d.episode_length for d in self.training_data],
            'success_rate': [d.success_rate * 100 for d in self.training_data],
            'policy_loss': [d.policy_loss for d in self.training_data],
            'value_loss': [d.value_loss for d in self.training_data],
            'entropy': [d.entropy for d in self.training_data]
        }

        for name, data in plots.items():
            ax = self.axes[name]
            ax.clear()
            ax.plot(episodes, data, linewidth=1.5)

            # Add moving average if enough data
            if len(data) > 20:
                window = min(50, len(data) // 4)
                moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')
                avg_episodes = episodes[window-1:]
                ax.plot(avg_episodes, moving_avg, 'r-', alpha=0.7, linewidth=2, label='Moving Avg')
                ax.legend(fontsize=8)

            # Reconfigure axis
            config = {
                'rewards': {'title': 'Episode Rewards', 'ylabel': 'Reward'},
                'episode_length': {'title': 'Episode Length', 'ylabel': 'Steps'},
                'success_rate': {'title': 'Success Rate', 'ylabel': 'Rate (%)'},
                'policy_loss': {'title': 'Policy Loss', 'ylabel': 'Loss'},
                'value_loss': {'title': 'Value Loss', 'ylabel': 'Loss'},
                'entropy': {'title': 'Policy Entropy', 'ylabel': 'Entropy'}
            }[name]

            ax.set_title(config['title'], fontsize=10, fontweight='bold')
            ax.set_ylabel(config['ylabel'], fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=8)

    def _update_kg_plots(self):
        """Update knowledge graph plots."""
        if not self.kg_data:
            return

        timestamps = [(d.timestamp - self.start_time) / 60 for d in self.kg_data]  # Convert to minutes

        # KG Nodes plot
        ax = self.axes['kg_nodes']
        ax.clear()
        nodes_data = [d.total_nodes for d in self.kg_data]
        ax.plot(timestamps, nodes_data, 'teal', linewidth=2)
        ax.set_title('Knowledge Graph Nodes', fontsize=10, fontweight='bold')
        ax.set_ylabel('Node Count', fontsize=9)
        ax.set_xlabel('Time (minutes)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)

        # KG Contexts plot
        ax = self.axes['kg_contexts']
        ax.clear()
        contexts_data = [d.total_contexts for d in self.kg_data]
        ax.plot(timestamps, contexts_data, 'pink', linewidth=2)
        ax.set_title('Stored Contexts', fontsize=10, fontweight='bold')
        ax.set_ylabel('Context Count', fontsize=9)
        ax.set_xlabel('Time (minutes)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)

    def _update_action_distribution(self):
        """Update action distribution plot."""
        if not self.kg_data:
            return

        # Get recent action data (last entry)
        latest_kg = self.kg_data[-1]

        ax = self.axes['action_distribution']
        ax.clear()

        # Create mock action distribution (would be passed in real implementation)
        actions = ['Respond LLM', 'Query KG', 'Store KG', 'Refine Q', 'Plan']
        success_counts = [latest_kg.successful_actions // 5] * 5  # Mock distribution
        failed_counts = [latest_kg.failed_actions // 5] * 5

        x = np.arange(len(actions))
        width = 0.35

        ax.bar(x - width/2, success_counts, width, label='Success', color='green', alpha=0.7)
        ax.bar(x + width/2, failed_counts, width, label='Failed', color='red', alpha=0.7)

        ax.set_title('Action Distribution', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(actions, fontsize=8, rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Update recent queries text
        if hasattr(latest_kg, 'recent_queries') and latest_kg.recent_queries:
            self.query_text.delete(1.0, tk.END)
            for query in latest_kg.recent_queries[-5:]:  # Show last 5 queries
                self.query_text.insert(tk.END, f"• {query}\n")
            self.query_text.see(tk.END)

    def _save_plots(self):
        """Save current plots to file."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"training_plots_{timestamp}.png"
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Plots saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save plots: {e}")

    def _clear_data(self):
        """Clear all data."""
        self.training_data.clear()
        self.kg_data.clear()
        self.current_episode = 0
        self.start_time = time.time()
        logger.info("Dashboard data cleared")

    def _export_data(self):
        """Export data to CSV."""
        try:
            import pandas as pd
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            if self.training_data:
                training_df = pd.DataFrame([
                    {
                        'episode': d.episode,
                        'reward': d.reward,
                        'episode_length': d.episode_length,
                        'success_rate': d.success_rate,
                        'entropy': d.entropy,
                        'policy_loss': d.policy_loss,
                        'value_loss': d.value_loss,
                        'timestamp': d.timestamp
                    } for d in self.training_data
                ])
                training_df.to_csv(f"training_data_{timestamp}.csv", index=False)

            if self.kg_data:
                kg_df = pd.DataFrame([
                    {
                        'total_nodes': d.total_nodes,
                        'total_edges': d.total_edges,
                        'total_contexts': d.total_contexts,
                        'avg_node_importance': d.avg_node_importance,
                        'successful_actions': d.successful_actions,
                        'failed_actions': d.failed_actions,
                        'timestamp': d.timestamp
                    } for d in self.kg_data
                ])
                kg_df.to_csv(f"kg_data_{timestamp}.csv", index=False)

            logger.info(f"Data exported with timestamp {timestamp}")
        except ImportError:
            logger.warning("pandas not available for data export")
        except Exception as e:
            logger.error(f"Failed to export data: {e}")


class TrainingMonitor:
    """Monitor for collecting and sending metrics to dashboard."""

    def __init__(self, dashboard: TrainingDashboard, internal_kg):
        """Initialize training monitor.

        Args:
            dashboard: Dashboard instance to send metrics to
            internal_kg: Internal knowledge graph to monitor
        """
        self.dashboard = dashboard
        self.internal_kg = internal_kg
        self.episode_count = 0
        self.recent_queries = deque(maxlen=50)
        self.successful_actions = 0
        self.failed_actions = 0

    def log_episode(self, reward: float, episode_length: int, success_rate: float,
                   entropy: float = 0.0, policy_loss: float = 0.0, value_loss: float = 0.0,
                   learning_rate: float = 3e-4):
        """Log episode metrics."""
        self.episode_count += 1

        metrics = TrainingMetrics(
            episode=self.episode_count,
            reward=reward,
            episode_length=episode_length,
            success_rate=success_rate,
            entropy=entropy,
            policy_loss=policy_loss,
            value_loss=value_loss,
            learning_rate=learning_rate,
            timestamp=time.time()
        )

        self.dashboard.update_training_metrics(metrics)

        # Also update KG metrics
        self._update_kg_metrics()

    def log_query(self, query: str):
        """Log a query."""
        if query and query.strip():
            self.recent_queries.append(query.strip())

    def log_action_result(self, success: bool):
        """Log action result."""
        if success:
            self.successful_actions += 1
        else:
            self.failed_actions += 1

    def _update_kg_metrics(self):
        """Update knowledge graph metrics."""
        try:
            stats = self.internal_kg.get_stats()

            metrics = KGMetrics(
                total_nodes=stats.get('total_nodes', 0),
                total_edges=stats.get('total_edges', 0),
                total_contexts=stats.get('total_contexts', 0),
                avg_node_importance=stats.get('avg_node_importance', 0.0),
                recent_queries=list(self.recent_queries)[-5:],  # Last 5 queries
                successful_actions=self.successful_actions,
                failed_actions=self.failed_actions,
                timestamp=time.time()
            )

            self.dashboard.update_kg_metrics(metrics)

        except Exception as e:
            logger.error(f"Failed to update KG metrics: {e}")