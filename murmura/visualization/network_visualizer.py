import os
from typing import Dict, List, Optional, Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.artist import Artist

from murmura.network_management.topology_manager import TopologyManager
from murmura.visualization.training_event import (
    TrainingEvent,
    LocalTrainingEvent,
    ParameterTransferEvent,
    AggregationEvent,
    ModelUpdateEvent,
    EvaluationEvent,
    InitialStateEvent,
)
from murmura.visualization.training_observer import TrainingObserver


class NetworkVisualizer(TrainingObserver):
    """
    Creates visualizations of the training process based on events.

    This visualizer creates both animations and frame sequences showing how
    model parameters flow through the network during training.
    """

    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize the network visualizer.

        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.topology: Optional[Dict[int, List[int]]] = None
        self.network_type: Optional[str] = None
        self.frames: List[Dict[str, Any]] = []
        # Use dictionaries instead of lists to store metric history by round number
        self.accuracy_history: Dict[int, float] = {}
        self.loss_history: Dict[int, float] = {}
        self.parameter_history: Dict[int, List[float]] = {}
        self.frame_descriptions: List[str] = []
        self.round_metrics: Dict[int, Dict[str, float]] = {}  # Store metrics by round
        # Track at which frame each parameter update occurs
        self.parameter_update_frames: Dict[int, Dict[int, int]] = {}  # {node_id: {parameter_idx: frame_idx}}

    def set_topology(self, topology_manager: TopologyManager) -> None:
        """
        Set the network topology information.

        Args:
            topology_manager: Topology manager containing network structure
        """
        self.topology = topology_manager.adjacency_list
        self.network_type = topology_manager.config.topology_type.value

    def on_event(self, event: TrainingEvent) -> None:
        """
        Process training events to build visualization data.

        Args:
            event: The training event to process
        """
        if self.topology is None and not isinstance(event, InitialStateEvent):
            return  # Can't visualize without topology information

        frame: Dict[str, Any] = {
            "round": event.round_num,
            "step": event.step_name,
            "timestamp": event.timestamp,
            "topology_type": self.network_type,
            "active_nodes": [],
            "active_edges": [],
            "node_params": {},
            "metrics": {},
        }

        description = f"Round {event.round_num}: {event.step_name}"

        if isinstance(event, InitialStateEvent):
            self.network_type = event.topology_type
            frame["topology_type"] = event.topology_type
            frame["num_nodes"] = event.num_nodes
            description = "Initial network state"

        elif isinstance(event, LocalTrainingEvent):
            frame["active_nodes"] = event.active_nodes
            frame["metrics"] = event.metrics
            description = f"Round {event.round_num}: Local training on {len(event.active_nodes)} nodes"

        elif isinstance(event, ParameterTransferEvent):
            for source in event.source_nodes:
                for target in event.target_nodes:
                    frame["active_edges"].append((source, target))

            # Track parameter summaries for visualization
            for node, summary in event.param_summary.items():
                if node not in self.parameter_history:
                    self.parameter_history[node] = []
                    self.parameter_update_frames[node] = {}

                if isinstance(summary, dict) and "norm" in summary:
                    # Store parameter update for this round
                    current_round = event.round_num
                    norm_value = summary["norm"]

                    # Only add if this is a new round or first update
                    if not self.parameter_history[node] or current_round > len(self.parameter_history[node]):
                        self.parameter_history[node].append(norm_value)
                        # Store at which frame this parameter update happened
                        self.parameter_update_frames[node][len(self.parameter_history[node]) - 1] = len(self.frames)
                else:
                    # If we just have raw parameters, use a simple norm
                    if not self.parameter_history[node] or event.round_num > len(self.parameter_history[node]):
                        self.parameter_history[node].append(1.0)  # Placeholder
                        self.parameter_update_frames[node][len(self.parameter_history[node]) - 1] = len(self.frames)

            frame["node_params"] = event.param_summary
            description = (
                f"Round {event.round_num}: Parameter transfer from {len(event.source_nodes)} "
                f"nodes to {len(event.target_nodes)} nodes"
            )

        elif isinstance(event, AggregationEvent):
            frame["active_nodes"] = event.participating_nodes
            frame["strategy_name"] = event.strategy_name

            if event.aggregator_node is not None:
                # For star topology, show edges to aggregator
                for node in event.participating_nodes:
                    if node != event.aggregator_node:
                        frame["active_edges"].append((node, event.aggregator_node))
                description = (
                    f"Round {event.round_num}: Aggregation at node {event.aggregator_node} "
                    f"using {event.strategy_name}"
                )
            else:
                description = f"Round {event.round_num}: Decentralized aggregation using {event.strategy_name}"

        elif isinstance(event, ModelUpdateEvent):
            frame["active_nodes"] = event.updated_nodes
            frame["param_convergence"] = event.param_convergence
            description = f"Round {event.round_num}: Model update on {len(event.updated_nodes)} nodes"

        elif isinstance(event, EvaluationEvent):
            frame["metrics"] = event.metrics

            # Store metrics by round number
            self.round_metrics[event.round_num] = event.metrics

            # Only update global lists if we have valid metrics
            if "accuracy" in event.metrics:
                # Store accuracy value by round number
                accuracy_value = event.metrics["accuracy"]
                self.accuracy_history[event.round_num] = accuracy_value

            if "loss" in event.metrics:
                # Store loss value by round number
                loss_value = event.metrics["loss"]
                self.loss_history[event.round_num] = loss_value

            description = f"Round {event.round_num}: Evaluation - "
            if "accuracy" in event.metrics:
                description += f"Accuracy: {event.metrics['accuracy']:.4f}"
            if "loss" in event.metrics:
                description += f", Loss: {event.metrics['loss']:.4f}"

        # Ensure all metrics data is available for all frames
        frame["all_metrics"] = self.round_metrics.copy()
        # Add current metrics history to each frame for consistent animation
        frame["accuracy_history"] = self.accuracy_history.copy()
        frame["loss_history"] = self.loss_history.copy()
        frame["parameter_history"] = {node: history.copy() for node, history in self.parameter_history.items()}

        self.frames.append(frame)
        self.frame_descriptions.append(description)

    def render_training_animation(
            self, filename: str = "training_animation.mp4", fps: int = 1
    ) -> None:
        """
        Creates an animation of the training process.

        Args:
            filename: Output filename for the animation
            fps: Frames per second
        """
        if not self.frames:
            print("No frames captured. Run training first.")
            return

        # Create figure for the animation
        fig = plt.figure(figsize=(16, 10))
        gs = plt.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[2, 1])
        ax_network = fig.add_subplot(
            gs[:, 0]
        )  # Network visualization (left side, full height)
        ax_params = fig.add_subplot(gs[0, 1])  # Parameter convergence (top right)
        ax_metrics = fig.add_subplot(gs[1, 1])  # Metrics (bottom right)

        # Create metrics twin axis once
        ax_metrics_twin = ax_metrics.twinx()
        ax_metrics_twin.set_ylabel("Loss", color="r")

        # Base graph structure from the topology
        G = nx.Graph()

        if self.topology:
            for node in self.topology.keys():
                G.add_node(node)

            for node, neighbors in self.topology.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
        else:
            # If no topology was set, create a default graph from the first frame
            num_nodes = self.frames[0].get("num_nodes", 5)
            for i in range(num_nodes):
                G.add_node(i)

        # Calculate layout once for consistency
        pos = nx.spring_layout(G, seed=42)

        # Create text objects for titles and descriptions
        ax_network.set_title("Network State", fontsize=14)
        ax_params.set_title("Parameter Convergence", fontsize=12)
        ax_metrics.set_title("Training Metrics", fontsize=12)
        description_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12)

        def update_frame(frame_idx: int) -> List[Artist]:
            """Update animation frame.

            Args:
                frame_idx: Index of the current frame

            Returns:
                List of artists that were updated
            """
            if frame_idx >= len(self.frames):
                return []

            frame = self.frames[frame_idx]
            current_round = frame["round"]

            # Clear previous plots
            ax_network.clear()
            ax_params.clear()
            ax_metrics.clear()
            ax_metrics_twin.clear()

            # Update titles
            ax_network.set_title("Network State", fontsize=14)
            ax_params.set_title("Parameter Convergence", fontsize=12)
            ax_metrics.set_title("Training Metrics", fontsize=12)
            description_text.set_text(self.frame_descriptions[frame_idx])

            # Draw network state
            # Regular edges
            nx.draw_networkx_edges(
                G, pos, ax=ax_network, edge_color="gray", alpha=0.3, width=1.0
            )

            # Active edges (data transfer)
            if frame.get("active_edges"):
                edge_list = frame["active_edges"]
                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=ax_network,
                    edgelist=edge_list,
                    edge_color="red",
                    width=2.0,
                )

            # Node colors based on activity
            node_colors = []
            node_sizes = []

            for node in G.nodes():
                if node in frame.get("active_nodes", []):
                    node_colors.append("orange")
                    node_sizes.append(1200)
                else:
                    node_colors.append("skyblue")
                    node_sizes.append(800)

            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos, ax=ax_network, node_color=node_colors, node_size=node_sizes
            )

            # Node labels
            nx.draw_networkx_labels(G, pos, ax=ax_network, font_size=10)

            # Add topology type text
            topology_text = f"Topology: {frame.get('topology_type', 'Unknown')}"
            if "strategy_name" in frame:
                topology_text += f"\nStrategy: {frame['strategy_name']}"
            ax_network.text(
                0.02,
                0.02,
                topology_text,
                transform=ax_network.transAxes,
                fontsize=10,
                verticalalignment="bottom",
            )

            # Plot parameter convergence - show all visible parameters up to current round
            added_legend_entries = False
            for node, history in self.parameter_history.items():
                if len(history) > 0:  # Show all nodes with parameter history
                    # Only show data up to current round
                    # Use the round number rather than frame index for x-axis
                    visible_data = []
                    for i in range(min(current_round, len(history))):
                        visible_data.append(history[i])

                    if visible_data:
                        # Create a simple list of integers for the x-axis
                        rounds = []
                        for r in range(1, len(visible_data) + 1):
                            rounds.append(r)
                        ax_params.plot(rounds, visible_data, label=f"Node {node}")
                        added_legend_entries = True

            ax_params.set_xlabel("Round")
            ax_params.set_ylabel("Parameter Norm")

            # Set x-axis ticks to be integers
            if current_round > 0:
                ax_params.set_xticks(range(1, current_round + 1))

            # Only add legend if we have data
            if added_legend_entries:
                ax_params.legend(loc="upper right", fontsize=8)

            # Plot metrics - show metrics up to current round
            rounds = []
            accuracy_data = []
            loss_data = []

            # Find all available round numbers up to the current round
            available_rounds = sorted([r for r in range(1, current_round + 1)])

            # For each round number, get the latest metrics value
            last_acc = 0
            last_loss = 0

            for round_num in available_rounds:
                rounds.append(round_num)

                # Get accuracy for this round or use the last known value
                if round_num in self.accuracy_history:
                    last_acc = self.accuracy_history[round_num]
                accuracy_data.append(last_acc)

                # Get loss for this round or use the last known value
                if round_num in self.loss_history:
                    last_loss = self.loss_history[round_num]
                loss_data.append(last_loss)

            # Plot metrics if we have data
            legend_handles = []
            legend_labels = []

            if rounds and accuracy_data:
                # Plot accuracy on left axis
                (acc_line,) = ax_metrics.plot(
                    rounds, accuracy_data, "g-o", label="Accuracy"
                )
                ax_metrics.set_ylabel("Accuracy", color="g")
                legend_handles.append(acc_line)
                legend_labels.append("Accuracy")

            if rounds and loss_data:
                # Plot loss on right axis
                (loss_line,) = ax_metrics_twin.plot(
                    rounds, loss_data, "r-o", label="Loss"
                )
                ax_metrics_twin.set_ylabel("Loss", color="r")
                legend_handles.append(loss_line)
                legend_labels.append("Loss")

            ax_metrics.set_xlabel("Round")

            # Set x-axis ticks to be integers
            if rounds:
                ax_metrics.set_xticks(rounds)

            # Add legend if we have data
            if legend_handles:
                ax_metrics.legend(legend_handles, legend_labels, loc="upper left")

            # Set proper limits
            ax_network.set_xlim((-1.2, 1.2))
            ax_network.set_ylim((-1.2, 1.2))
            ax_network.axis("off")

            # Return updated artists (required for animation)
            return [
                ax_network,
                ax_params,
                ax_metrics,
                ax_metrics_twin,
                description_text,
            ]

        # Create animation with proper typing
        ani = animation.FuncAnimation(
            fig=fig,
            func=update_frame,
            frames=len(self.frames),
            interval=1000 / fps,
            blit=False,
        )

        # Save animation
        writer = animation.FFMpegWriter(
            fps=fps, metadata=dict(artist="Murmura Framework"), bitrate=5000
        )
        output_path = os.path.join(self.output_dir, filename)
        ani.save(output_path, writer=writer)

        plt.close(fig)
        print(f"Animation saved to {output_path}")

    def render_frame_sequence(self, prefix: str = "training_step") -> None:
        """
        Renders each frame as a separate image file.

        Args:
            prefix: Prefix for output filenames
        """
        for i, frame in enumerate(self.frames):
            # Create a figure with network and metrics
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Create graph
            G = nx.Graph()

            if self.topology:
                for node in self.topology.keys():
                    G.add_node(node)

                for node, neighbors in self.topology.items():
                    for neighbor in neighbors:
                        G.add_edge(node, neighbor)
            else:
                # If no topology was set, create a default graph
                num_nodes = frame.get("num_nodes", 5)
                for j in range(num_nodes):
                    G.add_node(j)

            # Calculate layout
            pos = nx.spring_layout(G, seed=42)

            # Regular edges
            nx.draw_networkx_edges(G, pos, ax=ax1, edge_color="gray", alpha=0.3)

            # Active edges
            if frame.get("active_edges"):
                edge_list = frame["active_edges"]
                nx.draw_networkx_edges(
                    G, pos, ax=ax1, edgelist=edge_list, edge_color="red", width=2.0
                )

            # Nodes
            node_colors = []
            node_sizes = []

            for node in G.nodes():
                if node in frame.get("active_nodes", []):
                    node_colors.append("orange")
                    node_sizes.append(1000)
                else:
                    node_colors.append("skyblue")
                    node_sizes.append(800)

            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos, ax=ax1, node_color=node_colors, node_size=node_sizes
            )

            # Labels
            nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10)

            # Title and description
            description = (
                self.frame_descriptions[i] if i < len(self.frame_descriptions) else ""
            )
            ax1.set_title(description)

            # Add topology type
            topology_text = f"Topology: {frame.get('topology_type', 'Unknown')}"
            if "strategy_name" in frame:
                topology_text += f"\nStrategy: {frame['strategy_name']}"
            ax1.text(
                0.02,
                0.02,
                topology_text,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment="bottom",
            )

            # Plot metrics if available - accessing round_metrics for consistent data
            has_metrics = False

            if frame["all_metrics"]:
                acc_data = []
                loss_data = []
                rounds = []

                for round_num in sorted(frame["all_metrics"].keys()):
                    if round_num <= frame["round"]:
                        metrics = frame["all_metrics"][round_num]
                        rounds.append(round_num)

                        if "accuracy" in metrics:
                            acc_data.append(metrics["accuracy"])
                        if "loss" in metrics:
                            loss_data.append(metrics["loss"])

                if rounds:
                    has_metrics = True
                    # Add accuracy plot if available
                    if acc_data:
                        ax2.plot(rounds, acc_data, "g-o", label="Accuracy")
                        ax2.set_ylabel("Accuracy", color="g")

                    # Add loss plot with twin axis if available
                    if loss_data:
                        ax2_twin = ax2.twinx()
                        ax2_twin.plot(rounds, loss_data, "r-o", label="Loss")
                        ax2_twin.set_ylabel("Loss", color="r")

                    # Add legend
                    handles, labels = [], []
                    if acc_data:
                        h1, l1 = ax2.get_legend_handles_labels()
                        handles.extend(h1)
                        labels.extend(l1)
                    if loss_data and "ax2_twin" in locals():
                        h2, l2 = locals()["ax2_twin"].get_legend_handles_labels()
                        handles.extend(h2)
                        labels.extend(l2)

                    if handles:
                        ax2.legend(handles, labels, loc="upper left")

            if has_metrics:
                ax2.set_title("Training Metrics")
                ax2.set_xlabel("Round")
            else:
                ax2.set_title("No metrics available")
                ax2.set_xlabel("Round")
                ax2.set_ylabel("Value")

            # Adjust layout
            ax1.set_xlim([-1.2, 1.2])
            ax1.set_ylim([-1.2, 1.2])
            ax1.axis("off")

            plt.tight_layout()

            # Save the figure
            filename = f"{prefix}_{i:03d}.png"
            plt.savefig(
                os.path.join(self.output_dir, filename), dpi=100, bbox_inches="tight"
            )
            plt.close(fig)

        print(f"Frame sequence saved to {self.output_dir} ({len(self.frames)} frames)")

    def render_summary_plot(self, filename: str = "training_summary.png") -> None:
        """
        Creates a summary plot showing key metrics and parameter convergence.

        Args:
            filename: Output filename for the summary plot
        """
        if not self.frames:
            print("No frames captured. Run training first.")
            return

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Network topology (top left)
        ax_network = axes[0, 0]
        G = nx.Graph()

        if self.topology:
            for node in self.topology.keys():
                G.add_node(node)

            for node, neighbors in self.topology.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)

            # Draw network
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx(
                G, pos, ax=ax_network, node_color="skyblue", node_size=800, font_size=10
            )

            ax_network.set_title(f"Network Topology: {self.network_type}")
        else:
            ax_network.text(
                0.5, 0.5, "No topology information", ha="center", va="center"
            )
            ax_network.set_title("Network Topology")

        ax_network.axis("off")

        # 2. Parameter convergence (top right)
        ax_params = axes[0, 1]

        if self.parameter_history:
            for node, history in self.parameter_history.items():
                if history:  # Only show nodes with data
                    rounds = list(range(1, len(history) + 1))
                    ax_params.plot(rounds, history, label=f"Node {node}")

            ax_params.set_xlabel("Round")
            ax_params.set_ylabel("Parameter Norm")
            ax_params.set_title("Parameter Convergence")

            # Only add legend if we have data
            if any(len(h) > 0 for h in self.parameter_history.values()):
                ax_params.legend(loc="upper right", fontsize=8)
        else:
            ax_params.text(0.5, 0.5, "No parameter data", ha="center", va="center")
            ax_params.set_title("Parameter Convergence")

        # 3. Accuracy over time (bottom left)
        ax_acc = axes[1, 0]

        if self.accuracy_history:
            # Convert dictionary to sorted lists for plotting
            rounds = sorted(self.accuracy_history.keys())
            accuracy_values = [self.accuracy_history[r] for r in rounds]

            if rounds and accuracy_values:
                ax_acc.plot(rounds, accuracy_values, "g-o")
                ax_acc.set_xlabel("Round")
                ax_acc.set_ylabel("Accuracy")
                ax_acc.set_title("Model Accuracy")

                # Add final accuracy text
                final_acc = accuracy_values[-1]
                ax_acc.text(
                    0.5,
                    0.1,
                    f"Final accuracy: {final_acc:.4f}",
                    transform=ax_acc.transAxes,
                    ha="center",
                    bbox=dict(facecolor="white", alpha=0.8),
                )
            else:
                ax_acc.text(0.5, 0.5, "No accuracy data", ha="center", va="center")
                ax_acc.set_title("Model Accuracy")
        else:
            ax_acc.text(0.5, 0.5, "No accuracy data", ha="center", va="center")
            ax_acc.set_title("Model Accuracy")

        # 4. Loss over time (bottom right)
        ax_loss = axes[1, 1]

        if self.loss_history:
            # Convert dictionary to sorted lists for plotting
            rounds = sorted(self.loss_history.keys())
            loss_values = [self.loss_history[r] for r in rounds]

            if rounds and loss_values:
                ax_loss.plot(rounds, loss_values, "r-o")
                ax_loss.set_xlabel("Round")
                ax_loss.set_ylabel("Loss")
                ax_loss.set_title("Model Loss")

                # Add final loss text
                final_loss = loss_values[-1]
                ax_loss.text(
                    0.5,
                    0.1,
                    f"Final loss: {final_loss:.4f}",
                    transform=ax_loss.transAxes,
                    ha="center",
                    bbox=dict(facecolor="white", alpha=0.8),
                )
            else:
                ax_loss.text(0.5, 0.5, "No loss data", ha="center", va="center")
                ax_loss.set_title("Model Loss")
        else:
            ax_loss.text(0.5, 0.5, "No loss data", ha="center", va="center")
            ax_loss.set_title("Model Loss")

        # Add overall title
        plt.suptitle("Training Summary", fontsize=16)

        # Save figure - fix the rect parameter to be a tuple
        rect = (0, 0, 1, 0.97)  # left, bottom, right, top
        plt.tight_layout(rect=rect)
        plt.savefig(
            os.path.join(self.output_dir, filename), dpi=100, bbox_inches="tight"
        )
        plt.close(fig)

        print(f"Summary plot saved to {os.path.join(self.output_dir, filename)}")
