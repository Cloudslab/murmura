from murmura.orchestration.cluster_manager import ClusterManager
import networkx as nx  # type: ignore
import matplotlib.pyplot as plt


def visualize_network_topology(cluster_manager: ClusterManager):
    """Visualize the client network topology using NetworkX"""
    if not cluster_manager.topology_manager:
        raise ValueError("Topology not initialized")

    graph = nx.Graph()
    adjacency = cluster_manager.topology_manager.adjacency_list

    # Add nodes
    for node in adjacency.keys():
        graph.add_node(f"Client {node}")

    # Add edges
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            graph.add_edge(f"Client {node}", f"Client {neighbor}")

    # Draw visualization
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="skyblue",
        font_size=6,
        font_weight="bold",
        edge_color="gray",
    )
    plt.title("Client Network Topology")
    plt.show()
