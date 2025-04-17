import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import os
import networkx as nx
from NetworkEnergyComputer import compute_network_energy

def extract_graph_features(fALFF, edges, top_k=5):
    # Convert input lists to NumPy arrays
    fALFF = np.asarray(fALFF)
    edges = np.asarray(edges)

    # ---- node stats (fALFF) ----
    # Compute statistical features from the fALFF node values
    node_mean = np.mean(fALFF)
    node_std = np.std(fALFF)
    node_min = np.min(fALFF)
    node_max = np.max(fALFF)
    node_median = np.median(fALFF)
    node_skew = skew(fALFF)
    node_kurt = kurtosis(fALFF)

    # ---- edge stats ----
    # Extract edge weights by removing diagonal (self-loops)
    edge_weights = edges[~np.eye(edges.shape[0], dtype=bool)]
    # Compute statistical features from the edge weights
    edge_mean = np.mean(edge_weights)
    edge_std = np.std(edge_weights)
    edge_min = np.min(edge_weights)
    edge_max = np.max(edge_weights)
    edge_median = np.median(edge_weights)
    edge_skew = skew(edge_weights)
    edge_kurt = kurtosis(edge_weights)

    # Get top-k highest edge weights
    topk_edges = np.sort(edge_weights)[-top_k:]
    topk_dict = {f"edge_top_{i + 1}": val for i, val in enumerate(topk_edges[::-1])}

    # ---- graph-level features ----
    # Create a NetworkX graph from the adjacency matrix
    G = nx.from_numpy_array(edges)

    # Graph density: how densely the graph is connected
    graph_density = nx.density(G)

    # Graph diameter: Longest shortest path between any two nodes
    try:
        graph_diameter = nx.diameter(G)
    except nx.NetworkXError:
        graph_diameter = np.nan

    # Average path length: Average length of the shortest paths between all pairs of nodes
    try:
        avg_path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        avg_path_length = np.nan

    # Number of connected components
    num_components = nx.number_connected_components(G)

    # Modularity
    try:
        modularity = nx.community.modularity(G, list(nx.community.label_propagation_communities(G)))
    except Exception:
        modularity = np.nan

    # Network energy value
    energy_value = compute_network_energy(G)

    # ---- combine all features ----
    # Combine all extracted features into a dictionary
    features = {
        "node_mean": node_mean,
        "node_std": node_std,
        "node_min": node_min,
        "node_max": node_max,
        "node_median": node_median,
        "node_skew": node_skew,
        "node_kurt": node_kurt,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "edge_min": edge_min,
        "edge_max": edge_max,
        "edge_median": edge_median,
        "edge_skew": edge_skew,
        "edge_kurt": edge_kurt,
        "graph_density": graph_density,
        "graph_diameter": graph_diameter,
        "avg_path_length": avg_path_length,
        "num_components": num_components,
        "modularity": modularity,
        "energy_value": energy_value,
    }

    features.update(topk_dict)

    return features


def load_graphs_from_folder(folder_path, label):
    # Load GraphML files from a folder and convert to (fALFF, edges, label) format
    graphs = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".graphml"):
            path = os.path.join(folder_path, filename)
            graph_data = nx.read_graphml(path)

            # Initialize fALFF values (assuming 200 nodes)
            fALFF = np.zeros(200)
            for i in graph_data.nodes():
                fALFF[int(i) - 1] = float(graph_data.nodes[i]['fALFF'])

            # Initialize edge weights (adjacency matrix)
            edges = np.zeros((200, 200))
            for u, v, data in graph_data.edges(data=True):
                i = int(u) - 1
                j = int(v) - 1
                weight = float(data['weight'])
                edges[i, j] = weight
                edges[j, i] = weight

            # Append the data as a tuple
            graphs.append((fALFF, edges, label))
    return graphs
