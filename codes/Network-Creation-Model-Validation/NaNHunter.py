import os
import math
import networkx as nx
import numpy as np

GRAPH_DIR = ''

def is_invalid(w):
    """Return True if w is None or NaN."""
    return (w is None) or (isinstance(w, float) and math.isnan(w))

def interpolate_edge_weight(G, u, v):
    """
    Collect all valid weights on edges incident
    to u or v (excluding edge (u,v)), then return their mean.
    If no valid neighbors, return 0.0.
    """
    weights = []
    for nbr in G.neighbors(u):
        if nbr == v: 
            continue
        w = G[u][nbr].get('weight', None)
        if not is_invalid(w):
            weights.append(w)
    for nbr in G.neighbors(v):
        if nbr == u:
            continue
        w = G[v][nbr].get('weight', None)
        if not is_invalid(w):
            weights.append(w)
    if weights:
        return float(np.mean(weights))
    else:
        return 0.0

def fix_graph(G):
    """
    For every edge with invalid weight, interpolate
    from its neighbors and overwrite in-place.
    """
    # TODO: we need a list of edges to process, because we'll be mutating weights
    to_fix = [(u, v) for u, v, data in G.edges(data=True)
              if is_invalid(data.get('weight', None))]
    for u, v in to_fix:
        new_w = interpolate_edge_weight(G, u, v)
        G[u][v]['weight'] = new_w

if __name__ == "__main__":
    for fname in os.listdir(GRAPH_DIR):
        if not fname.lower().endswith(".graphml"):
            continue
        path = os.path.join(GRAPH_DIR, fname)
        try:
            G = nx.read_graphml(path)
        except Exception as e:
            print(f"[ERROR] could not read {fname}: {e}")
            continue

        fix_graph(G)

        try:
            nx.write_graphml(G, path)
            print(f"[OK] fixed and saved: {fname}")
        except Exception as e:
            print(f"[ERROR] could not write {fname}: {e}")
