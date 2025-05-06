import os
import glob
import copy
import random
import networkx as nx
import numpy as np

def compute_network_energy(G):
    """
    Compute the energy of network G based on the coevolutionary balance theory,
    using the continuous fALFF values (node attribute 'fALFF') and the edge weight
    (functional connectivity) directly.
    
    The energy is defined as:
        E = - (1/|E|) * Σ[fALFF_i * (edge weight) * fALFF_j]
    where the sum is taken over all edges.
    """
    total_product = 0.0
    edge_count = 0

    for u, v, attr in G.edges(data=True):
        try:
            falff_u = float(G.nodes[u].get('fALFF', 0))
        except ValueError:
            falff_u = 0.0
        try:
            falff_v = float(G.nodes[v].get('fALFF', 0))
        except ValueError:
            falff_v = 0.0
        try:
            weight = float(attr.get('weight', 0))
        except ValueError:
            weight = 0.0

        total_product += falff_u * weight * falff_v
        edge_count += 1

    if edge_count == 0:
        avg_product = np.nan
    else:
        avg_product = total_product / edge_count

    energy = -avg_product
    return energy
