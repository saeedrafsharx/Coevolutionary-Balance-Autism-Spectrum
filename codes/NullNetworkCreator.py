import os
import glob
import copy
import random
import networkx as nx
import numpy as np
def create_null_network(G):
    """
    Create a null network by copying G and shuffling the node 'fALFF' values.
    The network structure (nodes, edges, edge weights) is preserved.
    """
    G_null = G.copy()
    # Extract current fALFF values from all nodes
    falff_values = [G_null.nodes[n].get('fALFF', 0) for n in G_null.nodes()]
    # Shuffle the fALFF values
    random.shuffle(falff_values)
    # Assign the shuffled values back to the nodes
    for i, n in enumerate(G_null.nodes()):
        G_null.nodes[n]['fALFF'] = falff_values[i]
    return G_null
