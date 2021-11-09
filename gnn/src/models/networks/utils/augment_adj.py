from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index
from torch_sparse import spspmm


def augment_adj(edge_index, edge_weight, num_nodes):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
    edge_index, edge_weight = sort_edge_index(edge_index, edge_weight, num_nodes)
    edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index, edge_weight, num_nodes, num_nodes, num_nodes)
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    return edge_index, edge_weight
