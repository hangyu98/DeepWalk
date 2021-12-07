import csv
import networkx as nx

def read_node_file(node_file_name):
    """ Read the edge list file, and node label file.
        Return a dict mapping node id to node label, and a dict mapping node id to node name.
    Args:
        edge_file_name (str): name of the file
        node_file_name (str): name of the file
    """
    node_name_dict = {}
    node_label_dict = {}
    with open(node_file_name, 'r', encoding="utf-8") as node_list:
        csv_reader = csv.reader(node_list)
        next(csv_reader) # skip file header
        for row in csv_reader:
            n_id = row[0]
            name = row[2]
            label = row[3]
            node_name_dict[n_id] = name
            node_label_dict[n_id] = label
        node_list.close()
    return node_label_dict, node_name_dict

def build_graph(edge_list_path):
    """ Build an undirected graph from edge list

    Args:
        edge_list_path (str): file name

    Returns:
        G: a networkx undirected graph
    """
    fh = open(edge_list_path, "rb")
    G = nx.read_edgelist(fh, delimiter=',')
    fh.close()
    return G
    