# Standard library imports
import csv

# External library imports
import networkx as nx
from networkx import Graph


def get_node_mappings(node_file_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Read the edge list file, and node label file.
    Return a dict mapping node id to node label, and a dict mapping node id to node name.

    Args:
        node_file_path (str): name of the file

    Returns:
        tuple[dict[str, str], dict[str, str]]: mappings from node_id to labels, and node_id to names
    """

    NODE_ID_IDX = 0
    NODE_NAME_IDX = 2
    NODE_LABEL_IDX = 3

    with open(file=node_file_path, mode="r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip file header

        node_label_dict: dict[str, str] = {}
        node_name_dict: dict[str, str] = {}

        # Read each row and populate the dictionaries
        for row in csv_reader:
            node_label_dict[row[NODE_ID_IDX]] = row[NODE_LABEL_IDX]
            node_name_dict[row[NODE_ID_IDX]] = row[NODE_NAME_IDX]

        return node_label_dict, node_name_dict


def build_graph(edge_list_path: str) -> Graph:
    """Build an undirected graph from edge list

    Args:
        edge_list_path (str): file name

    Returns:
        A networkx undirected graph
    """
    with open(file=edge_list_path, mode="rb") as f:
        return nx.read_edgelist(path=f, delimiter=",")
