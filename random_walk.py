# Standard library imports
import random
from typing import Iterable

# External library imports
from networkx import Graph


def random_walk(
    graph: Graph, node_id: str, walk_length: int, num_of_iterations: int
) -> list[list[str]]:
    """
    Generate a number of fixed-len random walks from a certain node

    Args:
        graph (nx.Graph): undirected graph G
        node_id (str): node_id
        walk_length (int): length of each random_walk
        num_of_iterations (int): repeat the random_walk process for how many times

    Returns:
        walks: 2-D walks of shape [num_of_iterations, walk_length]
    """

    walk_path: list[list[str]] = []

    # repeat num_of_iterations times
    for _ in range(num_of_iterations):
        cur_node: str = node_id
        cur_walk_path: list[str] = [node_id]

        # random walk from cur_node for walk_length steps
        for _ in range(walk_length - 1):

            # get all neighbors of cur_node
            possible_next: list[str] = [
                neighbor for neighbor in graph.neighbors(n=cur_node)
            ]

            # randomly choose one node from all neighbors as the next node to visit
            next_node: str = random.choice(possible_next)
            cur_walk_path.append(next_node)

            cur_node = next_node

        walk_path.append(cur_walk_path)

    return walk_path


def sample_graph(
    graph: Graph, walk_length: int, num_of_iterations: int
) -> list[list[str]]:
    """Perform random-walk on all nodes in the graph

    Args:
        graph (nx.Graph): undirected graph
        walk_length (int): length of each random_walk
        num_of_iterations (int): repeat the random_walk process for how many times

    Returns:
        random walk results for the input graph of shape: [num_of_iterations * num_of_nodes, walk_length]
    """
    all_walk_paths: list[list[str]] = []

    for n in graph.nodes():

        walk_path: list[list[str]] = random_walk(
            graph=graph,
            node_id=n,
            walk_length=walk_length,
            num_of_iterations=num_of_iterations,
        )

        all_walk_paths = all_walk_paths + walk_path

    return all_walk_paths
