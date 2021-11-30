import random

def random_walk(G, node_id, walk_len, num_of_iter):
    """ Generate a number of fixed-len random walks from a certain node

    Args:
        G (nx.Graph): undirected graph G
        node_id (str): node_id
        walk_len (int): length of each random_walk
        num_of_iter (int): repeat the random_walk process for how many times

    Returns:
        walks: 2-d walks of shape [num_of_iter, walk_len]
    """
    walks = []
    # repeat num_of_iter times
    for _ in range(num_of_iter):
        cur_node = node_id
        cur_walk = [node_id]
        # sample (walk_len - 1) times
        for _ in range(walk_len - 1):
            # get all neighbors of cur_node
            all_neighbors = G.neighbors(cur_node)
            possible_next = []
            for nei in all_neighbors:
                possible_next.append(nei)
            # randomly choose one node from all neighbors as next node
            next_node = random.choice(possible_next)
            cur_walk.append(next_node)
            cur_node = next_node
        walks.append(cur_walk)
    # print(all_walks)
    return walks

def sample_graph(G, walk_len, num_of_iter):
    """ Perform random-walk on all nodes in the graph

    Args:
        G (nx.Graph): undirected graph G
        walk_len (int): length of each random_walk
        num_of_iter (int): repeat the random_walk process for how many times

    Returns:
        all_walks: all random_walks for graph G. shape: [num_of_iter * num_of_nodes, walk_len]
    """
    all_walks = []
    for n in G.nodes():
        n_walk = random_walk(G, n, walk_len=walk_len, num_of_iter=num_of_iter)
        all_walks = all_walks + n_walk # concat two lists
    print(len(all_walks))
    print(len(all_walks[0]))
    return all_walks
    