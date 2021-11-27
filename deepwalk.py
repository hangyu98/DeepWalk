from preprocess import read_node_file, build_graph
import random_walk

def main():
    # change to parse args later
    # parameters
    walk_len = 10
    num_of_iter = 10
    window_size = None # window_size for word2vec
    node_label_file_path = './data/facebook_large/musae_facebook_target.csv'
    edge_list_file_path = './data/facebook_large/musae_facebook_edges.csv'
    
    # generate random walks
    G = build_graph(edge_list_file_path)
    random_walk.sample_graph(G, walk_len=walk_len, num_of_iter=num_of_iter)
    
    # TODO: pass random walk results to word2vec model
    # TODO: perform node classification, calculate f1-score
    node_label_dict, node_name_dict = read_node_file(node_label_file_path)

if __name__ == "__main__":
    main()