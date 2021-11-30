from preprocess import read_node_file, build_graph
import random_walk
import classification
import word2vec
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

def main():
    # change to parse args later
    # parameters
    walk_len = 10
    num_of_iter = 10
    emb_size = 128
    window_size = 10 # window_size for word2vec
    node_label_file_path = './data/facebook_large/musae_facebook_target.csv'
    edge_list_file_path = './data/facebook_large/musae_facebook_edges.csv'
    
    # generate random walks
    G = build_graph(edge_list_file_path)
    print("Graph built")
    random_walk_res = random_walk.sample_graph(G, walk_len=walk_len, num_of_iter=num_of_iter)
    print("Random walk finished")
    emb_dict = word2vec.generate_emb(G=G, random_walks=random_walk_res, window_size=window_size, emb_size=emb_size)
    print("Embeddings generated")

    # perform node classification, calculate f1-score
    node_label_dict, node_name_dict = read_node_file(node_label_file_path)
    scores = classification.classify(emb_dict, node_label_dict)
    print(scores)
    
if __name__ == "__main__":
    main()