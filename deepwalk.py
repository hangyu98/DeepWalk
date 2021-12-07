from preprocess import read_node_file, build_graph
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from util import save_dict, save_word2vec
# from visualize import visualize
from word2vec import generate_emb
from random_walk import sample_graph
from classification import classify
from config import *
    
    
def main():
    print("----------------Phase 1: preprocess----------------")
    # load data
    id2label, id2name = read_node_file(node_label_file_path)
    save_dict(id2label_path, id2label)
    save_dict(id2name_path, id2name)
    print("Data loaded")
    # build graph
    G = build_graph(edge_list_file_path)
    print("Graph built")
    print("----------------Phase 2: deepwalk------------------")
    # generate random walks
    random_walk_res = sample_graph(G, walk_len=walk_len, num_of_iter=num_of_iter)
    print("Random walk finished")
    # use word2vec for embedding
    print(random_walk_res[0])
    model, embeddings = generate_emb(G=G, random_walks=random_walk_res, window_size=window_size, emb_size=emb_size)
    print("Embeddings generated")
    save_word2vec(word2vec_model_path, model)
    save_dict(embedding_path, embeddings)
    print("Results saved")
    print("--------------Phase 3: classification--------------")
    # perform node classification, calculate f1-score
    scores = classify(embeddings, id2label)
    print(scores)

    # optional step
    # visualize(word2vec_model_path, embedding_path)
    # interest_lst = ['0', '8005', '1293', '1366']
    # visualize(interest_lst=interest_lst, topk=20)
    
if __name__ == "__main__":
    main()