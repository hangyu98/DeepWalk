from gensim.models import Word2Vec

def generate_emb(G, random_walks, window_size, emb_size):
    """ Use word2vec with skip-gram for embedding

    Args:
        G (nx.Graph): training graph G
        random_walks (2-d list): random walks for all nodes
        window_size (int): window size
        emb_size (int): dimension of embeded vectors

    Returns:
        dict: map from id to embedding vector
    """
    word2vec = Word2Vec(random_walks,
                        window=window_size, sg=1)
    
    embeddings = {}
    for n in G.nodes():
        embeddings[n] = word2vec.wv[n]

    return embeddings
