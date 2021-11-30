from gensim.models import Word2Vec

def generate_emb(G, random_walks, window_size, emb_size):
    word2vec = Word2Vec(random_walks,
                        window=window_size, sg=1)
    embeddings = {}
    for n in G.nodes():
        embeddings[n] = word2vec.wv[n]
    
    return embeddings
