# Visualizing the Embeddings
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import pickle
import numpy as np

def visualize():
    f = open("embeddings.pkl", 'r')
    emb_dict = pickle.load(f)
    word_vectors = np.array([emb_dict[w] for w in emb_dict])
    # reduce dimenion from 30 to 2
    # PCA is better than TSNE for large dimension like 30 :)
    pca = PCA(n_components=2)
    vectors = pca.fit_transform(word_vectors)

    # normalize the results so that we can view them more comfortably in matplotlib
    normalizer = preprocessing.Normalizer()
    norm_vectors = normalizer.fit_transform(vectors, 'l2')

    # plot the 2D normalized vectors
    x_vec = []
    y_vec = []
    for x, y in norm_vectors:
        x_vec.append(x)
        y_vec.append(y)

    f, axs = plt.subplots(1, 1, figsize=(18, 16))
    plt.scatter(x_vec, y_vec)
    # for word in word2id:
    #       plt.annotate(word, (norm_vectors[word2id[word]][0], norm_vectors[word2id[word]][1]))
    plt.show()

# visualize()