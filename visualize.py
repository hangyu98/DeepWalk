# Visualizing the Embeddings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
from util import load_word2vec
from config import *


def visualize(interest_lst, topk):
    # load model and embeddings
    
    embeddings = pickle.load(open(embedding_path, 'rb'))
    id2name = pickle.load(open(id2name_path, 'rb'))
    
    # extract top k similar words for the list of interest
    wanted = []
    for id in interest_lst:
        vec = embeddings[id]
        if usingGensim == True:
            model = load_word2vec(word2vec_model_path)
            most_similar_words = model.wv.most_similar( [ vec ], [], topk)
        else:
            most_similar_words = find_closest(id, embeddings, topk)
        for id2sim in most_similar_words:
            wanted.append(id2sim[0])
            
    print(wanted)
    
    word_vectors = np.array([embeddings[w] for w in embeddings if w in wanted])
    word_names = np.array([w for w in embeddings if w in wanted])
    
    tsne = TSNE(n_components=2)
    vectors = tsne.fit_transform(word_vectors)

    # plot the 2D unormalized vectors
    x_vec = []
    y_vec = []
    names = []
    i = 0
    for x, y in vectors:
        x_vec.append(x)
        y_vec.append(y)
        names.append(id2name[word_names[i]])
        i = i + 1

    # combine data to be plotted
    df = {}
    df['x'] = x_vec
    df['y'] = y_vec
    df['name'] = names
    
    # plot
    sns.scatterplot(
        x="x", y="y",
        palette=sns.color_palette("hls", 16),
        data=df,
        legend="full",
        alpha=0.3,
    )
    
    # add text labels
    for line in range(0, len(df['name'])):
        plt.text(df['x'][line]+0.2, df['y'][line], df['name'][line], horizontalalignment='left', size=6, color='black', weight='semibold')
    
    plt.show()
    
def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

# Return the id of the word with smallest euclidean distance to the queried word(index)
def find_closest(id, vectors, topk):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[id]
    minVectors = []
    distList = []
    for nodes in vectors:
        dist = euclidean_dist(vector, query_vector)
        vector = embeddings[node]
        if len(dist)<10:
            minVector.appen(vector)
            distList.append(dist)
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return minVectors

 # the voice of china (tvshow), facebook (company), Bernie Sanders (politician), UK government (government)
interest_lst = ['0', '8005', '1293', '1366']
visualize(interest_lst=interest_lst, topk=20)
