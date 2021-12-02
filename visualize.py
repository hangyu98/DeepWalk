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
    model = load_word2vec(word2vec_model_path)
    embeddings = pickle.load(open(embedding_path, 'rb'))
    id2name = pickle.load(open(id2name_path, 'rb'))
    
    # extract top k similar words for the list of interest
    wanted = []
    for id in interest_lst:
        vec = embeddings[id]
        most_similar_words = model.wv.most_similar( [ vec ], [], topk)
        for id2sim in most_similar_words:
            wanted.append(id2sim[0])
    
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
        plt.text(df['x'][line]+0.2, df['y'][line], df['name'][line], horizontalalignment='left', size=5, color='black', weight='semibold')
    
    plt.show()

 # the voice of china (tvshow), facebook (company), Bernie Sanders (politician), UK government (government)
interest_lst = ['0', '8005', '1293', '1366']
visualize(interest_lst=interest_lst, topk=20)