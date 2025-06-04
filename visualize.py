# Standard library imports
from json import load

# External library imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

# Local application imports
from util import load_dict, load_word2vec
from config import EMBEDDING_PATH, ID2NAME_PATH, WORD2VEC_PATH, USE_GENSIM, plotting_untrained, PLOT_MEMBERS


def plot_node_embeddings(interest_lst: list[str], topk: int) -> None:
    """
    Visualize node embeddings using t-SNE and matplotlib.

    Args:
        interest_lst (list[str]): List of node ids to use as query nodes.
        topk (int): Number of most similar nodes to retrieve for each query node.

    This function loads embeddings and node names, finds the top-k most similar nodes
    for each node in interest_lst, reduces their embeddings to 2D using t-SNE, and
    plots the result with labels.
    """
    # load model and embeddings
    embeddings: dict[str, np.ndarray] = load_dict(EMBEDDING_PATH)
    id2name: dict[str, str] = load_dict(ID2NAME_PATH)

    selected_nodes: list[str] = []
    if USE_GENSIM:
        model = load_word2vec(WORD2VEC_PATH)
        for node_id in interest_lst:
            # Use node_id directly as a string key for most_similar
            if node_id in model.wv:
                most_similar_keys = model.wv.most_similar(node_id, topn=topk)
                for key, _ in most_similar_keys:
                    selected_nodes.append(key)
    else:
        for node_id in interest_lst:
            most_similar = find_closest(node_id, embeddings, topk)
            selected_nodes.extend(most_similar)

    if plotting_untrained:
        selected_nodes = PLOT_MEMBERS

    word_vectors = np.array([embeddings[w] for w in embeddings if w in selected_nodes])
    word_names = np.array([w for w in embeddings if w in selected_nodes])

    tsne = TSNE(n_components=2)
    vectors = tsne.fit_transform(word_vectors)

    # plot the 2D unormalized vectors
    x_coords: list[float] = []
    y_coords: list[float] = []
    names: list[str] = []

    for i, (x, y) in enumerate(vectors):
        x_coords.append(x)
        y_coords.append(y)
        # Use .get to avoid KeyError
        names.append(str(id2name.get(word_names[i], word_names[i])))

    # combine data to be plotted
    plot_data = {
        "x": x_coords,
        "y": y_coords,
        "name": names,
    }

    # plot
    sns.scatterplot(
        x="x",
        y="y",
        # Remove palette since no hue is set
        data=plot_data,
        legend="full",
        alpha=0.3,
    )

    # add text labels
    for i, name in enumerate(plot_data["name"]):
        plt.text(
            plot_data["x"][i] + 0.2,
            plot_data["y"][i],
            name,
            horizontalalignment="left",
            size=6,
            color="black",
            weight="semibold",
        )

    # To reduce font warnings for non-Latin glyphs, try to use a font that supports them
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "SimHei", "Noto Sans CJK SC"]

    plt.show()


def euclidean_dist(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Euclidean distance.
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def find_closest(node_id: str, embeddings: dict[str, np.ndarray], topk: int) -> list[str]:
    """
    Return the ids of the topk closest nodes by Euclidean distance.

    Args:
        node_id (str): The node id to query.
        embeddings (dict[str, np.ndarray]): Mapping from node id to embedding vector.
        topk (int): Number of closest nodes to return.

    Returns:
        list[str]: List of node ids of the topk closest nodes.
    """
    # Use a constant for the minimum number of candidates to start replacement
    MIN_CANDIDATES = topk
    min_vectors: list[np.ndarray] = []
    dist_list: list[float] = []
    node_list: list[str] = []
    query_vector = embeddings[node_id]
    for candidate_id, vector in embeddings.items():
        dist = euclidean_dist(vector, query_vector)
        if len(dist_list) < topk:
            min_vectors.append(vector)
            dist_list.append(dist)
            node_list.append(candidate_id)
        if len(dist_list) >= MIN_CANDIDATES and dist < max(dist_list):
            index = dist_list.index(max(dist_list))
            dist_list[index] = dist
            min_vectors[index] = vector
            node_list[index] = candidate_id
    return node_list


def main():
    """
    Main entry point for visualization. Visualizes the top-k most similar nodes for a set of interest nodes.
    """
    # the voice of china (tvshow), facebook (company), Bernie Sanders (politician), UK government (government)
    interest_lst = ["0", "8005", "1293", "1366"]
    plot_node_embeddings(interest_lst=interest_lst, topk=20)

if __name__ == "__main__":
    main()
