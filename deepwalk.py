# Standard library imports
import os

# External library imports
from networkx import Graph

# Local imports
from preprocess import get_node_mappings, build_graph
from util import save_dict, save_word2vec, load_dict
from word2vec import generate_embeddings
from random_walk import sample_graph
from classification import run_classification_pipeline
from config import (
    NODE_LABEL_PATH,
    ID2LABEL_PATH,
    ID2NAME_PATH,
    EDGE_LIST_PATH,
    WALK_LENGTH,
    NUM_OF_ITERATION,
    WORD2VEC_WINDOW_SIZE,
    EMBEDDING_DIMENSION,
    USE_GENSIM,
    WORD2VEC_PATH,
    EMBEDDING_PATH,
)


def preprocess() -> tuple[dict, dict, Graph]:
    """
    Load node label and name mappings, and build the graph.

    Returns:
        tuple:
            id2label (dict): Mapping from node id to label.
            id2name (dict): Mapping from node id to name.
            graph (Graph): NetworkX graph constructed from the edge list.
    """
    print("========== Phase 1: Preprocessing ==========")
    if os.path.exists(ID2LABEL_PATH):
        id2label = load_dict(ID2LABEL_PATH)
        id2name = load_dict(ID2NAME_PATH)
        print(f"Loaded id2label and id2name from {ID2LABEL_PATH} and {ID2NAME_PATH}.")
    else:
        id2label, id2name = get_node_mappings(NODE_LABEL_PATH)
        save_dict(ID2LABEL_PATH, id2label)
        save_dict(ID2NAME_PATH, id2name)
        print("Node label and name mappings loaded and saved.")

    graph = build_graph(edge_list_path=EDGE_LIST_PATH)
    print("Graph construction completed.")
    return id2label, id2name, graph


def run_deepwalk(graph: Graph) -> tuple[object, dict]:
    """
    Generate random walks and node embeddings using DeepWalk.

    Args:
        graph (Graph): The input NetworkX graph.

    Returns:
        tuple:
            model: Trained Word2Vec model.
            embeddings (dict): Mapping from node id to embedding vector.
    """
    print("========== Phase 2: DeepWalk ==========")
    random_walks: list[list[str]] = sample_graph(
        graph=graph, walk_length=WALK_LENGTH, num_of_iterations=NUM_OF_ITERATION
    )
    print(f"Random walks generated: {len(random_walks)} walks.")
    model, embeddings = generate_embeddings(
        graph=graph,
        random_walks=random_walks,
        word2vec_window_size=WORD2VEC_WINDOW_SIZE,
        embedding_dimension=EMBEDDING_DIMENSION,
    )
    print("Node embeddings generated.")
    if USE_GENSIM:
        save_word2vec(WORD2VEC_PATH, model)
        print(f"Gensim Word2Vec model saved to {WORD2VEC_PATH}.")
    save_dict(EMBEDDING_PATH, embeddings)
    print(f"Embeddings saved to {EMBEDDING_PATH}.")
    return model, embeddings


def run_classification(embeddings: dict, id2label: dict) -> None:
    """
    Perform node classification and print the classification report.

    Args:
        embeddings (dict): Mapping from node id to embedding vector.
        id2label (dict): Mapping from node id to label.
    """
    print("========== Phase 3: Node Classification ==========")
    scores = run_classification_pipeline(embeddings, id2label)
    print("Classification report:\n")
    print(scores)


def main():
    """
    Main entry point for the DeepWalk pipeline.
    Orchestrates preprocessing, embedding, and classification steps.
    """
    id2label, id2name, graph = preprocess()
    _, embeddings = run_deepwalk(graph)
    run_classification(embeddings, id2label)

if __name__ == "__main__":
    main()
