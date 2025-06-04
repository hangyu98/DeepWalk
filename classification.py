from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from typing import Any, Dict, Tuple

def _load_data(embeddings: dict[Any, Any], id2label: dict[Any, Any]) -> tuple[list[Any], list[Any]]:
    """ load training labels

    Args:
        embeddings (dict): id to embedding
        id2label (dict): [description]

    Returns:
        [type]: [description]
    """
    X = []
    y = []
    for key in embeddings:
        X.append(embeddings[key])
        y.append(id2label[key])
    return X, y

def _split(X: list[Any], y: list[Any]) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """
    Split the data into training and test sets.

    Args:
        X (list): Feature vectors.
        y (list): Corresponding labels.

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test

def _train_classifier(X: list[Any], y: list[Any]) -> MLPClassifier:
    """
    Train an MLP classifier on the provided data.

    Args:
        X (list): Feature vectors for training.
        y (list): Labels for training.

    Returns:
        MLPClassifier: Trained classifier.
    """
    classifier = MLPClassifier()
    classifier.fit(X, y) # type: ignore
    return classifier

def _evaluate_classifier(classifier: MLPClassifier, X: list[Any], y_true: list[Any]) -> str:
    """
    Evaluate the trained classifier and return a classification report.

    Args:
        classifier (MLPClassifier): Trained classifier.
        X (list): Test feature vectors.
        y_true (list): True labels for the test set.

    Returns:
        str: Classification report containing accuracy, f1-score, etc.
    """
    y_pred = classifier.predict(X)
    return str(classification_report(y_true, y_pred, target_names=['politician', 'company', 'government', 'tvshow']))

# Public API: run_classification_pipeline
def run_classification_pipeline(embeddings: dict[str, list[float]], id2label: dict[str, str]) -> str:
    """
    Run the full classification pipeline: split data, train, and evaluate.

    Args:
        embeddings (dict): Mapping from node id to embedding vector.
        id2label (dict): Mapping from node id to label.

    Returns:
        str: Classification report.
    """
    X, y = _load_data(embeddings, id2label)
    print("data loaded")
    x_train, x_test, y_train, y_test = _split(X, y)
    print("split finished")
    classifier = _train_classifier(x_train, y_train)
    print("training finished")
    scores = _evaluate_classifier(classifier, x_test, y_test)
    return scores
