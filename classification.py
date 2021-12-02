from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def load_data(embeddings, id2label):
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

def split(X, y):
    """ split the data into training and test sets

    Args:
        X (list)
        y (list)
        
    Returns:
        train and test splits of X and y
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test

def train(X, y):
    """ train a MLP classifier

    Args:
        X (list): features
        y (list): labels
        
    Returns:
        [classifier]: trained classifier
    """
    classifier = MLPClassifier()
    classifier.fit(X, y)
    return classifier

def test(classifier, X, y_true):
    """ test the performance of the trained classifier

    Args:
        classifier: ...
        X (list): test X
        y_true (list): test label

    Returns:
        classification report containing accuracy, f1-score, etc.
    """
    y_pred = classifier.predict(X)
    return classification_report(y_true, y_pred, target_names=['politician', 'company', 'government', 'tvshow'])

def classify(embeddings, id2label):
    X, y = load_data(embeddings, id2label)
    print("data loaded")
    x_train, x_test, y_train, y_test = split(X, y)
    print("split finished")
    classifier = train(x_train, y_train)
    print("training finished")
    scores = test(classifier, x_test, y_test)
    return scores
