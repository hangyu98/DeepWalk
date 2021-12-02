from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier



def load_x(embeddings_dict):
    x = []
    for key in embeddings_dict:
        x.append(embeddings_dict[key])
    return x


def load_y(embeddings_dict, node_labels_dict):
    y = []
    for key in embeddings_dict:
        y.append(node_labels_dict[key])
    return y


def split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test


def train(x, y):
    classifier = MLPClassifier()
    classifier.fit(x, y)
    return classifier


def test(classifier, x, y_true):
    y_pred = classifier.predict(x)
    # return classification_report(y_true, y_pred, labels=['politician', 'company', 'government', 'tvshow'])
    return classification_report(y_true, y_pred, target_names=['politician', 'company', 'government', 'tvshow'])


def classify(embeddings_dict, node_labels_dict):
    x = load_x(embeddings_dict)
    y = load_y(embeddings_dict, node_labels_dict)
    print("data loaded")
    x_train, x_test, y_train, y_test = split(x, y)
    print("split finished")
    classifier = train(x_train, y_train)
    print("training finished")
    scores = test(classifier, x_test, y_test)
    return scores
