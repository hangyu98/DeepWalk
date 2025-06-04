# External library imports
import tensorflow as tf
import networkx as nx
from keras import Model, optimizers
from gensim.models import Word2Vec

# Local imports
from config import USE_GENSIM


class W2V(Model):

    def __init__(self, embedding_dimension: int, vocabulary_size: int) -> None:

        # Create Network Weights

        # Embedding
        self.E = tf.Variable(
            initial_value=tf.random.normal(
                shape=[vocabulary_size, embedding_dimension],
                stddev=0.1,
                dtype=tf.float32,
            )
        )

        # Weight
        self.W = tf.Variable(
            initial_value=tf.random.normal(
                shape=[embedding_dimension, vocabulary_size],
                stddev=0.1,
                dtype=tf.float32,
            )
        )

        # Bias
        self.b = tf.Variable(
            initial_value=tf.random.normal(
                shape=[vocabulary_size], stddev=0.1, dtype=tf.float32
            )
        )

    # hint: tf.nn.embedding_lookup performs a lookup in the embedding matrix
    # and returns the embeddings (or in simple terms the vector representation) of words
    def call(self, inputs):
        embedding = tf.nn.embedding_lookup(self.E, inputs)
        logits = tf.matmul(embedding, self.W) + self.b
        return logits

    def loss_func(self, logits, labels):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        )


def generate_embeddings(
    graph: nx.Graph,
    random_walks: list[list[str]],
    word2vec_window_size: int,
    embedding_dimension: int,
) -> tuple[Word2Vec | W2V, dict[str, list[float]]]:
    """
    Use word2vec with skip-gram to generate embeddings for the graph

    Args:
        graph (nx.Graph): training graph G
        random_walks (2-D list): random walks for all nodes
        word2vec_window_size (int): window size
        embedding_dimension (int): dimension of embeded vectors

    Returns:
        dict: map from node id to embedding vector
    """

    if USE_GENSIM:
        word2vec = Word2Vec(random_walks, window=word2vec_window_size, sg=1)
        embeddings: dict[str, list[float]] = {
            str(node): list(word2vec.wv[node]) for node in graph.nodes()
        }
        return word2vec, embeddings

    else:
        __batch_size, __epochs, __vocabulary_size = 512, 20, 22470

        data: list[list[int]] = []
        for sentence in random_walks:
            for word_index, word in enumerate(sentence):
                for nb_word in sentence[
                    max(word_index - int(word2vec_window_size), 0) : min(
                        word_index + int(word2vec_window_size), len(sentence)
                    )
                ]:
                    if nb_word != word:
                        data.append([int(word), int(nb_word)])

        model: W2V = W2V(
            embedding_dimension=embedding_dimension, vocabulary_size=__vocabulary_size
        )

        optimizer = optimizers.Adam(learning_rate=0.0001)

        for ep in range(__epochs):
            loss = 0
            for start, end in zip(
                range(0, len(data) - __batch_size, __batch_size),
                range(__batch_size, len(data), __batch_size),
            ):
                batch_X = [pair[0] for pair in data[start:end]]
                batch_Y = [pair[1] for pair in data[start:end]]

                with tf.GradientTape() as tape:
                    cur_loss = run_batch(model, batch_X, batch_Y)

                    loss += cur_loss
                    gradients = tape.gradient(loss, model.trainable_variables)
                    if gradients is not None:
                        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                if start % 10 == 0:
                    print("Epoch %d\t batch %d\tLoss: %.3f" % (ep, start, loss))

            embedding_matrix = model.E.read_value()
            embeddings = {
                str(node): embedding_matrix[int(node)] for node in embedding_matrix
            }
        else:
            # If the training loop does not run, still define embeddings
            embedding_matrix = model.E.read_value()
            embeddings = {
                str(node): embedding_matrix[int(node)] for node in embedding_matrix
            }

        return model, embeddings


def run_batch(model: W2V, inputs, labels):
    logits = model.call(inputs=inputs)
    loss = model.loss_func(logits, labels)
    return loss
