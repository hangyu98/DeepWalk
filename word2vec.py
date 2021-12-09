from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import random
from config import *

class W2V(Model):
  # TODO: Create Network Weights
  def __init__(self, EMBEDDING_SZ, VOCAB_SZ):
    super(W2V, self).__init__()
    self.EMBEDDING_SZ = 128
    self.VOCAB_SZ = VOCAB_SZ
    self.E = tf.Variable(tf.random.normal([self.VOCAB_SZ,self.EMBEDDING_SZ], stddev=.1, dtype=tf.float32))
    self.W = tf.Variable(tf.random.normal([self.EMBEDDING_SZ,self.VOCAB_SZ], stddev=.1, dtype=tf.float32))
    self.b = tf.Variable(tf.random.normal([self.VOCAB_SZ], stddev=.1, dtype=tf.float32))

  # TODO: Build Inference Pipeline
  # hint: tf.nn.embedding_lookup performs a lookup in the embedding matrix and returns the embeddings (or in simple terms the vector representation) of words
  def call(self, inputs):
    embedding = tf.nn.embedding_lookup(self.E, inputs, max_norm=None, name=None)
    logits = tf.matmul(embedding,self.W)+self.b
    return logits

  # Build Loss 
  def loss_func(self, logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels,logits))

def generate_emb(G, random_walks, window_size, emb_size):
    """ Use word2vec with skip-gram for embedding

    Args:
        G (nx.Graph): training graph G
        random_walks (2-d list): random walks for all nodes
        window_size (int): window size
        emb_size (int): dimension of embeded vectors

    Returns:
        dict: map from id to embedding vector
    """
    
    if usingGensim == True:
      print("---using gensim library")
      word2vec = Word2Vec(random_walks,window=window_size, sg=1)
      embeddings = {}
      for n in G.nodes():
        embeddings[n] = word2vec.wv[n]
      return word2vec, embeddings

    else:
      window_size = 3
      data = []
      for sentence in random_walks:
        for word_index, word in enumerate(sentence):
          for nb_word in sentence[max(word_index-int(window_size),0):min(word_index+int(window_size),len(sentence))]:
            if nb_word != word:
              data.append([int(word), int(nb_word)])

      print("---finished splitting data---")   
      BSZ, EPOCHS = 512, 20
      data = np.array(data)
      print(data.shape)            
      vocab_size = 22500

      model = W2V(emb_size,vocab_size)
      optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
      
      print("---training starts---")
      for ep in range(EPOCHS):
        print(EPOCHS)
        curr_loss = 0
        step = 0
        for start, end in zip(range(0, len(data) - BSZ, BSZ), range(BSZ, len(data), BSZ)):
          batch_X = data[start:end, 0]
          batch_Y = data[start:end, 1]
          with tf.GradientTape() as tape:
            loss = run_batch(model,batch_X,batch_Y)
          curr_loss += loss
          step += 1
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))

          if start % 10 == 0:
            print('Epoch %d\t batch %d\tLoss: %.3f' % (ep, start, loss))

        embeddingsMatrix = model.E.read_value()
        embeddings = {}
      for n in G.nodes():
        embeddings[n] = embeddingsMatrix[int(n)]

      return model, embeddings


    

def run_batch(model, inputs, labels):
  logits = model(inputs)
  loss = model.loss_func(logits, labels)
  return loss
