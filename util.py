import pickle
from gensim.models import Word2Vec

def save_dict(filename, dict_to_save):
    # create a binary pickle file 
    f = open(filename,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(dict_to_save, f)
    # close file
    f.close()
    
def save_word2vec(path, model):
    model.save(path)

def load_word2vec(path):
    model = Word2Vec.load(path)
    return model