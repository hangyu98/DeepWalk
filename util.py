import pickle
from typing import Any
from gensim.models import Word2Vec

def save_dict(file_path: str, data: dict) -> None:
    """
    Save a dictionary to a file using pickle in binary mode.

    Args:
        file_path (str): Path to the file where the dictionary will be saved.
        data (dict): Dictionary object to serialize and save.
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_dict(file_path: str) -> dict:
    """
    Load a dictionary from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        dict: The loaded dictionary object.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_word2vec(model_path: str, model: Word2Vec) -> None:
    """
    Save a gensim Word2Vec model to disk.

    Args:
        model_path (str): Path to save the model.
        model (Word2Vec): Trained Word2Vec model to save.
    """
    model.save(model_path)

def load_word2vec(model_path: str) -> Word2Vec:
    """
    Load a gensim Word2Vec model from disk.

    Args:
        model_path (str): Path to the saved Word2Vec model.

    Returns:
        Word2Vec: Loaded Word2Vec model.
    """
    return Word2Vec.load(model_path)