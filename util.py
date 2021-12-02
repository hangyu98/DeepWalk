import pickle

def save_dict(filename, dict_to_save):
    # create a binary pickle file 
    f = open(filename,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(dict_to_save, f)
    # close file
    f.close()