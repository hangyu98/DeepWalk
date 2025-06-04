# Global config vars

# Deepwalk model parameters
WALK_LENGTH = 10
NUM_OF_ITERATION = 10
EMBEDDING_DIMENSION = 128
WORD2VEC_WINDOW_SIZE = 10  # WORD2VEC_WINDOW_SIZE for word2vec

# File paths
NODE_LABEL_PATH = "./data/facebook_large/musae_facebook_target.csv"
EDGE_LIST_PATH = "./data/facebook_large/musae_facebook_edges.csv"
ID2LABEL_PATH = "./data/facebook_large/labels.pkl"
ID2NAME_PATH = "./data/facebook_large/names.pkl"
WORD2VEC_PATH = "./models/word2vec.model"
EMBEDDING_PATH = "./embeddings/embeddings.pkl"

# Runtime feature flags
USE_GENSIM = True

PLOT_MEMBERS = [
    "0",
    "14305",
    "4783",
    "9884",
    "6692",
    "16486",
    "1898",
    "12117",
    "4831",
    "21386",
    "12726",
    "3758",
    "6821",
    "16343",
    "16687",
    "20150",
    "18165",
    "1365",
    "21437",
    "18873",
    "18431",
    "10989",
    "12339",
    "17810",
    "1158",
    "17108",
    "3984",
    "1657",
    "8030",
    "19760",
    "8005",
    "10780",
    "2011",
    "7874",
    "6407",
    "22107",
    "18565",
    "363",
    "8150",
    "19787",
    "10495",
    "18247",
    "12502",
    "36",
    "15703",
    "14952",
    "6563",
    "9390",
    "10616",
    "13855",
    "18755",
    "1293",
    "6542",
    "6862",
    "6990",
    "6594",
    "11802",
    "13161",
    "9842",
    "13948",
    "9494",
    "10291",
    "12193",
    "15037",
    "12967",
    "4056",
    "1539",
    "1366",
    "2713",
    "21603",
    "5608",
    "14852",
    "17937",
    "11433",
    "4239",
    "638",
    "10642",
    "4684",
    "10533",
    "4680",
]
plotting_untrained = False
