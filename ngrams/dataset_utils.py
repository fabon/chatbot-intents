import pickle
from nlp_utils import *

DATASET_TRAIN_PATH="./english_data_train.pkl"
DATASET_TEST_PATH="./english_data_test.pkl"
NGRAMS_ORDER=[1,2,3]
STOPWORDS=[]

def load_datasets():
    dataset_train=pickle.load(open(DATASET_TRAIN_PATH, "rb"))
    train_msgs=[v[1] for v in dataset_train]
    y_train=[int(v[0]) for v in dataset_train]

    index, word_counts=extract_index(train_msgs, NGRAMS_ORDER, STOPWORDS)
    index=clean_index(index, word_counts)
    index=rebuild_index(index)

    dataset_test=pickle.load(open(DATASET_TEST_PATH, "rb"))
    test_msgs=[v[1] for v in dataset_test]
    y_test=[v[0] for v in dataset_test]

    X_train=encode_messages_binary(index, train_msgs, NGRAMS_ORDER, STOPWORDS)
    X_test=encode_messages_binary(index, test_msgs, NGRAMS_ORDER, STOPWORDS)

    return index, X_train, y_train, X_test, y_test

