import time
import pickle
from .nlp_utils import *
from .path_utils import DATA_DIR

def load_datasets(encode_func, language):
    dataset_train_path= DATA_DIR + language + "_data_train.pkl"
    dataset_test_path= DATA_DIR + language + "_data_test.pkl"

    startTime=time.time()
    print ("Extrating training data ..")
    dataset_train=pickle.load(open(dataset_train_path, "rb"))
    train_msgs=[v[1] for v in dataset_train]
    y_train=[int(v[0]) for v in dataset_train]
    print ("Done. Extracting training data [%s sec(s)]" % (time.time()-startTime))

    startTime=time.time()
    print ("Compiling index ...")
    higher_order_ngrams = encode_func != encode_messages_sequence
    index, word_counts=extract_index(train_msgs, higher_order_ngrams)
    index=clean_index(index, word_counts)
    index=rebuild_index(index)
    print ("Done. Compiling index [%s sec(s)]" % (time.time()-startTime))

    startTime=time.time()
    print ("Extracting test data ...")
    dataset_test=pickle.load(open(dataset_test_path, "rb"))
    test_msgs=[v[1] for v in dataset_test]
    y_test=[int(v[0]) for v in dataset_test]
    print ("Done. Extracting test data [%s sec(s)]" % (time.time()-startTime))

    startTime=time.time()
    print ("Encoding data ... [%s]" % encode_func.__name__)
    X_train=encode_func(index, train_msgs)
    X_test=encode_func(index, test_msgs)
    print ("Done. Encoding data [%s sec(s)]" % (time.time()-startTime))

    return index, X_train, y_train, X_test, y_test

def load_datasets_binary(language):
    return load_datasets(encode_messages_binary, language)

def load_datasets_sequence(language):
    return load_datasets(encode_messages_sequence, language)
