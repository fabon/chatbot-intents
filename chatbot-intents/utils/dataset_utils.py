import time
import pickle
import numpy as np
from .nlp_utils import *
from .path_utils import DATA_DIR

def import_training_sets(language):
    training_sets=pickle.load(open(DATA_DIR + "/training_inds_%s.pkl" % language, "rb"))
    return training_sets

def import_clean_labels(path):
    labels=[]
    with open(path, "r") as f:
        for line in f:
            label= int(line[:-1]) if line[0] != "-" else -1
            labels.append(label)
    print (len(labels))
    return labels

def load_datasets(encode_func, language, training_inds):
    dataset_train_path= DATA_DIR + language + "_data_train.pkl"
    dataset_test_path= DATA_DIR + language + "_data_test.pkl"

#     dataset_train_path= DATA_DIR + language + "_data_train_cleaned.pkl"
#     dataset_test_path= DATA_DIR + language + "_data_test_cleaned.pkl"

    startTime=time.time()
    print ("Extracting training data ..")
    dataset_train=pickle.load(open(dataset_train_path, "rb"))
    train_msgs=[v[1] for v in dataset_train]
    y_train=[int(v[0]) for v in dataset_train]

    if training_inds is not None:
        train_msgs=(np.array(train_msgs)[training_inds]).tolist()
        y_train=(np.array(y_train)[training_inds]).tolist()
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
    print (len(index))
    X_train=encode_func(index, train_msgs)
    X_test=encode_func(index, test_msgs)
    print ("Done. Encoding data [%s sec(s)]" % (time.time()-startTime))

    return index, word_counts, X_train, y_train, X_test, y_test

def load_datasets_binary(language, training_inds=None):
    return load_datasets(encode_messages_binary, language, training_inds)

def load_datasets_sequence(language, training_inds=None):
    return load_datasets(encode_messages_sequence, language, training_inds)
