import sys
import os
import pickle
import keras
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.backend import argmax
import tensorflow as tf
from .model_lstm import load_model_lstm
from ..utils.dataset_utils import load_datasets_sequence
from ..utils.path_utils import *

from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

def train_model(argv, max_f1):
    language=argv[1]
    if not os.path.isfile(OUTPUT_DIR + language + "_index_lstm.pkl") or not os.path.isfile(OUTPUT_DIR + language + "_X_train_lstm.pkl"):
        index, X_train, y_train, X_test, y_test=load_datasets_sequence(language)
        pickle.dump(index, open(OUTPUT_DIR + language +   "_index_lstm.pkl", "wb"))
        pickle.dump(X_train, open(OUTPUT_DIR + language + "_X_train_lstm.pkl", "wb"))
        pickle.dump(y_train, open(OUTPUT_DIR + language + "_y_train_lstm.pkl", "wb"))
        pickle.dump(X_test, open(OUTPUT_DIR + language +  "_X_test_lstm.pkl", "wb"))
        pickle.dump(y_test, open(OUTPUT_DIR + language +  "_y_test_lstm.pkl", "wb"))
    else:
        index=pickle.load(open(OUTPUT_DIR + language +    "_index_lstm.pkl", "rb"))
        X_train=pickle.load(open(OUTPUT_DIR + language +  "_X_train_lstm.pkl", "rb"))
        y_train=pickle.load(open(OUTPUT_DIR + language +  "_y_train_lstm.pkl", "rb"))
        X_test=pickle.load(open(OUTPUT_DIR + language +   "_X_test_lstm.pkl", "rb"))
        y_test=pickle.load(open(OUTPUT_DIR + language +   "_y_test_lstm.pkl", "rb"))

    model=load_model_lstm(X_train.shape[1], len(index), 1e-3)

    opt=Adam(learning_rate=1e-3, decay=1e-2)
    model.compile(opt, loss="categorical_crossentropy")
    print (model.summary())
    history=model.fit(X_train, to_categorical(y_train), batch_size=128, epochs=20, shuffle=True)
#     history=model.fit(X_train, to_categorical(y_train), batch_size=128, epochs=1, shuffle=True)

    preds_train=model.predict(X_train)
    preds_labels=argmax(preds_train)
    train_f1=f1_score(preds_labels, y_train)
    train_prec=precision_score(preds_labels, y_train)
    train_rec=recall_score(preds_labels, y_train)
    print ("TRAIN F1[%f] PREC[%f] REC[%f]" % (train_f1, train_prec, train_rec))

    preds_test=model.predict(X_test)
    preds_labels=argmax(preds_test)
    test_f1=f1_score(preds_labels, y_test)
    test_prec=precision_score(preds_labels, y_test)
    test_rec=recall_score(preds_labels, y_test)
    print ("TEST F1[%f] PREC[%f] REC[%f]" % (test_f1, test_prec, test_rec))

    if max_f1[0] == 0 or max_f1[0] < test_f1:
        pickle.dump(model, open(OUTPUT_DIR + language + "_lstm_model.pkl", "wb"))
        pickle.dump(history, open(OUTPUT_DIR + language + "_lstm_history.pkl", "wb"))
    return train_f1, train_prec, train_rec, test_f1, test_prec, test_rec

def main(argv):
    data=[]
#     for i in range(100):
    for i in range(1):
        max_f1=[0]
        train_f1, train_prec, train_rec, test_f1, test_prec, test_rec=train_model(argv, max_f1)
        data.append((train_f1, train_prec, train_rec, test_f1, test_prec, test_rec))

    with open("replication_lstm_%s.csv" % argv[1], "w") as f:
        f.write("train_f1,train_prec,train_rec,test_f1,test_prec,test_rec\n")
        for row in data:
            f.write(",".join([str(v) for v in row]) + "\n")

if __name__ == "__main__":
    main(sys.argv)
