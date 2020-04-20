import sys
import os
import pickle
import keras
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.backend import argmax
import tensorflow as tf
from model_ngrams import load_model_ngrams
from dataset_utils import load_datasets

from sklearn.metrics import f1_score, precision_score, recall_score

def main():
    if not os.path.isfile("index.pkl"):
        index, X_train, y_train, X_test, y_test=load_datasets()
        pickle.dump(index, open("index.pkl", "wb"))
        pickle.dump(X_train, open("X_train.pkl", "wb"))
        pickle.dump(y_train, open("y_train.pkl", "wb"))
        pickle.dump(X_test, open("X_test.pkl", "wb"))
        pickle.dump(y_test, open("y_test.pkl", "wb"))
    else:
        index=pickle.load(open("index.pkl", "rb"))
        X_train=pickle.load(open("X_train.pkl", "rb"))
        y_train=pickle.load(open("y_train.pkl", "rb"))
        X_test=pickle.load(open("X_test.pkl", "rb"))
        y_test=pickle.load(open("y_test.pkl", "rb"))

    model=load_model_ngrams(X_train.shape[1], 1e-4)

    opt=SGD(learning_rate=1.0, decay=1e-2)
    model.compile(opt, loss="categorical_crossentropy")
    history=model.fit(X_train, to_categorical(y_train), batch_size=32, epochs=20, shuffle=True)

    pickle.dump(model, open("ngrams_model.pkl", "wb"))
    pickle.dump(history, open("ngrams_history.pkl", "wb"))

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


if __name__ == "__main__":
    main()
