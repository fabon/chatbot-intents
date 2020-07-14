import sys
import os
import pickle
import keras
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.backend import argmax
import tensorflow as tf
from .model_lstm import load_model_lstm
from ..utils.dataset_utils import load_datasets_sequence, import_clean_labels
from ..utils.path_utils import *

from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

def perfs_binary(y_test, preds_labels):
    f1=f1_score(y_test, preds_labels)
    prec=precision_score(y_test, preds_labels)
    rec=recall_score(y_test, preds_labels)
    return (f1, prec, rec)

def perfs_multiclass(y_test, preds_labels, mode="micro"):
    f1=f1_score(y_test, preds_labels, average=mode)
    prec=precision_score(y_test, preds_labels, average=mode)
    rec=recall_score(y_test, preds_labels, average=mode)
    return (f1, prec, rec)

def train_model(argv, max_f1, trial):
    language=argv[1]
    if not os.path.isfile(OUTPUT_DIR + language + "_index_lstm.pkl") or not os.path.isfile(OUTPUT_DIR + language + "_X_train_lstm.pkl"):
        index, word_counts, X_train, y_train, X_test, y_test=load_datasets_sequence(language)
        pickle.dump(index, open(OUTPUT_DIR + language +   "_index_lstm.pkl", "wb"))
        pickle.dump(X_train, open(OUTPUT_DIR + language + "_X_train_lstm.pkl", "wb"))
        pickle.dump(y_train, open(OUTPUT_DIR + language + "_y_train_lstm.pkl", "wb"))
        pickle.dump(X_test, open(OUTPUT_DIR + language +  "_X_test_lstm.pkl", "wb"))
        pickle.dump(y_test, open(OUTPUT_DIR + language +  "_y_test_lstm.pkl", "wb"))
    else:
        index=pickle.load(open(OUTPUT_DIR + language +    "_index_lstm.pkl", "rb"))
        X_train=pickle.load(open(OUTPUT_DIR + language +  "_X_train_lstm.pkl", "rb"))
#         y_train=pickle.load(open(OUTPUT_DIR + language +  "_y_train_lstm.pkl", "rb"))
        y_train=pickle.load(open(  OUTPUT_DIR  + language + "_train_multiclass_labels_merged.pkl", "rb"))
        X_test=pickle.load(open(OUTPUT_DIR + language +   "_X_test_lstm.pkl", "rb"))
#         y_test=pickle.load(open(OUTPUT_DIR + language +   "_y_test_lstm.pkl", "rb"))
        y_test=pickle.load(open(  OUTPUT_DIR  + language + "_test_multiclass_labels_merged.pkl", "rb"))
        clean_labels=import_clean_labels("/Users/fabon.dzogang/projects/perso/chatbots-intent/chatbot-intents/multiclass/translate/%s/denoise_%s_labels.txt" % (language,language))
        clean_labels=np.array(clean_labels)

    model=load_model_lstm(X_train.shape[1], len(index), 1e-3)

    opt=Adam(learning_rate=1e-3, decay=1e-2)
    model.compile(opt, loss="categorical_crossentropy")
    print (model.summary())
    class_weights={}
    for yval in range(len(set((y_train)))):
        class_weights[yval]=len(y_train)/float(np.sum(np.array(y_train) == yval))


#     history=model.fit(X_train, to_categorical(y_train), batch_size=128, epochs=20, shuffle=True)
#     history=model.fit(X_train, to_categorical(y_train), batch_size=128, epochs=1, shuffle=True)
    history=model.fit(X_train, to_categorical(y_train), batch_size=128, epochs=20, shuffle=True, class_weight=class_weights)

#     preds_train=model.predict(X_train)
#     preds_labels=argmax(preds_train)
#     train_f1=f1_score(preds_labels, y_train)
#     train_prec=precision_score(preds_labels, y_train)
#     train_rec=recall_score(preds_labels, y_train)
#     print ("TRAIN F1[%f] PREC[%f] REC[%f]" % (train_f1, train_prec, train_rec))

#     preds_test=model.predict(X_test)
#     preds_labels=argmax(preds_test)
#     test_f1=f1_score(preds_labels, y_test)
#     test_prec=precision_score(preds_labels, y_test)
#     test_rec=recall_score(preds_labels, y_test)
#     print ("TEST F1[%f] PREC[%f] REC[%f]" % (test_f1, test_prec, test_rec))

#     if max_f1[0] == 0 or max_f1[0] < test_f1:
#         pickle.dump(model, open(OUTPUT_DIR + language + "_lstm_model.pkl", "wb"))
#         pickle.dump(history, open(OUTPUT_DIR + language + "_lstm_history.pkl", "wb"))
#     return train_f1, train_prec, train_rec, test_f1, test_prec, test_rec
    preds_train=model.predict(X_train)
    preds_labels=argmax(preds_train)

    train_f1, train_prec, train_rec=perfs_multiclass(y_train, preds_labels, "macro")
    print ("TRAIN{MACRO} F1[%f] PREC[%f] REC[%f]" % (train_f1, train_prec, train_rec))
    train_f1, train_prec, train_rec=perfs_multiclass(y_train, preds_labels, "micro")
    print ("TRAIN{MICRO} F1[%f] PREC[%f] REC[%f]" % (train_f1, train_prec, train_rec))

    preds_test=model.predict(X_test)
    preds_labels=argmax(preds_test)
#     test_f1, test_prec, test_rec=perfs_binary(y_test, preds_labels)
#     print ("TEST F1[%f] PREC[%f] REC[%f]" % (test_f1, test_prec, test_rec))

    test_f1, test_prec, test_rec=perfs_multiclass(y_test, preds_labels, "macro")
    print ("TEST{MACRO} F1[%f] PREC[%f] REC[%f]" % (test_f1, test_prec, test_rec))
    test_f1, test_prec, test_rec=perfs_multiclass(y_test, preds_labels, "micro")
    print ("TEST{MICRO} F1[%f] PREC[%f] REC[%f]" % (test_f1, test_prec, test_rec))

    keep_labels=np.where(clean_labels!=-1)[0]
    print (keep_labels)
    y_test=clean_labels[keep_labels]
    preds_labels=preds_labels.numpy()
    preds_labels=preds_labels[keep_labels]

    clean_f1, clean_prec, clean_rec=perfs_multiclass(y_test, preds_labels, "macro")
    print ("CLEAN{MACRO} F1[%f] PREC[%f] REC[%f]" % (clean_f1, clean_prec, clean_rec))
    clean_f1, clean_prec, clean_rec=perfs_multiclass(y_test, preds_labels, "micro")
    print ("CLEAN{MICRO} F1[%f] PREC[%f] REC[%f]" % (clean_f1, clean_prec, clean_rec))

#     if max_f1[0] == 0 or max_f1[0] < test_f1:
#         pickle.dump(model, open(OUTPUT_DIR + language + "_lstms_model.pkl", "wb"))
#         pickle.dump(history, open(OUTPUT_DIR + language + "_lstms_history.pkl", "wb"))
    pickle.dump(model, open(OUTPUT_DIR + language + "_lstm_model_%i.pkl" % trial, "wb"))
    pickle.dump(history, open(OUTPUT_DIR + language + "_lstm_history_%i.pkl" % trial, "wb"))

    return train_f1, train_prec, train_rec, test_f1, test_prec, test_rec, clean_f1, clean_prec, clean_rec


def main(argv):
    data=[]
#     for i in range(99):
#     for i in range(1):
#     for i in range(100):
#         max_f1=[0]
#         train_f1, train_prec, train_rec, test_f1, test_prec, test_rec=train_model(argv, max_f1)
#         data.append((train_f1, train_prec, train_rec, test_f1, test_prec, test_rec))

#     with open("replication_lstm_%s.csv" % argv[1], "w") as f:
#         f.write("train_f1,train_prec,train_rec,test_f1,test_prec,test_rec\n")
#         for row in data:
#             f.write(",".join([str(v) for v in row]) + "\n")


    final_acc_scores=[]
#     nb_trials=100
    nb_trials=10
    for i in range(nb_trials):
        max_f1=[0]
        train_f1, train_prec, train_rec, test_f1, test_prec, test_rec, clean_f1, clean_prec, clean_rec=train_model(argv, max_f1, i)
        data.append((train_f1, train_prec, train_rec, test_f1, test_prec, test_rec, clean_f1, clean_prec, clean_rec))
        final_acc_scores.append(clean_f1)

    with open("replication_lstm_%s.csv" % argv[1], "w") as f:
        f.write("train_f1,train_prec,train_rec,test_f1,test_prec,test_rec,clean_f1,clean_prec,clean_rec\n")
        for row in data:
            f.write(",".join([str(v) for v in row]) + "\n")

    import scipy.stats
    print ("STATS micro")
    print ("REPLICATIONS [%i]" % nb_trials)
    print ("AVG [%.2f]" % (100*np.average(final_acc_scores)))
    print ("MED [%.2f]" % (100*np.median(final_acc_scores)))
    print ("MIN [%.2f]" % (100*np.min(final_acc_scores)))
    print ("MAX [%.2f]" % (100*np.max(final_acc_scores)))
    print ("IQR [%.4f]" % scipy.stats.iqr(final_acc_scores))



if __name__ == "__main__":
    main(sys.argv)
