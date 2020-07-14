import sys
import os
import collections
import pickle
import numpy as np
import keras
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.backend import argmax
import tensorflow as tf
from .model_ngrams import load_model_ngrams
from ..utils.dataset_utils import load_datasets_binary, import_clean_labels, import_training_sets
from ..utils.path_utils import *

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

def test_model(model, X_train, y_train, X_test, y_test, clean_labels, max_f1):
    preds_train=model.predict(X_train)
    preds_labels=argmax(preds_train)

    train_f1, train_prec, train_rec=perfs_multiclass(y_train, preds_labels, "macro")
    print ("TRAIN{MACRO} F1[%f] PREC[%f] REC[%f]" % (train_f1, train_prec, train_rec))
    train_f1, train_prec, train_rec=perfs_multiclass(y_train, preds_labels, "micro")
    print ("TRAIN{MICRO} F1[%f] PREC[%f] REC[%f]" % (train_f1, train_prec, train_rec))

    preds_test=model.predict(X_test)
    preds_labels=argmax(preds_test)

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
#         pickle.dump(model, open(OUTPUT_DIR + language + "_ngrams_model.pkl", "wb"))
#         pickle.dump(history, open(OUTPUT_DIR + language + "_ngrams_history.pkl", "wb"))
    return train_f1, train_prec, train_rec, test_f1, test_prec, test_rec, clean_f1, clean_prec, clean_rec

def load_english_model(recompute_index, training_inds, max_f1, trial):
    language="english"
    index=pickle.load(open(   OUTPUT_DIR  + language + "_index_ngrams.pkl", "rb"))
    X_train=pickle.load(open( OUTPUT_DIR  + language + "_X_train_ngrams.pkl", "rb"))
    y_train=pickle.load(open( OUTPUT_DIR  + language + "_train_multiclass_labels.pkl", "rb"))
    X_test=pickle.load(open(  OUTPUT_DIR  + language + "_X_test_ngrams.pkl", "rb"))
    y_test=pickle.load(open(  OUTPUT_DIR  + language + "_test_multiclass_labels.pkl", "rb"))
    word_counts=pickle.load(open (OUTPUT_DIR + language + "_wc.pkl", "rb"))
    clean_labels=import_clean_labels("/Users/fabon.dzogang/projects/perso/chatbots-intent/chatbot-intents/multiclass/translate/%s/denoise_%s_labels.txt" % (language,language))
    clean_labels=np.array(clean_labels)
    model=pickle.load(open(   OUTPUT_DIR + language + "_ngrams_model_%i_%i.pkl" % (trial, len(training_inds))))
    return model, X_train, y_train, X_test, y_test, clean_labels

def train_model(recompute_index, training_inds, language, trial):
#     if not os.path.isfile(OUTPUT_DIR + language + "_index_ngrams.pkl") or not os.path.isfile(OUTPUT_DIR + language + "_X_train_ngrams.pkl"):
    if recompute_index:
        index, word_counts, X_train, _, X_test, _=load_datasets_binary(language, training_inds)
        y_train=pickle.load(open(  OUTPUT_DIR  + language + "_train_multiclass_labels.pkl", "rb"))
        y_test=pickle.load(open(  OUTPUT_DIR  + language + "_test_multiclass_labels.pkl", "rb"))
        pickle.dump(index, open(  OUTPUT_DIR  + language + "_index_ngrams_%i.pkl" % len(training_inds), "wb"))
#         return None, None, None, None, None, None
        pickle.dump(X_train, open(OUTPUT_DIR  + language + "_X_train_ngrams.pkl", "wb"))
        pickle.dump(y_train, open(OUTPUT_DIR  + language + "_y_train_ngrams.pkl", "wb"))
        pickle.dump(X_test, open( OUTPUT_DIR  + language + "_X_test_ngrams.pkl", "wb"))
        pickle.dump(y_test, open( OUTPUT_DIR  + language + "_y_test_ngrams.pkl", "wb"))
        pickle.dump(word_counts, open(OUTPUT_DIR + language + "_wc.pkl", "wb"))
    else:
        index=pickle.load(open(   OUTPUT_DIR  + language + "_index_ngrams_%i.pkl" % len(training_inds), "rb"))
        X_train=pickle.load(open( OUTPUT_DIR  + language + "_X_train_ngrams.pkl", "rb"))
        y_train=pickle.load(open(  OUTPUT_DIR  + language + "_train_multiclass_labels.pkl", "rb"))
        X_test=pickle.load(open(  OUTPUT_DIR  + language + "_X_test_ngrams.pkl", "rb"))
        y_test=pickle.load(open(  OUTPUT_DIR  + language + "_test_multiclass_labels.pkl", "rb"))
        word_counts=pickle.load(open (OUTPUT_DIR + language + "_wc.pkl", "rb"))
#     clean_labels=import_clean_labels("/Users/fabon.dzogang/projects/perso/chatbots-intent/chatbot-intents/multiclass/translate/spanish/round2_denoise/denoise_%s_labels.txt" % language)
    clean_labels=import_clean_labels("/Users/fabon.dzogang/projects/perso/chatbots-intent/chatbot-intents/multiclass/translate/%s/denoise_%s_labels.txt" % (language,language))
    clean_labels=np.array(clean_labels)

    y_train=(np.array(y_train)[training_inds]).tolist()
    print (X_train.shape)
    print (len(y_train))
    print (X_test.shape)
    print (len(y_test))

    model=load_model_ngrams(X_train.shape[1], 1e-4)

    opt=SGD(learning_rate=1.0, decay=1e-2)
    model.compile(opt, loss="categorical_crossentropy")
    print (model.summary())
    class_weights={}
    for yval in range(len(set((y_train)))):
        class_weights[yval]=len(y_train)/float(np.sum(np.array(y_train) == yval))

    import sklearn

    history=model.fit(X_train, to_categorical(y_train), batch_size=32, epochs=20, shuffle=True, class_weight=class_weights)
    pickle.dump(model, open(OUTPUT_DIR + language + "_ngrams_model_%i_%i.pkl" % (trial, len(training_inds)), "wb"))
    pickle.dump(history, open(OUTPUT_DIR + language + "_ngrams_history_%i_%i.pkl" % (trial, len(training_inds)), "wb"))

    return model, X_train, y_train, X_test, y_test, clean_labels

def main(argv):
    data=[]
    final_acc_scores=[]
#     nb_trials=100
    import scipy.stats

    f=open("replication_ngrams_%s.csv" % argv[1], "w")
    f.write("sample_size,train_f1,train_prec,train_rec,test_f1,test_prec,test_rec,clean_f1,clean_prec,clean_rec\n")

    language=argv[1]
    training_sets=import_training_sets(language)
    training_sets=collections.OrderedDict(sorted(training_sets.items(), key=lambda tup:tup[0]))
    print_buff=[]
    for sample_size in training_sets.keys():
        print ("%i samples" % sample_size)
        training_set=training_sets[sample_size]
        nb_trials=10
        for i in range(nb_trials):
            max_f1=[0]
            model, X_train, y_train, X_test, y_test, clean_labels=train_model(i==0,training_set, language, i)
#             model, X_train, y_train, X_test, y_test, clean_labels=load_english_model(i==0, training_set, i)
            train_f1, train_prec, train_rec, test_f1, test_prec, test_rec, clean_f1, clean_prec, clean_rec=test_model(model, X_train, y_train, X_test, y_test, clean_labels, max_f1)
            data.append((train_f1, train_prec, train_rec, test_f1, test_prec, test_rec, clean_f1, clean_prec, clean_rec))
            final_acc_scores.append(clean_f1)

            for row in data:
                f.write(",".join([str(v) for v in [sample_size]+list(row)]) + "\n")
                f.flush()

        print_buff.append("STATS micro")
        print_buff.append("SAMPLE SIZE [%i]" % sample_size)
        print_buff.append("REPLICATIONS [%i]" % nb_trials)
        print_buff.append("AVG [%.2f]" % (100*np.average(final_acc_scores)))
        print_buff.append("MED [%.2f]" % (100*np.median(final_acc_scores)))
        print_buff.append("MIN [%.2f]" % (100*np.min(final_acc_scores)))
        print_buff.append("MAX [%.2f]" % (100*np.max(final_acc_scores)))
        print_buff.append("IQR [%.4f]" % scipy.stats.iqr(final_acc_scores))
        print_buff.append("")

        print ("\n".join(print_buff[-9:]))
    f.close()
    print ("\n".join(print_buff))

if __name__ == "__main__":
    main(sys.argv)
