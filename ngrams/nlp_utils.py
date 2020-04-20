import copy
import numpy as np
from nltk.tokenize import TweetTokenizer
from scipy.sparse import lil_matrix

tokenizer = TweetTokenizer()
# MIN_WORD_COUNT=3
MIN_WORD_COUNT=10

def import_stopwords():
    stopwords=set()
    with open("stopwords.txt", "r") as f:
        for line in f:
            if line[0] != ' ':
                stopword=line.split(" ")[0].replace("\n", "")
                if len(stopword):
                    stopwords.add(stopword)
    return stopwords

def load_data(data):
    labels = []
    texts = []

    for (label, msg) in data:
        labels.append(label)
        texts.append(msg)

    return labels, texts

def token_is_digit(token):
    return any(c.isdigit() for c in token)

def token_is_1char(token):
    return len(token) == 1 and token.lower() != "i"

def clean_token(token):
    return not token_is_digit(token) and not token_is_1char(token)

def ngrams(tokens, ngrams_orders):
    ngrams_tokens = []
    for order in ngrams_orders:
        j = 0
        while j < len(tokens) - order + 1:
            ngrams_tokens.append("_".join(tokens[j:j + order]))
            j = j + 1
    return ngrams_tokens

def extract_index(data, ngrams_orders, stopwords):
    index = {}
    entry = 0
    word_counts = {}
    for msg in data:
        tokens = tokenize_ngrams(msg, ngrams_orders, stopwords)

        for token in tokens:
            if token not in index:
                index[token] = entry
                entry = entry + 1
            if token not in word_counts:
                word_counts[token] = 0
            word_counts[token] = word_counts[token] + 1
    return index, word_counts

def encode_messages_binary(index, messages, ngrams_orders, stopwords):
    vocab_size=np.max([ind for ind in index.values()])
#     X = lil_matrix(np.zeros((len(messages), len(index))))
    X = lil_matrix(np.zeros((len(messages), vocab_size+1)))
    i = 0
    for msg in messages:
        for word in tokenize_ngrams(msg, ngrams_orders, stopwords):
            if word in index:
                entry = index[word]
                X[i, entry] = 1
        i = i + 1
    return X

def clean_index(index, word_counts):
    for word in word_counts.keys():
        if word_counts[word] < MIN_WORD_COUNT:
            del index[word]
    return index

def rebuild_index(index):
    i = 0
    for word in index.keys():
        index[word] = i
        i = i + 1
    return index

