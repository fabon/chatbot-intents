import copy
import numpy as np
from nltk.tokenize import TweetTokenizer
from scipy.sparse import lil_matrix

tokenizer = TweetTokenizer()
MIN_WORD_COUNT=10
NGRAMS_ORDERS=[1,2,3]
STOPWORDS=[]
MAX_LEN=200

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

def tokenize_ngrams(msg, ngrams_orders, stopwords):
    tokens = tokenizer.tokenize(msg)
    clean_tokens = []
    for token in tokens:
        if not clean_token(token) or token in stopwords:
            continue
        clean_tokens.append(token)
    ngrams_tokens = ngrams(clean_tokens, ngrams_orders)
    return ngrams_tokens

def extract_index(data, higher_order_ngrams=True):
    index = {}
    entry = 0
    word_counts = {}
    ngrams_orders=NGRAMS_ORDERS if higher_order_ngrams else [1]
    for msg in data:
        tokens = tokenize_ngrams(msg, ngrams_orders, STOPWORDS)

        for token in tokens:
            if token not in index:
                index[token] = entry
                entry = entry + 1
            if token not in word_counts:
                word_counts[token] = 0
            word_counts[token] = word_counts[token] + 1
    return index, word_counts

def encode_messages_binary(index, messages):
    vocab_size=np.max([ind for ind in index.values()])
#     X = lil_matrix(np.zeros((len(messages), len(index))))
    X = lil_matrix(np.zeros((len(messages), vocab_size+1)))
    i = 0
    for msg in messages:
        for word in tokenize_ngrams(msg, NGRAMS_ORDERS, STOPWORDS):
            if word in index:
                entry = index[word]
                X[i, entry] = 1
        i = i + 1
    return X

def encode_messages_sequence(index, messages):
    vocab_size=np.max([ind for ind in index.values()])
    X = np.zeros((len(messages), MAX_LEN))
    i = 0
    for msg in messages:
        position=0
        for word in tokenize_ngrams(msg, [1], STOPWORDS):
            if word in index:
                entry = index[word]
                X[i, position] = entry+1
#                 print (word)
#                 print (entry)
#                 print (X[i, position])
                position = position + 1
            if position >= MAX_LEN:
                break
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

