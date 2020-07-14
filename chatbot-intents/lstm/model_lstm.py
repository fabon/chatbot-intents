import keras
from keras.layers import Input,Dense,LSTM,Embedding
from keras.models import Model
from keras import regularizers
import tensorflow as tf

EMBEDDING_SIZE=256
MEM_SIZE=16
# OUTPUT_UNITS=2
OUTPUT_UNITS=3

def load_model_lstm(max_len, vocab_size, reg_value=1e-3):
    input_layer = Input(shape=(max_len,))

    embedding_layer = Embedding(vocab_size+1,
                                EMBEDDING_SIZE,
                                mask_zero=True,
                                input_length=max_len,
                                embeddings_regularizer=regularizers.l2(reg_value))(input_layer)
    lstm_layer = LSTM(MEM_SIZE,
                      activation="tanh",
                      return_sequences=False)(embedding_layer)
#                       kernel_regularizer=regularizers.l2(reg_value))(embedding_layer)
    output_layer = Dense(OUTPUT_UNITS,
                         kernel_regularizer=regularizers.l2(reg_value),
                         activation="softmax")(lstm_layer)
    return Model(input_layer, output_layer)

