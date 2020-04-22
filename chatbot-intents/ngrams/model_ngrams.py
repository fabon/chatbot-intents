import keras
from keras.layers import Input,Dense
from keras.models import Model
from keras import regularizers
import tensorflow as tf

OUTPUT_UNITS=2

def load_model_ngrams(input_size, reg_value=1e-3):
    input_layer = Input(shape=(input_size,))
    output_layer = Dense(OUTPUT_UNITS,
                         kernel_regularizer=regularizers.l2(reg_value),
                         activation="softmax")(input_layer)
    return Model(input_layer, output_layer)

