from tensorflow.keras import Model
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import pathlib

# from Timestamp2Vec_Class import helper_functions
from Timestamp2Vec_Model.helper_functions import *

ORG = pathlib.Path(__file__).parents[1]
LOC_VARS = str(ORG) + "/Data/important_variables/"
NAME_MAX_VAL = "max_val.npy"
NAME_MIN_VAL = "min_val.npy"
ENCODER_LOCATION = str(pathlib.Path(__file__).parents[0]) + "/encoder_VAE"

min_val = np.load(LOC_VARS + NAME_MIN_VAL)
max_val = np.load(LOC_VARS + NAME_MAX_VAL)

class Timestamp2Vec(Model):
    def __init__(self):
        super(Timestamp2Vec, self).__init__()
        self.vectorize = Vectorize()
        self.encoder = keras.models.load_model(ENCODER_LOCATION)
    
    def call(self, x):
        # vectorize the input into features
        x = self.vectorize(x)
        # obtain the latent variable and take the mean
        z = self.encoder.predict(x)[0]
        return z


class Vectorize(Layer):
    # vectorize the incoming timestamp
    def call(self, inputs):
        if type(inputs) == str:
            inputs = np.array([inputs])
        elif type(inputs) == list:
            inputs = np.array(inputs)
        inputs = np.array(list(map(np.datetime64, inputs)))
        inputs = inputs.astype('datetime64[ms]')
        inputs = np.array(list(map(extract_features_date, inputs)))
        inputs = np.array(list(map(normalize, inputs)))
        inputs = np.asarray(inputs).astype('float32')
        inputs = tf.reshape(inputs, [-1, 22])

        return inputs