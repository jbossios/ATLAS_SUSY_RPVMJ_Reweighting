'''
Authors: Jonathan Bossio, Anthony Badea
Date: Monday April 25, 2022
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def myacc(y_true,y_pred):
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    return tf.reduce_sum(tf.cast((y_true - y_pred)<=0.1, tf.int32))/tf.size(y_true)

def mean_pred(y_true,y_pred):
    return K.mean(y_pred)

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -1 * ((y_true) * K.log(y_pred + epsilon) + (1 - y_true) * K.log(1 - y_pred + epsilon)) * weights 
    return K.mean(t_loss)

def sqrtR_loss(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # exponentiate prediction to avoid requiring y_pred to be positive for sqrt (numerical trick)
    y_pred = K.exp(y_pred)

    # sqrt weights to reduce the very small exponenents (numerical trick)
    weights = K.sqrt(weights)

    # loss
    loss = K.sum(K.sqrt(tf.boolean_mask(y_pred,y_true==0)) * tf.boolean_mask(weights,y_true==0))
    loss += K.sum(1/K.sqrt(tf.boolean_mask(y_pred,y_true==1)) * tf.boolean_mask(weights,y_true==1))
    return loss
    
def make_model(**kargs):
    
    # Create model
    model = tf.keras.Sequential()
    nodes = [kargs["nnode_per_dense"] for i in range(kargs["ndense"]-1)] + [1]
    input_dims = [kargs["input_dim"]] + nodes
    activations = ["relu" for i in range(kargs["ndense"]-1)] + ["sigmoid"]
    for node, input_dim, activation in zip(nodes, input_dims, activations):
        model.add(tf.keras.layers.Dense(node, input_dim=input_dim, activation=activation))

    # Compile model
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=kargs["learning_rate"]),
        loss=kargs["loss"], #tf.keras.losses.BinaryCrossentropy(),
        metrics=[myacc],
    )

    return model

def simple_model(**kargs):
    inputs = Input((kargs["input_dim"], ))
    hidden_layer_1 = Dense(50, activation='relu')(inputs)
    hidden_layer_2 = Dense(100, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
    outputs = Dense(1, activation='linear')(hidden_layer_3) # sigmoid
    model = Model(inputs=inputs, outputs=outputs)
    # Compile model
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=kargs["learning_rate"]),
        loss=sqrtR_loss,
        metrics=[mean_pred],
    )
    return model

if __name__ == "__main__":
    model, _ = make_model(input_dim=10, nodes=100, learning_rate=1e4)
    model.summary()
