import tensorflow as tf
import numpy as np

def make_model(**kargs):
    input_dim = kargs['input_dim']
    nodes = kargs['nodes']
    learning_rate = kargs['learning_rate']
    # Create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(nodes, input_dim=input_dim))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    # Callbacks
    callbacks = []
    # EarlyStopping
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, mode='min', restore_best_weights=True, monitor="val_loss")
    callbacks.append(early_stopping)
    # ModelCheckpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
    callbacks.append(model_checkpoint)
    # Terminate on NaN such that it is easier to debug
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    return model, callbacks

if __name__ == '__main__':
    model, _ = make_model(input_dim=10, nodes=100, learning_rate=1e4)
    model.summary()
