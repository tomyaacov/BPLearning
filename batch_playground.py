import numpy as np
from bp_network import *

FIXED_M = np.array([[[0,1,0],
                     [1,0,0],
                     [0,0,1]],
                    [[0,0,1],
                     [0,0,1],
                     [1,0,0]]])

DICTIONARY = ['A', 'B']
NUMBER_OF_STATES = 3
START_POSITION = [[1., 0., 0.]]

cell = CustomRNNCell(NUMBER_OF_STATES, dictionary_size=len(DICTIONARY))
rnn = tf.keras.layers.RNN(cell)  # , return_sequences=True
input_1 = tf.keras.Input((None, len(DICTIONARY), NUMBER_OF_STATES, NUMBER_OF_STATES), dtype=tf.float32)
rnn1 = rnn(input_1, initial_state=tf.convert_to_tensor(START_POSITION))
model = tf.keras.models.Model(input_1, rnn1)
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
input_1_data = np.array([[1, 0, 2]])
input_1_data = preprocess(input_1_data, len(DICTIONARY), NUMBER_OF_STATES)
print(model.predict(input_1_data))