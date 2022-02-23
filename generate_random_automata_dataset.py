from bp_network import CustomRNNCell, preprocess
# import graphviz
import numpy as np
import tensorflow as tf
import rstr
from train_config import *

FIXED_M = np.zeros((len(DICTIONARY), NUMBER_OF_STATES, NUMBER_OF_STATES))

for i in range(FIXED_M.shape[0]):
    for j in range(FIXED_M.shape[1]):
        FIXED_M[i,j,np.random.choice(NUMBER_OF_STATES)] = 1

# dot = graphviz.Digraph()
# # for i in range(NUMBER_OF_STATES):
# #     dot.node(str(i))
# for i in range(FIXED_M.shape[0]):
#     for j in range(FIXED_M.shape[1]):
#         dot.edge(str(j), str(np.argmax(FIXED_M[i,j,:])),label=DICTIONARY[i])
# dot.render("data/" + EXPERIMENT_NAME + "_graph")

cell = CustomRNNCell(NUMBER_OF_STATES, dictionary_size=len(DICTIONARY), fixed_weights=FIXED_M)
rnn = tf.keras.layers.RNN(cell)  # , return_sequences=True
input_1 = tf.keras.Input((None, len(DICTIONARY), NUMBER_OF_STATES, NUMBER_OF_STATES), dtype=tf.float32)
rnn1 = rnn(input_1, initial_state=tf.convert_to_tensor(START_POSITION))
model = tf.keras.models.Model(input_1, rnn1)
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

# input_1_data = np.array([[1, 0, 2]])
#
# input_1_data = preprocess(input_1_data, len(DICTIONARY), NUMBER_OF_STATES)
# print(model.predict(input_1_data))

data_to_file = ""
for i in range(NUMBER_OF_SAMPLES):
    current_word = rstr.rstr("".join(DICTIONARY), SEQUENCE_LENGTH)
    input_data = np.array([[DICTIONARY.index(x) for x in current_word]])
    input_data = preprocess(input_data, len(DICTIONARY), NUMBER_OF_STATES)
    output_data = model.predict(input_data)
    current_final_state = np.argmax(output_data)
    data_to_file += current_word + "," + str(current_final_state) + "\n"

with open("data/" + EXPERIMENT_NAME + "_graph_data.csv", "w") as f:
    f.write(data_to_file)