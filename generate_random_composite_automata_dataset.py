from bp_network import CustomRNNCell, preprocess
import numpy as np
import tensorflow as tf
import rstr
import itertools
import pandas as pd
from train_config import *

# FIXED_M = [np.zeros((len(DICTIONARY), num_of_s, num_of_s)) for num_of_s in COMPOSITE_STATE_NUM]
# for a in FIXED_M:
#     for i in range(a.shape[0]):
#         for j in range(a.shape[1]):
#             a[i,j,np.random.choice(a.shape[2])] = 1

FIXED_M = []
FIXED_M.append(np.array([]))

print(FIXED_M)
model_inputs = []
cells = []
rnns = []
rnn_outputs = []
models = []
for ind, num_of_s in enumerate(COMPOSITE_STATE_NUM):
    model_inputs.append(tf.keras.Input((None, len(DICTIONARY), num_of_s, num_of_s), dtype=tf.float32))
    if isinstance(FIXED_M[ind], np.ndarray):
        cells.append(CustomRNNCell(num_of_s, dictionary_size=len(DICTIONARY), fixed_weights=FIXED_M[ind]))
    else:
        cells.append(CustomRNNCell(num_of_s, dictionary_size=len(DICTIONARY)))
    rnns.append(tf.keras.layers.RNN(cells[ind]))  # , return_sequences=True
    rnn_outputs.append(rnns[ind](model_inputs[ind], initial_state=tf.convert_to_tensor(COMPOSITE_START_POSITION[ind])))
    models.append(tf.keras.models.Model(model_inputs[ind], rnn_outputs[ind]))
    models[ind].summary()

#added = tf.keras.layers.Add()(rnn_outputs)
con = tf.keras.layers.concatenate(rnn_outputs)
# l = tf.keras.layers.Dense(NUMBER_OF_STATES, activation='relu')
# intermediate = l(con)
# l2 = tf.keras.layers.Dense(NUMBER_OF_STATES, activation='softmax')
# final_output = l2(intermediate)
composed_model = tf.keras.models.Model(inputs=model_inputs, outputs=con)
composed_model.summary()
composed_model.compile(optimizer="adam", loss="mse", metrics=["accuracy", 'mse'])


data_to_file = pd.DataFrame(columns=["word", "label"])
counter = 0
for item in itertools.product(DICTIONARY, repeat=SEQUENCE_LENGTH):
    current_word = "".join(item)
    input_1_data = np.array([[DICTIONARY.index(x) for x in current_word]])
    X = []
    for ind, num_of_s in enumerate(COMPOSITE_STATE_NUM):
        X.append(preprocess(input_1_data, len(DICTIONARY), num_of_s))
    current_final_state = composed_model.predict(X)
    a = np.where(current_final_state[0])[0]
    data_to_file.loc[counter] = [current_word, a[0] * 9 + (a[1] - 3) * 3 + (a[2] - 6)]
    counter += 1

data_to_file.to_csv("data/" + EXPERIMENT_NAME + "_graph_data.csv", index=False)
