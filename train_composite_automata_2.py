import pandas as pd

from bp_network import CustomRNNCell, preprocess
import numpy as np
import tensorflow as tf
import rstr
from sklearn.model_selection import train_test_split
from train_config import *
import os

DIR = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(DIR + '/data/' + EXPERIMENT_NAME + '_graph_data.csv')
labels = df.iloc[:, 1:].to_numpy()
dataset = []
for word in list(df.iloc[:,0]):
    x = [DICTIONARY.index(i) for i in word]
    dataset.append(x)
dataset = np.array(dataset)
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=TEST_SPLIT_PCT)

model_inputs = []
cells = []
rnns = []
rnn_outputs = []
models = []
X_train_adj = []
X_test_adj = []
for ind, num_of_s in enumerate(COMPOSITE_STATE_NUM):
    model_inputs.append(tf.keras.Input((None, len(DICTIONARY), num_of_s, num_of_s), dtype=tf.float32))
    cells.append(CustomRNNCell(num_of_s, dictionary_size=len(DICTIONARY)))
    rnns.append(tf.keras.layers.RNN(cells[ind]))  # , return_sequences=True
    rnn_outputs.append(rnns[ind](model_inputs[ind], initial_state=tf.convert_to_tensor(COMPOSITE_START_POSITION[ind])))
    models.append(tf.keras.models.Model(model_inputs[ind], rnn_outputs[ind]))
    models[ind].summary()
    X_train_adj.append(preprocess(X_train, len(DICTIONARY), num_of_s))
    X_test_adj.append(preprocess(X_test, len(DICTIONARY), num_of_s))

con = tf.keras.layers.concatenate(rnn_outputs)
l = tf.keras.layers.Dense(NUMBER_OF_STATES, activation='relu')
intermediate = l(con)
l2 = tf.keras.layers.Dense(NUMBER_OF_STATES, activation='softmax')
final_output = l2(intermediate)
composed_model = tf.keras.models.Model(inputs=model_inputs, outputs=final_output)
composed_model.summary()
composed_model.compile(optimizer="adam", loss="mse", metrics=['mse'])


def batch_gen(dataset_list, labels):
    while True:
        ind = np.random.choice(dataset_list[0].shape[0])
        yield [x[ind:(ind + 1)] for x in dataset_list], labels[ind:(ind + 1)]


history = composed_model.fit_generator(batch_gen(X_train_adj, y_train),
                                       epochs=EPOCHS,
                                       steps_per_epoch=X_train_adj[0].shape[0],
                                       validation_data=batch_gen(X_test_adj, y_test),
                                       validation_steps=X_test_adj[0].shape[0],
                                       verbose=0
                                       )

print("ours mean_squared_error", history.history["mean_squared_error"])
ours_comp_acc = history.history["mean_squared_error"]
print("ours val mean_squared_error", history.history["val_mean_squared_error"])
ours_comp_val_acc = history.history["val_mean_squared_error"]

# ours
cell = CustomRNNCell(NUMBER_OF_STATES, dictionary_size=len(DICTIONARY))
rnn = tf.keras.layers.RNN(cell)  # , return_sequences=True
input_1 = tf.keras.Input((None, len(DICTIONARY), NUMBER_OF_STATES, NUMBER_OF_STATES), dtype=tf.float32)
rnn1 = rnn(input_1, initial_state=tf.convert_to_tensor(START_POSITION))
model = tf.keras.models.Model(input_1, rnn1)
model.compile(optimizer="adam", loss="mse", metrics=["mse"])
X_train_adj = preprocess(X_train, len(DICTIONARY), NUMBER_OF_STATES)
X_test_adj = preprocess(X_test, len(DICTIONARY), NUMBER_OF_STATES)
history = model.fit(X_train_adj, y_train, epochs=EPOCHS, batch_size=1, verbose=0, validation_data=(X_test_adj, y_test))
print("ours mean_squared_error", history.history["mean_squared_error"])
ours_acc = history.history["mean_squared_error"]
print("ours val mean_squared_error", history.history["val_mean_squared_error"])
ours_val_acc = history.history["val_mean_squared_error"]

# lstm
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(NUMBER_OF_STATES, input_shape=(SEQUENCE_LENGTH, len(DICTIONARY))))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(NUMBER_OF_STATES, activation='relu'))
model.add(tf.keras.layers.Dense(NUMBER_OF_STATES, activation='softmax'))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
X_train_adj = tf.keras.utils.to_categorical(X_train)
X_test_adj = tf.keras.utils.to_categorical(X_test)
history = model.fit(X_train_adj, y_train, epochs=EPOCHS, batch_size=1, verbose=0, validation_data=(X_test_adj, y_test))
print("lstm mean_squared_error", history.history["mean_squared_error"])
lstm_acc = history.history["mean_squared_error"]
print("lstm val mean_squared_error", history.history["val_mean_squared_error"])
lstm_val_acc = history.history["val_mean_squared_error"]

# rnn
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(NUMBER_OF_STATES, input_shape=(SEQUENCE_LENGTH, len(DICTIONARY)), activation='relu'))
model.add(tf.keras.layers.Dense(NUMBER_OF_STATES, activation='relu'))
model.add(tf.keras.layers.Dense(units=NUMBER_OF_STATES, activation='softmax'))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
X_train_adj = tf.keras.utils.to_categorical(X_train)
X_test_adj = tf.keras.utils.to_categorical(X_test)
history = model.fit(X_train_adj, y_train, epochs=EPOCHS, batch_size=1, verbose=0, validation_data=(X_test_adj, y_test))
print("rnn mean_squared_error", history.history["mean_squared_error"])
rnn_acc = history.history["mean_squared_error"]
print("rnn val mean_squared_error", history.history["val_mean_squared_error"])
rnn_val_acc = history.history["val_mean_squared_error"]

import matplotlib.pyplot as plt
plt.plot(ours_comp_acc, "y-", label='ours_comp_acc')
plt.plot(ours_comp_val_acc, "y--", label='ours_comp_val_acc')
plt.plot(ours_acc, "b-", label='ours_acc')
plt.plot(ours_val_acc, "b--", label='ours_val_acc')
plt.plot(lstm_acc, "g-", label='lstm_acc')
plt.plot(lstm_val_acc, "g--", label='lstm_val_acc')
plt.plot(rnn_acc, "r-", label='rnn_acc')
plt.plot(rnn_val_acc, "r--", label='rnn_val_acc')
plt.legend()
plt.savefig(DIR + "/out/" + EXPERIMENT_NAME + '.png')