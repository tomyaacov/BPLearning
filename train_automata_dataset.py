from bp_network import CustomRNNCell, preprocess
#import graphviz
import numpy as np
import tensorflow as tf
import rstr
from sklearn.model_selection import train_test_split
from train_config import *
import os

DIR = os.path.dirname(os.path.realpath(__file__))

with open(DIR + '/data/' + EXPERIMENT_NAME + '_graph_data.csv', 'r') as f:
    dataset = []
    labels = []
    for line in f:
        x, y = line.strip().split(',')
        x = [DICTIONARY.index(i) for i in list(x)]
        dataset.append(x)
        y_final = [0]*NUMBER_OF_STATES
        y_final[int(y)] = 1
        labels.append(y_final)
    dataset = np.array(dataset)
    labels = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=TEST_SPLIT_PCT)


# ours
cell = CustomRNNCell(NUMBER_OF_STATES, dictionary_size=len(DICTIONARY))
rnn = tf.keras.layers.RNN(cell)  # , return_sequences=True
input_1 = tf.keras.Input((None, len(DICTIONARY), NUMBER_OF_STATES, NUMBER_OF_STATES), dtype=tf.float32)
rnn1 = rnn(input_1, initial_state=tf.convert_to_tensor(START_POSITION))
l = tf.keras.layers.Dense(NUMBER_OF_STATES, activation='relu')
intermediate = l(rnn1)
l2 = tf.keras.layers.Dense(NUMBER_OF_STATES, activation='softmax')
final_output = l2(intermediate)
model = tf.keras.models.Model(input_1, final_output)
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
X_train_adj = preprocess(X_train, len(DICTIONARY), NUMBER_OF_STATES)
X_test_adj = preprocess(X_test, len(DICTIONARY), NUMBER_OF_STATES)
history = model.fit(X_train_adj, y_train, epochs=EPOCHS, batch_size=1, verbose=0, validation_data=(X_test_adj, y_test))
print("ours acc", history.history["acc"])
ours_acc = history.history["acc"]
print("ours val acc", history.history["val_acc"])
ours_val_acc = history.history["val_acc"]
# lstm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
model = Sequential()
model.add(LSTM(NUMBER_OF_STATES, input_shape=(SEQUENCE_LENGTH, len(DICTIONARY))))
model.add(Dropout(0.5))
model.add(Dense(NUMBER_OF_STATES, activation='relu'))
model.add(Dense(NUMBER_OF_STATES, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_adj = to_categorical(X_train)
X_test_adj = to_categorical(X_test)
history = model.fit(X_train_adj, y_train, epochs=EPOCHS, batch_size=1, verbose=0, validation_data=(X_test_adj, y_test))
print("lstm acc", history.history["acc"])
lstm_acc = history.history["acc"]
print("lstm val acc", history.history["val_acc"])
lstm_val_acc = history.history["val_acc"]

# rnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
model = Sequential()
model.add(SimpleRNN(NUMBER_OF_STATES, input_shape=(SEQUENCE_LENGTH, len(DICTIONARY)), activation='relu'))
model.add(Dense(NUMBER_OF_STATES, activation='relu'))
model.add(Dense(units=NUMBER_OF_STATES, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_adj = to_categorical(X_train)
X_test_adj = to_categorical(X_test)
history = model.fit(X_train_adj, y_train, epochs=EPOCHS, batch_size=1, verbose=0, validation_data=(X_test_adj, y_test))
print("rnn acc", history.history["acc"])
rnn_acc = history.history["acc"]
print("rnn val acc", history.history["val_acc"])
rnn_val_acc = history.history["val_acc"]

# ours
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
composed_model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

def batch_gen(dataset_list, labels):
    while True:
        ind = np.random.choice(dataset_list[0].shape[0])
        yield [x[ind:(ind + 1)] for x in dataset_list], labels[ind:(ind + 1)]


history = composed_model.fit_generator(batch_gen(X_train_adj, y_train),
                                       epochs=EPOCHS,
                                       steps_per_epoch=X_train_adj[0].shape[0],
                                       validation_data=batch_gen(X_test_adj, y_test),
                                       validation_steps=X_test_adj[0].shape[0],
                                       verbose=0)

print("ours composite acc", history.history["acc"])
ours_com_acc = history.history["acc"]
print("ours composite val acc", history.history["val_acc"])
ours_com_val_acc = history.history["val_acc"]

import matplotlib.pyplot as plt
plt.ylim(0,1)
plt.plot(ours_val_acc, "b-", label='single')
plt.plot(lstm_val_acc, "g-", label='lstm')
plt.plot(rnn_val_acc, "r-", label='rnn')
plt.plot(ours_com_val_acc, "y-", label='composite')
plt.legend()
plt.savefig(DIR + "/out/" + EXPERIMENT_NAME + '.png')