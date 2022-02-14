from bp_network import CustomRNNCell, preprocess
import graphviz
import numpy as np
import tensorflow as tf
import rstr
from sklearn.model_selection import train_test_split
from train_config import *

with open('data/graph_data.csv', 'r') as f:
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
model = tf.keras.models.Model(input_1, rnn1)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
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
model.add(LSTM(10, input_shape=(SEQUENCE_LENGTH, len(DICTIONARY))))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
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
model.add(SimpleRNN(10, input_shape=(SEQUENCE_LENGTH, len(DICTIONARY)), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(units=NUMBER_OF_STATES, activation='relu'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_adj = to_categorical(X_train)
X_test_adj = to_categorical(X_test)
history = model.fit(X_train_adj, y_train, epochs=EPOCHS, batch_size=1, verbose=0, validation_data=(X_test_adj, y_test))
print("rnn acc", history.history["acc"])
rnn_acc = history.history["acc"]
print("rnn val acc", history.history["val_acc"])
rnn_val_acc = history.history["val_acc"]

import matplotlib.pyplot as plt
plt.plot(ours_acc, "b-", label='ours_acc')
plt.plot(ours_val_acc, "b--", label='ours_val_acc')
plt.plot(lstm_acc, "g-", label='lstm_acc')
plt.plot(lstm_val_acc, "g--", label='lstm_val_acc')
plt.plot(rnn_acc, "r-", label='rnn_acc')
plt.plot(rnn_val_acc, "r--", label='rnn_val_acc')
plt.legend()
plt.savefig(EXPERIMENT_NAME + '.png')