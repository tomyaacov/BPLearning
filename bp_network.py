import tensorflow as tf
import numpy as np


class CustomRNNCell(tf.keras.layers.Layer):
    def __init__(self, unit_1, dictionary_size, fixed_weights=None, **kwargs):
        self.unit_1 = unit_1
        self.dictionary_size = dictionary_size
        self.fixed_weights = fixed_weights
        self.state_size = tf.TensorShape([unit_1])
        self.output_size = tf.TensorShape([unit_1])
        super(CustomRNNCell, self).__init__(**kwargs)

    def build(self, input_shapes):
        # expect input_shape to contain 1 item, (batch, i1)
        i1 = input_shapes[2]
        if isinstance(self.fixed_weights, np.ndarray):
            self.M = self.add_weight(  # TODO: change
                shape=self.fixed_weights.shape, initializer=lambda shape, dtype: self.fixed_weights, name="M"
            )
        else:
            self.M = self.add_weight(  # TODO: change
                shape=(self.dictionary_size, i1, i1), initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.), name="M"
            )

    def call(self, inputs, states):
        # inputs should be in (batch, input_1)
        # state should be in shape (batch, unit_1)
        a = tf.multiply(inputs, self.M)
        b = tf.reduce_sum(a, 1)
        c = tf.matmul(states, b)
        output = tf.divide(c,tf.reduce_sum(c)) # try softmax
        return output[:,0], output[:,0]

    def get_config(self):
        return {"unit_1": self.unit_1}




def preprocess(a, depth, number_of_states):
    b = np.zeros((a.shape[0], a.shape[1], depth))
    c = np.meshgrid(np.arange(a.shape[0]), np.arange(a.shape[1]), indexing="ij")
    d = np.zeros(a.shape + (depth, number_of_states, number_of_states), dtype=np.float64)
    b[c[0], c[1], a] = 1
    d[c[0], c[1], a, :, :] = b[c[0], c[1], a] = 1
    return d


if __name__ == "__main__":
    DICTIONARY = ['A', 'B', 'C']
    NUMBER_OF_STATES = 3
    START_POSITION = [[1., 0., 0.]]

    cell = CustomRNNCell(NUMBER_OF_STATES)
    rnn = tf.keras.layers.RNN(cell)  # , return_sequences=True
    input_1 = tf.keras.Input((None, len(DICTIONARY), NUMBER_OF_STATES, NUMBER_OF_STATES), dtype=tf.float32)
    rnn1 = rnn(input_1, initial_state=tf.convert_to_tensor(START_POSITION))
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(rnn1)
    model = tf.keras.models.Model(input_1, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    input_1_data = np.array([[1, 0, 2]])

    input_1_data = preprocess(input_1_data, len(DICTIONARY))
    print(model.predict(input_1_data))

    f = open('other_paper_code/generate_dataset.txt', 'r')
    dataset = []
    labels = []
    for line in f:
        x, y = line.split(' ')
        x = [DICTIONARY.index(i) for i in list(x)]
        isOk = 1
        if y[0] == '0':
            isOk = 0
        dataset.append(x)
        labels.append(isOk)
    dataset = np.array(dataset)
    labels = np.array(labels)

    dataset = preprocess(dataset, len(DICTIONARY))

    print(model.summary())
    model.fit(dataset, labels, epochs=3, batch_size=1, verbose=2)


# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, SimpleRNN, Input
# from tensorflow.keras.utils import to_categorical
# from tensorflow.python.keras import backend as K
#
# DICTIONARY = ['A', 'B', 'C']
# NUMBER_OF_STATES = 5
# START_POSITION = 0
#
# f = open('generate_dataset.txt', 'r')
# dataset = []
# labels = []
# for line in f:
#     x, y = line.split(' ')
#     x = [DICTIONARY.index(i) for i in list(x)]
#     isOk = 1
#     if y[0] == '0':
#         isOk = 0
#     dataset.append(x)
#     labels.append(isOk)
# dataset = to_categorical(dataset)
#
# model = Sequential()
# model.add(SimpleRNN(NUMBER_OF_STATES, input_shape=(dataset.shape[1], dataset.shape[2]), activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation='relu'))
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(dataset, labels, epochs=25, batch_size=1, verbose=2)
#
#
# def from_categorical(a):
#     return [DICTIONARY[x] for x in range(a.shape[0]) if a[x] == 1][0]
#
#
# transitions = set()
# accepting_states = set()
# inputs1 = Input(shape=(dataset.shape[1], dataset.shape[2]))
# rnn1 = SimpleRNN(NUMBER_OF_STATES, activation='relu', return_sequences=True)(inputs1)
# outputs1 = Dense(1, activation='sigmoid')(rnn1)
# demo_model = Model(inputs=inputs1, outputs=[outputs1, rnn1])
# demo_model.set_weights(model.get_weights())
# results = demo_model.predict(dataset)
# for i in range(dataset.shape[0]):
#     before = 0
#     for j in range(dataset.shape[1]):
#         letter = from_categorical(dataset[i,j])
#         after = results[1][i,j].argmax()
#         transitions.add((before, after, letter))
#         before = after
#     if results[0][i,-1][0] > 0.5:
#         accepting_states.add(before)
# for t in transitions:
#     print(str(t[0]) + ' -> ' + str(t[1]) + ' [label="' + str(t[2]) + '"]')
# for a in accepting_states:
#     print(str(a) + " [style=filled, fillcolor=red]")





