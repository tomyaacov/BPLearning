from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K

DICTIONARY = ['A', 'B', 'C']
NUMBER_OF_STATES = 5
START_POSITION = 0

f = open('generate_dataset.txt', 'r')
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
dataset = to_categorical(dataset)

model = Sequential()
model.add(SimpleRNN(NUMBER_OF_STATES, input_shape=(dataset.shape[1], dataset.shape[2]), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(dataset, labels, epochs=25, batch_size=1, verbose=2)


def from_categorical(a):
    return [DICTIONARY[x] for x in range(a.shape[0]) if a[x] == 1][0]


transitions = set()
accepting_states = set()
inputs1 = Input(shape=(dataset.shape[1], dataset.shape[2]))
rnn1 = SimpleRNN(NUMBER_OF_STATES, activation='relu', return_sequences=True)(inputs1)
outputs1 = Dense(1, activation='sigmoid')(rnn1)
demo_model = Model(inputs=inputs1, outputs=[outputs1, rnn1])
demo_model.set_weights(model.get_weights())
results = demo_model.predict(dataset)
for i in range(dataset.shape[0]):
    before = 0
    for j in range(dataset.shape[1]):
        letter = from_categorical(dataset[i,j])
        after = results[1][i,j].argmax()
        transitions.add((before, after, letter))
        before = after
    if results[0][i,-1][0] > 0.5:
        accepting_states.add(before)
for t in transitions:
    print(str(t[0]) + ' -> ' + str(t[1]) + ' [label="' + str(t[2]) + '"]')
for a in accepting_states:
    print(str(a) + " [style=filled, fillcolor=red]")





