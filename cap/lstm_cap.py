# lstm model
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.model_selection import LeaveOneOut


# # load a single file as a numpy array
# def load_file(filepath):
#     dataframe = read_csv(filepath, header=None, delim_whitespace=True)
#     return dataframe.values
#
#
# # load a list of files and return as a 3d numpy array
# def load_group(filenames, prefix=''):
#     loaded = list()
#     for name in filenames:
#         data = load_file(prefix + name)
#         loaded.append(data)
#     # stack group so that features are the 3rd dimension
#     loaded = dstack(loaded)
#     return loaded
#
#
# # load a dataset group, such as train or test
# def load_dataset_group(group, prefix=''):
#     filepath = prefix + group + '/Inertial Signals/'
#     # load all 9 files as a single array
#     filenames = list()
#     # total acceleration
#     filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
#     # body acceleration
#     filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
#     # body gyroscope
#     filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
#     # load input data
#     X = load_group(filenames, filepath)
#     # load class output
#     y = load_file(prefix + group + '/y_' + group + '.txt')
#     return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix='data/cap-sleep-preprocessed/'):
    X = np.load(prefix + 'data.npy')
    y = np.load(prefix + 'labels.npy')
    X = to_categorical(X)
    return X, y


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 10, trainX.shape[0]
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=1, verbose=0)
    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(repeats=1):
    # load data
    X, y= load_dataset()
    loo = LeaveOneOut()
    scores = list()
    counter = 1
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        score = evaluate_model(X_train, y_train, X_test, y_test)
        score = score * 100.0
        print('>#%d: %.3f' % (counter, score))
        counter += 1
        scores.append(score)
    summarize_results(scores)

# run the experiment
run_experiment()


