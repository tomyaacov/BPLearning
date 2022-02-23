


# EXPERIMENT_NAME = "1A"
# DICTIONARY = ['A', 'B', 'C']
# NUMBER_OF_STATES = 3
# SEQUENCE_LENGTH = 4
# START_POSITION = [[1., 0., 0.]]
# NUMBER_OF_SAMPLES = 200
# EPOCHS = 100
# TEST_SPLIT_PCT = 0.1


# EXPERIMENT_NAME = "1B"
# DICTIONARY = ['A', 'B', 'C']
# NUMBER_OF_STATES = 4
# SEQUENCE_LENGTH = 4
# START_POSITION = [[1., 0., 0., 0.]]
# NUMBER_OF_SAMPLES = 200
# EPOCHS = 100
# TEST_SPLIT_PCT = 0.1

# EXPERIMENT_NAME = "2"
# DICTIONARY = ['A', 'B', 'C', 'D']
# NUMBER_OF_STATES = 5
# SEQUENCE_LENGTH = 5
# START_POSITION = [[1., 0., 0., 0., 0.]]
# NUMBER_OF_SAMPLES = 1500
# EPOCHS = 100
# TEST_SPLIT_PCT = 0.1

# EXPERIMENT_NAME = "3"
# DICTIONARY = ['A', 'B', 'C']
# NUMBER_OF_STATES = 12
# COMPOSITE_STATE_NUM = [2,2,3]
# COMPOSITE_START_POSITION = [[[1., 0.]],[[1., 0.]],[[1., 0., 0.]]]
# SEQUENCE_LENGTH = 8
# START_POSITION = [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
# EPOCHS = 100
# TEST_SPLIT_PCT = 0.1
# import re
# def f(w):
#     final = 0
#     if w.count("A")%2==1:
#         final += 1
#     if w.count("B")%2==1:
#         final += 2
#     if re.match('.*BB.*', w):
#         final += 8
#     else:
#         if w.endswith("B"):
#             final += 4
#     return final
# LABELER = lambda x: f(x)

EXPERIMENT_NAME = "4"
DICTIONARY = ['A', 'B', 'C']
NUMBER_OF_STATES = 12
COMPOSITE_STATE_NUM = [2,2,3]
COMPOSITE_START_POSITION = [[[1., 0.]],[[1., 0.]],[[1., 0., 0.]]]
SEQUENCE_LENGTH = 8
START_POSITION = [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
EPOCHS = 20
TEST_SPLIT_PCT = 0.1

