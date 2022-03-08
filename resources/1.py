from dfa import DFA
from dfa_utils import compose


EXPERIMENT_NAME = "1"
DICTIONARY = ['A', 'B', 'C']
NUMBER_OF_STATES = 4
COMPOSITE_STATE_NUM = [2,2]
COMPOSITE_START_POSITION = [[[1., 0.]],[[1., 0.]]]
SEQUENCE_LENGTH = 8
START_POSITION = [[1., 0., 0., 0.]]
EPOCHS = 100
TEST_SPLIT_PCT = 0.1

dfa1 = DFA(  # number of As is odd
    start=0,
    inputs=set(DICTIONARY),
    label=lambda s: (s % 2) == 1,
    transition=lambda s, c: (s + 1) % 2 if c == "A" else s,
)

dfa2 = DFA(  # number of Bs is odd
    start=0,
    inputs=set(DICTIONARY),
    label=lambda s: (s % 2) == 1,
    transition=lambda s, c: (s + 1) % 2 if c == "B" else s,
)


COMP_DFA = compose([dfa1, dfa2])

