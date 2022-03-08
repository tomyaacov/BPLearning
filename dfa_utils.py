from dfa import DFA


def compose(dfas):
    def transition(state, char):
        return tuple(d._transition(s, char) for d, s in zip(dfas, state))

    def label(state):
        return tuple(d._label(s) for d, s in zip(dfas, state))

    return DFA(
        start=tuple(d.start for d in dfas),
        inputs=dfas[0].inputs,
        label=label,
        transition=transition,
    )