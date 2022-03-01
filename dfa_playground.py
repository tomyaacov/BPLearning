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


dfa1 = DFA(  # number of As is odd
    start=0,
    inputs={"A", "B", "C"},
    label=lambda s: (s % 2) == 1,
    transition=lambda s, c: (s + 1) % 2 if c == "A" else s,
)

dfa2 = DFA(  # number of Bs is odd
    start=0,
    inputs={"A", "B", "C"},
    label=lambda s: (s % 2) == 1,
    transition=lambda s, c: (s + 1) % 2 if c == "B" else s,
)

dfa3 = DFA(  # two Bs in a row
    start=0,
    inputs={"A", "B", "C"},
    label=lambda s: s == 2,
    transition=lambda s, c: min(s+1, 2) if c == "B" or s == 2 else 0,
)

comp_dfa = compose([dfa1, dfa2, dfa3])
#print(comp_dfa.transition(["A", "A", "B", "B", "A", "A"]))

with open("data/3_graph_data.csv", "r") as f:
    for line in f.readlines():
        w, l = line.strip().split(",")
        dfa_l = comp_dfa.transition(list(w))
        dfa_l = sum([x*(2**i) for i, x in enumerate(dfa_l)])
        if dfa_l != int(l):
            raise ValueError(w)
