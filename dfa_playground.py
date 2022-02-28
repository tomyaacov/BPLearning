# from dfa import DFA
#
#
# def par_compose(dfas: list[DFA]) -> DFA:
#     def transition(state, char):
#         return tuple(d._transition(s, c) for d, s, c in zip(dfas, state, char))
#
#     def label(state):
#         return tuple(d._label(s) for d, s in zip(dfas, state))
#
#     return DFA(
#         start=tuple(d.start for d in dfas),
#         inputs=product(d.inputs for d in dfas),
#         outputs=product(d.outputs for d in dfas),
#         label=label,
#         transition=transition,
#     )
