from bppy import *

@b_thread
def req(letter):
    for i in range(10):
        yield {request: BEvent(letter)}

@b_thread
def prevent():
    not_b = EventSet(lambda e: e.name != "B")
    while True:
        yield {waitFor: BEvent("A")}
        e = yield {waitFor: not_b}
        if e.name == "A":
            yield {block: BEvent("A"), waitFor: not_b}


b_program = BProgram(bthreads=[req(x) for x in ["A", "B", "C"]] + [prevent()],
                     event_selection_strategy=SimpleEventSelectionStrategy(),
                     listener=PrintBProgramRunnerListener())
b_program.run()
