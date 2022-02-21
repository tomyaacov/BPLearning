from train_config import *
import itertools

data_to_file = ""
for item in itertools.product(DICTIONARY, repeat=SEQUENCE_LENGTH):
    current_word = "".join(item)
    current_final_state = LABELER(current_word)
    data_to_file += current_word + "," + str(current_final_state) + "\n"

with open("data/" + EXPERIMENT_NAME + "_graph_data.csv", "w") as f:
    f.write(data_to_file)