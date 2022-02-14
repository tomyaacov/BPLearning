import rstr
import random

words_length = 6
a_in_a_row = 3
dataset_size = 1000

in_language = []
not_in_language = []

while len(in_language) + len(not_in_language) < dataset_size:
    current_word = rstr.rstr('ABC', words_length)
    if random.random() > 0.5:
        start = random.randint(0, words_length-a_in_a_row)
        current_word = current_word[:start] + a_in_a_row*"A" + current_word[start+a_in_a_row:]
        in_language.append(current_word)
    else:
        if a_in_a_row*"A" not in current_word:
            not_in_language.append(current_word)

data = "\n".join([x + " 1" for x in in_language]) + "\n" + "\n".join([x + " 0" for x in not_in_language])
with open("generate_dataset.txt", "w") as f:
    f.write(data)

