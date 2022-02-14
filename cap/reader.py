import matlab.engine
import numpy as np
import os

RAW_DATA_DIR = "/Users/tomyaacov/university/BPLearning/data/cap-sleep-database-1.0.0/"
PROCESSED_DATA_DIR = "/Users/tomyaacov/university/BPLearning/data/cap-sleep-preprocessed"
data = {}
for file in os.listdir(RAW_DATA_DIR):
    if (file.startswith('nfle') or file.startswith('rbd')) and file.endswith(".txt"):
        print("processing", file)
        try:
            eng = matlab.engine.start_matlab()
            matlab_array = eng.ScoringReader(file, RAW_DATA_DIR)
            np_array = np.array(matlab_array._data).reshape(matlab_array.size[::-1]).T
            data[file.replace(".txt", "")] = np_array
            eng.quit()
        except Exception:
            continue

min_timestamp = min([x[0, 1] for x in data.values()])
max_timestamp = max([x[-1, 1] for x in data.values()])
series_length = (max_timestamp - min_timestamp) // 30
total_data = np.zeros((data.__len__(), int(series_length)), dtype=int)
labels = np.zeros((data.__len__(), 2), dtype=int)
series_counter = 0
for k, v in data.items():
    new_counter = 0
    older_counter = 0
    timestamp_counter = min_timestamp
    while new_counter < total_data.shape[1]:
        if timestamp_counter < v[0, 1]:
            total_data[series_counter, new_counter] = 0
            new_counter += 1
            timestamp_counter += 30
        elif timestamp_counter < v[-1, 1]:
            total_data[series_counter, new_counter] = v[older_counter, 0]
            new_counter += 1
            timestamp_counter += 30
            older_counter += 1
        else:
            total_data[series_counter, new_counter] = 0
            new_counter += 1
            timestamp_counter += 30
    if k.startswith("nfle"):
        labels[series_counter, 0] = 1
    else:
        labels[series_counter, 1] = 1
    series_counter += 1

np.save(os.path.join(PROCESSED_DATA_DIR, "data.npy"), total_data)
np.save(os.path.join(PROCESSED_DATA_DIR, "labels.npy"), labels)

# for file in os.listdir(PROCESSED_DATA_DIR):
#     a = np.load(os.path.join(PROCESSED_DATA_DIR, file))
#     print(a)
