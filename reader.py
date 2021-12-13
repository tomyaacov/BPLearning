import matlab.engine
import numpy as np
import os

RAW_DATA_DIR = "/Users/tomyaacov/university/BPLearning/data/cap-sleep-database-1.0.0/"
PROCESSED_DATA_DIR = "/Users/tomyaacov/university/BPLearning/data/cap-sleep-preprocessed"

# for file in os.listdir(RAW_DATA_DIR):
#     if (file.startswith('nfle') or file.startswith('rbd')) and file.endswith(".txt"):
#         print("processing", file)
#         try:
#             eng = matlab.engine.start_matlab()
#             matlab_array = eng.ScoringReader(file, RAW_DATA_DIR)
#             np_array = np.array(matlab_array._data).reshape(matlab_array.size[::-1]).T
#             np.save(os.path.join(PROCESSED_DATA_DIR, file.replace(".txt", ".npy")), np_array)
#             eng.quit()
#         except Exception:
#             continue

for file in os.listdir(PROCESSED_DATA_DIR):
    a = np.load(os.path.join(PROCESSED_DATA_DIR, file))
    print(a.shape)
