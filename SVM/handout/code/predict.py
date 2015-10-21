
import sys

import numpy as np

W_FILE_PATH = sys.argv[1]
PREDICTION_FILE_PATH = sys.argv[2]

w = None

for line in open(W_FILE_PATH, 'r'):
    w = np.fromstring(line.strip(), sep=' ')

for line in open(PREDICTION_FILE_PATH, 'r'):
    x = np.fromstring(line.strip(), sep=' ')
    predicted_label = int(np.sign(x.dot(w)))

    print predicted_label
