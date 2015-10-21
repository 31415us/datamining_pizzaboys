#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.

LAMBDA = 1.0

def transform(x_original):
    return x_original

if __name__ == "__main__":
    inv_sqrt_lambda = 1.0 / np.sqrt(LAMBDA)
    w = np.zeros(DIMENSION)
    #w = np.random.random(DIMENSION)
    nu = 1.0
    nb_iter = 1
    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  # Use our features.

        score = label * x.dot(w)

        nu = 1.0 / np.sqrt(nb_iter)

        if score < 1.0:
            unprojected = w - nu * label * x
            inv_norm = 1.0 / np.linalg.norm(unprojected)
            w = min(1.0, inv_sqrt_lambda * inv_norm) * unprojected

        nb_iter = nb_iter + 1

    # use dummy key to ensure that we reduce over all mapper outputs
    str_repr = ' '.join(map(str, w))
    print "{key}\t{val}".format(key=1, val=str_repr)

