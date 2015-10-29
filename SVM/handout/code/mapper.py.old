#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.

LAMBDA = 0.1

OUT_DIMENSION = DIMENSION # dimension after transformation

def transform(x_original):
    #out = np.concatenate([x_original, x_original * x_original])
    out = np.sqrt(x_original * x_original * x_original)
    return out

if __name__ == "__main__":
    inv_sqrt_lambda = 1.0 / np.sqrt(LAMBDA)
    w = np.zeros(OUT_DIMENSION)
    #w = np.random.random(DIMENSION)
    nu = 1.0
    nb_iter = 1

    diag = np.ones(OUT_DIMENSION)

    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  # Use our features.

        score = label * x.dot(w)

        #nu = 1.0 / np.sqrt(nb_iter)
        nu = 1.0 / (LAMBDA * nb_iter)

        grad = w

        if score < 1.0:
            grad = grad + label * x

        diag = diag + grad * grad

        #unprojected = w - nu * grad
        #inv_norm = 1.0 / np.linalg.norm(unprojected)
        #w = min(1.0, inv_sqrt_lambda * inv_norm) * unprojected

        steps = nu * (np.ones(OUT_DIMENSION) / np.sqrt(diag))

        w = w - steps * grad

        # projection
        inv_norm = 1.0 / np.linalg.norm(w)
        w = min(1.0, inv_sqrt_lambda * inv_norm) * w

        nb_iter = nb_iter + 1

    # use dummy key to ensure that we reduce over all mapper outputs
    str_repr = ' '.join(map(str, w))
    print "{key}\t{val}".format(key=1, val=str_repr)

