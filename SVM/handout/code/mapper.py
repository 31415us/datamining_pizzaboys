#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

from sklearn import linear_model as lm

DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.

OUT_DIMENSION = 400
RAND_SEED = 5911

class RandomFourierProjector(object):

    def __init__(self, out_dim, seed):
        np.random.seed(seed)
        self.out_dim = out_dim
        self.Omega = np.array([np.random.normal(0.0, 1.0, DIMENSION) for i in range(0, self.out_dim)])
        self.b = np.random.uniform(0, 2 * np.pi, self.out_dim)

    def project(self, x):
        projection = self.Omega.dot(x) + self.b

        return np.sqrt(2.0 / self.out_dim) * np.cos(projection)

PROJECTOR = RandomFourierProjector(OUT_DIMENSION, RAND_SEED)

def transform(x_original):
    return PROJECTOR.project(x_original)

if __name__ == "__main__":

    classifier = lm.SGDClassifier(loss='hinge', penalty='l2', fit_intercept=False)

    #nb_iter = 1
    #x_vecs = []
    #labels = []

    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  # Use our features.

        #x_vecs.append(x)
        #labels.append(label)

        # single point updates
        classifier.partial_fit(np.array([x]), np.array([label]), classes=CLASSES)

        # use for batch updates
        #if nb_iter % 10000 == 0:
        #    classifier.partial_fit(np.array(x_vecs), np.array(labels), classes=CLASSES)
        #    x_vecs = []
        #    labels = []

        #nb_iter = nb_iter + 1

    # use dummy key to ensure that we reduce over all mapper outputs
    str_repr = ' '.join(map(str, classifier.coef_[0]))
    print "{key}\t{val}".format(key=1, val=str_repr)
