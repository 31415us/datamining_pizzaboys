#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

from sklearn import linear_model as lm

DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.

def transform(x_original):
    return x_original

if __name__ == "__main__":

    classifier = lm.SGDClassifier(loss='hinge', penalty='l2', fit_intercept=False)

    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  # Use our features.

        classifier.partial_fit(np.array([x]), np.array([label]), classes=CLASSES)

    # use dummy key to ensure that we reduce over all mapper outputs
    str_repr = ' '.join(map(str, classifier.coef_[0]))
    print "{key}\t{val}".format(key=1, val=str_repr)
