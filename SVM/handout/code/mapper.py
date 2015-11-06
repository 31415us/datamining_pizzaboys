#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

from sklearn import linear_model as lm

DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.

def transform(x_original):
    return np.concatenate([np.ones(1), np.arctan(x_original), np.sqrt(np.abs(x_original))]) 

def cv_error(X_valid, Y_valid, w):
    pred = np.sign(X_valid.dot(w))
    correct = np.sum(Y_valid == pred)

    return float(correct) / np.size(Y_valid)


if __name__ == "__main__":

    classifier = lm.SGDClassifier(
            loss='modified_huber',
            penalty='l1',
            fit_intercept=False,
            warm_start=False,
            power_t=1.0,
            learning_rate='optimal',
            n_iter=10,
            shuffle=True)

    x_valid = []
    y_valid = []

    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  # Use our features.

        ## uncomment for cross validation
        #if np.random.random() <= 0.1:
        #    x_valid.append(x)
        #    y_valid.append(label)
        #else:
        #    # single point updates
        #    classifier.partial_fit(np.array([x]), np.array([label]), classes=CLASSES)

        classifier.partial_fit(np.array([x]), np.array([label]), classes=CLASSES)

    #print cv_error(np.array(x_valid), np.array(y_valid), classifier.coef_[0])

    # use dummy key to ensure that we reduce over all mapper outputs
    str_repr = ' '.join(map(str, classifier.coef_[0]))
    print "{key}\t{val}".format(key=1, val=str_repr)
