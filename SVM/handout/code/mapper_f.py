#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

from sklearn import linear_model as lm
from sklearn import preprocessing
from math import sqrt

DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.
LAMBDA = 1.0
NB_ITER = 20

# def transform(x_original):
#     x =  np.concatenate(
#             [np.ones(1),
#              np.sqrt(np.abs(x_original)) +
#              (np.cosh((np.pi / 2) * x_original) - 1) +
#              np.sin((np.pi / 2) * x_original),
#              ]) 
#     return x

def transform(x_original):
    x =  np.concatenate(
            [np.ones(1),
             preprocessing.scale(x_original),
             ]) 
    return x

def PEGASOS_batch(w,X,Y,nIter):
    currentT = 0
    nSample, nFeature = X.shape
    for t in range(0, nIter):
        selectR = range(0, nSample)
        np.random.shuffle(selectR)
        currentT = 1
        for i in range(0, nSample):
            w = PEGASOS(w, X[selectR[i]], Y[selectR[i]], currentT)
            currentT += 1

    return w

def PEGASOS(w,x,y,t):
    eta = 1.0 / (LAMBDA * t)
    if np.dot(w,x) * y < 1.0:
        w = (1 - eta * LAMBDA) * w + eta * y * x
    else:
        w = (1 - eta * LAMBDA) * w

    if np.linalg.norm(w) * np.sqrt(LAMBDA) > 1e-5:
        w = w * min(1, 1.0/(np.linalg.norm(w) * np.sqrt(LAMBDA)))
    return w

def cv_accuracy(X_valid, Y_valid, w):
    pred = np.sign(X_valid.dot(w))
    correct = np.sum(Y_valid == pred)

    return float(correct) / np.size(Y_valid)

if __name__ == "__main__":

    classifier = lm.SGDClassifier(
            #loss='modified_huber',
            loss='hinge',
            penalty='l1',
            fit_intercept=False,
            warm_start=False,
            power_t=1.0,
            learning_rate='optimal',
            n_iter=10,
            shuffle=True)

    x_valid = []
    y_valid = []

    x_batch100 = []
    y_batch100 = []
    nbSamples = 0
    ws = []
    
    

    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  # Use our features.

        x_batch100.append(x)
        

        y_batch100.append(label)

        nbSamples += 1

        if nbSamples == 100:

            w = np.zeros(401)
            w_up = PEGASOS_batch(w, np.array(x_batch100), np.array(y_batch100), 20)
            ws.append(w_up)
            nbSamples = 0
            x_batch100 = []
            y_batch100 = []
            #print np.array(ws).shape


        # uncomment for cross validation
        #if np.random.random() <= 0.1:
        #    x_valid.append(x)
        #    y_valid.append(label)
        #else:
        #    # single point updates
        #    classifier.partial_fit(np.array([x]), np.array([label]), classes=CLASSES)



        #classifier.partial_fit(np.array([x]), np.array([label]), classes=CLASSES)

    if nbSamples > 0:
        w = np.zeros(401)
        w_up = PEGASOS_batch(w, x_batch100, y_batch100, 20)
        ws.append(w_up)
        print 'nbSamples : %d' % nbSamples
        print np.array(ws).shape

    [nw, f] = np.array(ws).shape

    w_final = (1.0/nw)* np.sum(ws, axis=0)
    


    #print cv_accuracy(np.array(x_valid), np.array(y_valid), classifier.coef_[0])

    # use dummy key to ensure that we reduce over all mapper outputs
    str_repr = ' '.join(map(str, w_final))
    # str_repr = ' '.join(map(str, classifier.coef_[0]))
    print "{key}\t{val}".format(key=1, val=str_repr)
