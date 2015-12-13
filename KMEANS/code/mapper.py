#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans


d = 500
k = 600

def emit(key, value):
    print('%s\t%s' % (key, value))

if __name__ == "__main__":

    X = np.zeros(shape=(0,d))
    for line in sys.stdin:
        line = line.strip()
        sample = np.fromstring(line, sep=" ")
        X = np.vstack( (X, sample) )

    clusterer = MiniBatchKMeans(n_clusters=k, batch_size=1000, n_init=10)

    clusterer.fit(X)

    for center in clusterer.cluster_centers_:
        str_repr = ' '.join(map(str, center))
        emit(1, str_repr)

