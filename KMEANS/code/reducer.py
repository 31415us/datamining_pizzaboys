#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

from sklearn.cluster import KMeans

k = 100
d = 500

if __name__ == "__main__":

    X = np.zeros(shape=(0,d))

    for line in sys.stdin:
        line = line.strip()
        key, value = line.split('\t')
        x = np.fromstring(value, sep=" ")
        X = np.vstack( (X, x) )

    clusterer = KMeans(n_clusters=k)

    clusterer.fit(X)

    for center in clusterer.cluster_centers_:
        str_repr = ' '.join(map(str, center))
        print str_repr

