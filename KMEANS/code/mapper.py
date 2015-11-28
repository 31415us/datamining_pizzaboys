#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
import random

from sklearn.cluster import MiniBatchKMeans


d = 500
k = 100
verbose = False

t = 0.1

def uniq(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def emit(key, value):
    print('%s\t%s' % (key, value))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '-v':
        verbose = True

    X = np.zeros(shape=(0,d))

    for line in sys.stdin:
        line = line.strip()
        # Create sampling distribution
        sample = np.fromstring(line, sep=" ")
        if random.random() < t:
            X = np.vstack( (X, sample) )

    clusterer = MiniBatchKMeans(n_clusters=k, batch_size=1000, n_init=10)

    clusterer.fit(X)

    for center in clusterer.cluster_centers_:
        str_repr = ' '.join(map(str, center))
        emit(1, str_repr)

