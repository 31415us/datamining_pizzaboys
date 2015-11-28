#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
import random
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import KDTree


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
    print("Start")
    i = 0
    for line in sys.stdin:
        line = line.strip()
        # Create sampling distribution
        sample = np.fromstring(line, sep=" ")
        # if random.random() < t:
            # X = np.vstack( (X, sample) )
        X = np.vstack( (X, sample) )
        i+=1
        sys.stdout.write(str(i)+'\r')

    D_prime = X
    B = []
    print("Stop")

    while len(B) < 0.1*len(X):
        numSamples = 1000
        sample_indices = np.random.randint(0, len(D_prime), size=numSamples) # for S
       
        S = [D_prime[i] for i in sample_indices]
        D_prime = np.delete(D_prime, sample_indices, 1)

        tree = KDTree(D_prime)

        indices_to_remove = set()
        for sample in S:
            indices_to_remove = indices_to_remove.union(set(tree.query(sample, k=3)[1]))

        D_prime = np.delete(D_prime, indices_to_remove, 1)   

        B+=S
        print(len(B))


    clusterer = MiniBatchKMeans(n_clusters=k, batch_size=1000, n_init=10, init=B)

    clusterer.fit(X)

    for center in clusterer.cluster_centers_:
        str_repr = ' '.join(map(str, center))
        emit(1, str_repr)

