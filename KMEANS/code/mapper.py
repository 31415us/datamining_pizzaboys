#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
import random
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from scipy.spatial import Voronoi
from scipy.spatial import KDTree


d = 500
k = 100
epsilon = 0.999 # How to choose epsilon ?
verbose = False


def uniq(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def emit(key, value):
    print('%s\t%s' % (key, value))
    
def q(x, b, Db, sumDist):
    sumDist = 
    return (5/len(Db)+ (euclidean(x, b)**2)/(sumDist**2))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '-v':
        verbose = True

    D = np.zeros(shape=(0,d))
    

    t = 0.3

    np.random.seed(seed=42)

    for line in sys.stdin:
        line = line.strip()
        # Create sampling distribution

        if random.random() < t:
            sample = np.fromstring(line, sep=" ")
            D = np.vstack( (D, sample) )



    # Coresets via Adaptive Sampling
    B = []
    D_prime = D
    
    while len(D_prime) > 0:
        numSamples = min(len(D_prime), 10 * d * k * np.log(1/epsilon))
        sample_indices = np.random.randint(0, len(D_prime), size=numSamples) # for S
       
        S = [D_prime[i] for i in sample_indices]

        # Compute distance from S to D'
        dist = cdist(S, D_prime, 'euclidean')
        # Zip distances and sample index
        distances_tuple = [ (dist[i][j], i) for i in xrange(len(dist)) for j in xrange(len(dist[0])) ]
        distances_tuple = sorted(set(distances_tuple), key=lambda x: x[0])
        # Remove 0 distances
        distances_tuple = [t for t in distances_tuple if t[0] != 0.0]
        # Remove nearest points
        indices_to_remove = uniq(map(lambda e: e[1], distances_tuple))
        if verbose:
            print '--------------------'
            print 'Number of samples = '+ str(numSamples)
            print 'Previous len of D_prime = '+ str(len(D_prime))
            print 'New len of D_prime = '+ str(len(D_prime)/2)


        # D_prime = np.asarray([ D_prime[t] for t in indices_to_remove[:(len(D_prime)/2)] ])

        if len(D_prime)/2 > 1:
            D_prime = np.delete(D_prime, indices_to_remove[:(len(D_prime)/2)], 1)
        else:
            D_prime = []
        B+=S

    # Remove enventual duplicates
    B = np.unique(map(lambda a: str(list(a.flat)), B))

    tree = KDTree(B)


    
    # if verbose:
    #     print 'Now Voronoi partitioning'
    
    # vor = Voronoi(B)
    # if verbose:
    #     print 'Voronoi regions :'
    #     print vor.regions
    # if verbose:
    #     print 'Adding D to vor :'
    # vor.add_points(D)


    emit(1, map(str, B))
        





    
