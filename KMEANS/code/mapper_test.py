#!/usr/bin/env python2.7

import sys
import random
import numpy as np
from ast import literal_eval
from scipy.spatial.distance import cdist

d = 500
k = 100

threshold = 0.3
epsilon = 0.9999

verbose = False

def emit(key, value):
    print('%s\t%s' % (key, value))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '-v':
        verbose = True
    
    np.random.seed(seed=42)
    D = np.zeros(shape=(0, d))
    
    i = 0
    for line in sys.stdin:
        if random.random() < threshold:
            sample = np.fromstring(line, sep=" ")
            D = np.vstack( (D, sample) )
    
    if verbose:
        print 'D dimensions = '+ str(D.shape)
        print '10 * d * k * np.log(1/epsilon) = '+ str(10 * d * k * np.log(1/epsilon))
    
    B = []
    D_prime = D
    while len(D_prime) > 0:
        num_samples = min(len(D_prime), 10 * d * k * np.log(1/epsilon))
        S_positions = np.random.randint(0, len(D_prime), num_samples)
        S = [ D_prime[i] for i in S_positions ]
        
        max_distances = np.amax(cdist(np.asarray(S), D_prime, 'euclidean'), axis=0)
        # list of tuples (distance, index)
        max_tup_distances = [ (max_distances[i], i) for i in xrange(len(max_distances)) ]
        max_tup_distances.sort()
        
        new_D_prime_len = len(D_prime) / 2
        if verbose:
            print '--------------------'
            print 'Number of samples = '+ str(num_samples)
            print 'Previous len of D_prime = '+ str(len(D_prime))
            print 'New len of D_prime = '+ str(new_D_prime_len)
        
        D_prime = np.asarray([ D_prime[t[1]] for t in max_tup_distances[:new_D_prime_len] ])
        B += S
    
    # removing duplicates
    B = np.unique(map(lambda a: str(list(a.flat)), B))
    
    emit(1, map(literal_eval, B))