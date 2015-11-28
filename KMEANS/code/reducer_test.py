#!/usr/bin/env python2.7

import sys
import numpy as np
from ast import literal_eval
from scipy.spatial.distance import cdist

if __name__ == "__main__":
    C = None
    
    for line in sys.stdin:
        line = line.strip()
        key, value = line.split('\t')
        subset = literal_eval(value)
        
        if C is None:
            C = subset
        else:
            C += subset
    
    C = np.asarray(C)
    S_positions = np.random.randint(0, len(C), 100)
    
    i = 0
    centers = [ C[index] for index in S_positions ]
    while i < len(centers):
        j = 0
        while j < len(centers[i]):
            if j != len(centers[i]) -1:
                sys.stdout.write(str(centers[i][j]) + ' ')
            else:
                sys.stdout.write(str(centers[i][j]) + '\n')
            
            j += 1
        i += 1 
