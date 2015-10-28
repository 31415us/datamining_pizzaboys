#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

from mapper import OUT_DIMENSION

if __name__ == "__main__":
    count = 0
    w_acc = np.zeros(OUT_DIMENSION)
    for line in sys.stdin:
        line = line.strip()

        key, value = line.split('\t')

        w_acc = w_acc + np.fromstring(value, sep=' ')
        count = count + 1

    w_final = (1.0 / count) * w_acc
    str_repr = ' '.join(map(str, w_final))

    print str_repr
    
