#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
from ast import literal_eval



if __name__ == "__main__":

    C = None

    for line in sys.stdin:
        line = line.strip()
        key, value = line.split('\t')
        cset = literal_eval(value)

        if C is None:
            C = cset
        else:
            C += cset

        print len(C)

    C = np.asarray(C)


