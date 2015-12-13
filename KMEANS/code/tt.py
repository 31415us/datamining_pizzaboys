#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys

for i in xrange(0, 100000):
    sys.stdout.write(str(i)+'\r')

