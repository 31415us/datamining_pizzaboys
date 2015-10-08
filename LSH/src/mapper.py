#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys


if __name__ == "__main__":
    # VERY IMPORTANT:
    # Make sure that each machine is using the
    # same seed when generating random numbers for the hash functions.
    np.random.seed(seed=42)
    MAX_INT = np.iinfo(np.int16).max
    BUCKET_NUM = 10
    BUCKET_SIZE = 10
    HASH_FUNC_NUM = BUCKET_SIZE*BUCKET_NUM
    hash_functions = np.random.randint(MAX_INT, size=(HASH_FUNC_NUM, 2))
    bucket_hash_functions = np.random.randint(MAX_INT, size=(BUCKET_SIZE + 1))
    by_buckets = {}
    for line in sys.stdin:
        line = line.strip()
        video_id = int(line[6:15])
        shingles = np.fromstring(line[16:], sep=" ")
        signature = np.ones(HASH_FUNC_NUM)*np.iinfo(np.int16).max
        for r in shingles:
            for i in range(HASH_FUNC_NUM):
                hash_value = np.mod(hash_functions[i, 0]*r + hash_functions[i, 1], MAX_INT)
                signature[i] = np.minimum(hash_value, signature[i])
        for i in range(BUCKET_NUM):
            s = signature[i*BUCKET_SIZE:(i+1)*BUCKET_SIZE]
            s = np.append(s, [1])
            key = sum(s[:]*bucket_hash_functions[:])
            print str(key) + '\t' + line
