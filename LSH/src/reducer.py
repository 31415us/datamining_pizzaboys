#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys


def print_duplicates(lines):
    unique = np.unique(lines)
    for i in xrange(len(unique)):
        for j in xrange(i + 1, len(unique)):
            line_a = unique[i]
            line_b = unique[j]
            video_a = int(line_a[6:15])
            video_b = int(line_b[6:15])
            shingles_a = np.fromstring(line_a[16:], sep=" ")
            shingles_b = np.fromstring(line_b[16:], sep=" ")
            sim = compute_similarity(shingles_a, shingles_b)
            if sim >= 0.9:
                print "%d\t%d" % (min(video_a, video_b),
                                  max(video_a, video_b))


def compute_similarity(shingles1, shingles2):
    inters = np.intersect1d(shingles1, shingles2).size
    union = np.union1d(shingles1, shingles2).size
    return float(inters)/union


last_key = None
key_count = 0
duplicates = []

for line in sys.stdin:
    line = line.strip()
    key, line = line.split("\t")
    line = line.strip()
    if last_key is None:
        last_key = key

    if key == last_key:
        duplicates.append(line)
    else:
        # Key changed (previous line was k=x, this line is k=y)
        print_duplicates(duplicates)
        duplicates = [line]
        last_key = key

if len(duplicates) > 0:
    print_duplicates(duplicates)
