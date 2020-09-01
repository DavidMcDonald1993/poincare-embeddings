#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import count
from collections import defaultdict as ddict
import numpy as np
import torch as th

import gzip

def parse_seperator(line, length, sep='\t'):
    d = line.strip().split(sep)
    if len(d) == length:
        w = 1
    elif len(d) == length + 1:
        w = int(float(d[-1]))
        d = d[:-1]
    else:
        raise RuntimeError(f'Malformed input ({line.strip()})')
    return tuple(d) + (w,)


def parse_tsv(line, length=2):
    return parse_seperator(line, length, '\t')


def parse_space(line, length=2):
    return parse_seperator(line, length, ' ')


def iter_line(fname, fparse, length=2, comment='#'):
    if fname.endswith(".gz"):
        # with gzip.open(fname, "rt") as fin:
        fin = gzip.open(fname, "rt")
    else:
        fin = open(fname, "r")
    # with open(fname, 'r') as fin:

    for line in fin:
        if line[0] == comment:
            continue
        tpl = fparse(line, length=length)
        if tpl is not None:
            yield tpl
    fin.close()


def intmap_to_list(d):
    arr = [None for _ in range(len(d))]
    for v, i in d.items():
        arr[i] = v
    assert not any(x is None for x in arr)
    return arr


def slurp(fin, fparse=parse_tsv, symmetrize=False):

    class IdentityDict(ddict):
        def missing(self, key):
            self.update({key: int(key)})
            return self.get(key)
        
        __missing__ = missing

    # ecount = count()
    # enames = ddict(ecount.__next__)
    enames = IdentityDict()

    subs = []
    for i, j, w in iter_line(fin, fparse, length=2):
        subs.append((enames[i], enames[j], w))
        # subs.append((i, j, w))
        if symmetrize:
            subs.append((enames[j], enames[i], w))
            # subs.append((j, i, w))
    idx = th.from_numpy(np.array(subs, dtype=np.int))

    # freeze defaultdicts after training data and convert to arrays
    objects = intmap_to_list(dict(enames))
    print(f'slurp: objects={len(objects)}, edges={len(idx)}')
    return idx, objects
