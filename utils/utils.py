"""util functions
# many old functions, need to clean up
# homography --> homography
# warping
# loss --> delete if useless
"""

import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn

threshold = 0.55
reproj_err = 3
params = [threshold, reproj_err]

query_index_offset = 0
refer_index_offset = 0

pos_ptr = np.array(
    [
        [-99, -98, -97, -96, -95, -94, -93],
        [-67, -66, -65, -64, -63, -62, -61],
        [-35, -34, -33, -32, -31, -30, -29],
        [-3, -2, -1, 0, 1, 2, 3],
        [29, 30, 31, 32, 33, 34, 35],
        [61, 62, 63, 64, 65, 66, 67],
        [93, 94, 95, 96, 97, 98, 99],
    ]
)

idx_table = np.reshape(np.array([val for val in range(0, 32 * 32)]), (32, 32))
cache_table = np.zeros((1024, 2), dtype=int)
for cnt in range(1024):
    ridx = int(cnt / 32)
    cidx = int(cnt % 32)
    cache_table[cnt] = np.array([ridx, cidx])


def load_checkpoint(load_path, filename="checkpoint.pth.tar"):
    file_prefix = ["superPointNet"]
    filename = "{}__{}".format(file_prefix[0], filename)
    # torch.save(net_state, save_path)
    checkpoint = torch.load(load_path / filename)
    print("load checkpoint from ", filename)
    return checkpoint
