import numpy as np
import json

mapping = json.load(open("vocab.json"))
weight = np.load("embedding-weight.npy")
d = weight.shape[-1]

with open(f"bpe-embedding{d}.txt", "w+") as f:
    for k in mapping.keys():
        v = weight[mapping[k]]
        f.write(k)
        for w in v:
            f.write(" {:.5}".format(w))

        f.write("\n")
