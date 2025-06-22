from model import dawid_skene
import numpy as np

if __name__ == "__main__":
    Y = [1, 1, 3, 1, 1, 5, 4, 4, 4, 3, 1, 5, 3, 1, 1, 1, 3, 1, 1, 3, 1, 3, 5, 5, 2, 5, 1, 1, 4, 3, 5, 4, 3, 3, 3, 1, 3, 4, 4, 1, 5, 3, 1, 4, 3, 1, 3, 4, 5, 3, 3, 4, 3, 1, 3, 5, 1, 4, 3, 3, 1, 3, 4, 1, 2, 3, 4, 3, 3, 3, 2, 1]
    X = dawid_skene(np.load("peer_tensor.npy", allow_pickle = True))