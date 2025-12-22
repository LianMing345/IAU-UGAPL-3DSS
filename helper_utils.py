import numpy as np
import scipy.stats

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


def asymmetricKL(P, Q):
    return np.sum(P * np.log((P + 1e-12) / (Q + 1e-12) + 1e-12), axis=1)  # Calculate the kl-divergence between P and Q


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M, base=2, axis=1) + 0.5 * scipy.stats.entropy(q, M, base=2, axis=1)



def comput_similarity(a, b):  # (f,) (f,)
    a = a / np.sqrt(np.sum(np.square(a)))
    b = b / np.sqrt(np.sum(np.square(b)))
    return np.sum(a * b)


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))
