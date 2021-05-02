"""General utilities for training.

Author:
    Shrey Desai
"""

import os
import json
import gzip
import pickle

import torch
from tqdm import tqdm


def cuda(args, tensor):
    """
    Places tensor on CUDA device (by default, uses cuda:0).

    Args:
        tensor: PyTorch tensor.

    Returns:
        Tensor on CUDA device.
    """
    if args.use_gpu and torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def unpack(tensor):
    """
    Unpacks tensor into Python list.

    Args:
        tensor: PyTorch tensor.

    Returns:
        Python list with tensor contents.
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy().tolist()


def load_dataset(path):
    """
    Loads MRQA-formatted dataset from path.

    Args:
        path: Dataset path, e.g. "datasets/squad_train.jsonl.gz"

    Returns:
        Dataset metadata and samples.
    """
    with gzip.open(path, 'rb') as f:
        elems = [
            json.loads(l.rstrip())
            for l in tqdm(f, desc=f'loading \'{path}\'', leave=False)
        ]
    meta, samples = elems[0], elems[1:]
    return (meta, samples)


def load_cached_embeddings(path):
    """
    Loads embedding from pickle cache, if it exists, otherwise embeddings
    are loaded into memory and cached for future accesses.

    Args:
        path: Embedding path, e.g. "glove/glove.6B.300d.txt".

    Returns:
        Dictionary mapping words (strings) to vectors (list of floats).
    """
    bare_path = os.path.splitext(path)[0]
    cached_path = f'{bare_path}.pkl'
    if os.path.exists(cached_path):
        return pickle.load(open(cached_path, 'rb'))
    embedding_map = load_embeddings(path)
    pickle.dump(embedding_map, open(cached_path, 'wb'))
    return embedding_map


def load_embeddings(path):
    """
    Loads GloVe-style embeddings into memory. This is *extremely slow* if used
    standalone -- `load_cached_embeddings` is almost always preferable.

    Args:
        path: Embedding path, e.g. "glove/glove.6B.300d.txt".

    Returns:
        Dictionary mapping words (strings) to vectors (list of floats).
    """
    embedding_map = {}
    with open(path) as f:
        next(f)  # Skip header.
        for line in f:
            try:
                pieces = line.rstrip().split()
                embedding_map[pieces[0]] = [float(weight) for weight in pieces[1:]]
            except:
                pass
    return embedding_map


def search_span_endpoints(start_probs, end_probs, window=15):
    """
    Finds an optimal answer span given start and end probabilities.
    Specifically, this algorithm finds the optimal start probability p_s, then
    searches for the end probability p_e such that p_s * p_e (joint probability
    of the answer span) is maximized. Finally, the search is locally constrained
    to tokens lying `window` away from the optimal starting point.

    Args:
        start_probs: Distribution over start positions.
        end_probs: Distribution over end positions.
        window: Specifies a context sizefrom which the optimal span endpoint
            is chosen from. This hyperparameter follows directly from the
            DrQA paper (https://arxiv.org/abs/1704.00051).

    Returns:
        Optimal starting and ending indices for the answer span. Note that the
        chosen end index is *inclusive*.
    """
    max_start_index = start_probs.index(max(start_probs))
    max_end_index = -1
    max_joint_prob = 0.

    for end_index in range(len(end_probs)):
        if max_start_index <= end_index <= max_start_index + window:
            joint_prob = start_probs[max_start_index] * end_probs[end_index]
            if joint_prob > max_joint_prob:
                max_joint_prob = joint_prob
                max_end_index = end_index

    return (max_start_index, max_end_index)


## Longest non-continguous substring/subarray, using dynamic programming
def lcs(X, Y):
    inputs_are_lists = type(X) in [list, tuple] and type(Y) in [list, tuple]
    m = len(X)
    n = len(Y)

    ## DP arrays:
    L = [[0] * (n + 1) for _ in range(m + 1)]
    backtrack = [[' '] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
                backtrack[i][j] = '⬉'
            else:
                if L[i - 1][j] > L[i][j - 1]:
                    backtrack[i][j] = '↑'
                else:
                    backtrack[i][j] = '←'
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    ## Go backwards and find the longest NON-CONTIGUOUS substring/subarray
    if inputs_are_lists:
        common = []  ## Works for lists of string/integers
    else:
        common = ''
    i = m
    j = n
    while i != 0 and j != 0:
        if backtrack[i][j] == '←':
            j = j - 1
        elif backtrack[i][j] == '↑':
            i = i - 1
        else:
            if inputs_are_lists:
                common.append(X[i - 1])
            else:
                common += X[i - 1]
            i = i - 1
            j = j - 1
    common = common[::-1]
    ## L[m][n] contains the length of LCS of X[0..n-1] and Y[0..m-1]
    assert len(common) == L[m][n]
    return L[m][n], common


if __name__ == '__main__':
    assert lcs('abc', 'abc') == (3, 'abc')
    assert lcs('xyzhatsourdough', 'thatsourmantra') == (7, 'hatsour')
    assert lcs('xyzhatANOTHERSTRINGsourdough', 'thatsourmantra') == (7, 'hatsour')
    assert lcs(
        ['what', 'in', 'the', 'world', '!', 'That', '...', 'is', 'that', 'what', 'I', 'think', '...', 'this', 'is',
         'amazing', '!'],
        ['arguably', 'the', 'greatest', 'F1', 'driver', 'in', 'the', 'world', ',', 'Lewis', 'Hamilton', 'is', '36',
         'this', 'January'],
    ) == (5, ['in', 'the', 'world', 'is', 'this'])
