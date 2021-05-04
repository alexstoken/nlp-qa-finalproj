"""General utilities for training.

Author:
    Shrey Desai
"""
from typing import *
import os, json, gzip, pickle, time, logging
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm

QID_KEY = 'qid'
PASSAGE_TOKENS_KEY = 'passage_tokens'
PASSAGE_KEY = 'passage'
QUESTION_TOKENS_KEY = 'question_tokens'
QUESTION_KEY = 'question'
ANSWER_KEY = 'answer'
ANSWER_START_IDX_KEY = 'answer_start_idx'
ANSWER_END_IDX_KEY = 'answer_end_idx'


def cuda(args, tensor):
    """
    Places tensor on CUDA device (by default, uses cuda:0).

    Args:
        tensor: PyTorch tensor.

    Returns:
        Tensor on CUDA device.
    """
    if args.use_gpu and torch.cuda.is_available():
        return tensor.cuda(args.device)
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


def get_ngrams(tokens: Union[str, List[str]], n_gram: int) -> Counter:
    assert isinstance(n_gram, int) and n_gram >= 1
    ngrams_counts = Counter()
    if n_gram == 1:
        for token in tokens:
            ngrams_counts[token] += 1
        return ngrams_counts

    for i in range(len(tokens) - n_gram + 1):
        n_gram_slice = tuple(tokens[i: i + n_gram])
        ngrams_counts[n_gram_slice] += 1
    return ngrams_counts


def backtranslate_questions(train_dataset, batch_size, forward_translate, backward_translate, n=None):
    start = time.time()
    new_samples: List[Tuple] = []
    if n is None:
        n = len(train_dataset.samples)
    for batch_start in range(0, len(train_dataset.samples[:n]), batch_size):
        # build a batch as a list
        batch = []
        batched_questions = []
        for s in train_dataset.samples[batch_start:batch_start + batch_size]:
            batched_questions.append(' '.join(s[QUESTION_TOKENS_KEY]))
            batch.append(list(s))

        ## TODO @alex
        batched_backtrans_qs = backtranslate(batched_questions, forward_translate, backward_translate)

        for backtrans_q, new_sample in zip(batched_backtrans_qs, batch):
            new_sample[2] = backtrans_q.split()
            new_sample = tuple(new_sample)
        new_samples += batch
    # print(batch_start, time.time()-start)
    return new_samples


def backtranslate(sent, forward, backward):
    """
    Parameters
    ----------
    sent (str): sentence to be backtranslated
    forward (torch.nn): model to translate from langauge 1 to language 2
    backward (torch.nn): model to translate from langauge 2 to langauge 1
    
    Return:
    backtrans_sent: backtranslated sentence
    """
    forward = forward.eval()
    backward = backward.eval()
    return backward.translate(forward.translate(sent))


def load_translators(forward_model_path='transformer.wmt19.en-de.single_model',
                     backward_model_path='transformer.wmt19.de-en.single_model'):
    # Round-trip translations between English and German:
    forward = torch.hub.load('pytorch/fairseq', forward_model_path, tokenizer='moses', bpe='fastbpe')
    backward = torch.hub.load('pytorch/fairseq', backward_model_path, tokenizer='moses', bpe='fastbpe')

    return forward, backward


def set_root_logger(log_level=logging.INFO, log_format='%(message)s'):
    logging.basicConfig(format=log_format, level=log_level)
    ## Forcefully set logging level if overridden by other libraries. Ref: https://stackoverflow.com/a/46710435
    log_formatter = logging.Formatter(log_format)
    logging.getLogger().setLevel(log_level)
    for handler in logging.getLogger().handlers:
        handler.setFormatter(log_formatter)


def generate_random_string(max_num_chars=10) -> str:
    alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return "".join(np.random.choice(alphabet, max_num_chars))
