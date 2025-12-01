import torch
import numpy as np
import collections
from sklearn.preprocessing import label_binarize
from scipy.stats import rankdata
import Constants

def precision_at_k(relevance_score, k):
    """ Precision at K given binary relevance scores. """
    assert k >= 1
    relevance_score = np.asarray(relevance_score)[:k] != 0
    if relevance_score.size != k:
        raise ValueError('Relevance score length < K')
    return np.mean(relevance_score)


def recall_at_k(relevance_score, k, m):
    """ Recall at K given binary relevance scores. """
    assert k >= 1
    relevance_score = np.asarray(relevance_score)[:k] != 0
    if relevance_score.size != k:
        raise ValueError('Relevance score length < K')
    return np.sum(relevance_score) / float(m)



def mean_precision_at_k(relevance_scores, k):
    """ Mean Precision at K given binary relevance scores. """
    mean_p_at_k = np.mean(
        [precision_at_k(r, k) for r in relevance_scores]).astype(np.float32)
    return mean_p_at_k



def mean_recall_at_k(relevance_scores, k, m_list):
    """ Mean Recall at K:  m_list is a list containing # relevant target entities for each data point. """
    mean_r_at_k = np.mean([recall_at_k(r, k, M) for r, M in zip(relevance_scores, m_list)]).astype(np.float32)
    return mean_r_at_k


def average_precision(relevance_score, K, m):
    """ For average precision, we use K as input since the number of prediction targets is not fixed
    unlike standard IR evaluation. """
    r = np.asarray(relevance_score) != 0
    out = [precision_at_k(r, k + 1) for k in range(0, K) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / float(min(K, m))


def MAP(relevance_scores, k, m_list):
    """ Mean Average Precision -- MAP. """
    map_val = np.mean([average_precision(r, k, M) for r, M in zip(relevance_scores, m_list)]).astype(np.float32)
    return map_val


def MRR(relevance_scores):
    """ Mean reciprocal rank -- MRR. """
    rs = (np.asarray(r).nonzero()[0] for r in relevance_scores)
    mrr_val = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs]).astype(np.float32)
    return mrr_val


def get_masks(top_k, inputs):
    """ Mask the dummy sequences  -- : 0 if .. 1 if seed set is of size > 1. """
    masks = []
    for i in range(0, top_k.shape[0]):
        seeds = set(inputs[i])
        if len(seeds) == 1 and list(seeds)[0] == 0:
            masks.append(0)
        else:
            masks.append(1)
    return np.array(masks).astype(np.int32)


def remove_seeds(top_k, inputs):
    """ Replace seed users from top-k predictions with -1. """
    result = []
    for i in range(0, top_k.shape[0]):
        seeds = set(inputs[i])
        lst = list(top_k[i])  # top-k predicted users.
        for s in seeds:
            if s in lst:
                lst.remove(s)
        for k in range(len(top_k[i]) - len(lst)):
            lst.append(-1)
        result.append(lst)
    return np.array(result).astype(np.int32)


def get_relevance_scores(top_k_filter, targets):
    """ Create binary relevance scores by checking if the top-k predicted users are in target set. """
    output = []
    for i in range(0, top_k_filter.shape[0]):
        z = np.isin(top_k_filter[i], targets[i])
        output.append(z)
    return np.array(output)


def one_hot(values, num_classes):
    batch_size = values.shape[0]
    seq_len = values.shape[1]
    result = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        for j in range(seq_len):
            if values[i][j] != -1:
                result[i][values[i][j]] = 1
            else:
                continue
    return result.long()

def masked_select(inputs, masks):
    result = []
    for i, mask in enumerate(masks, 0):
        if mask == 1:
            result.append(inputs[i])
    return np.array(result).astype(np.int32)


def _retype(y_prob, y):
    if not isinstance(y, (collections.Sequence, np.ndarray)):
        y_prob = [y_prob]
        y = [y]
    y_prob = np.array(y_prob)
    y = np.array(y)

    return y_prob, y

def _binarize(y, n_classes=None):
    return label_binarize(y, classes=range(n_classes))

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(y_prob, y, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    predicted = [np.argsort(p_)[-k:][::-1] for p_ in y_prob]
    actual = [[y_] for y_ in y]
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def mean_rank(y_prob, y):
    ranks = []
    n_classes = y_prob.shape[1]
    for p_, y_ in zip(y_prob, y):
        ranks += [n_classes - rankdata(p_, method='max')[y_]]

    return sum(ranks) / float(len(ranks))


def hits_k(y_prob, y, k=10):
    acc = []
    for p_, y_ in zip(y_prob, y):
        top_k = p_.argsort()[-k:][::-1]
        acc += [1. if y_ in top_k else 0.]
    return sum(acc) / len(acc)

def portfolio(pred, gold, k_list=[1,5,10,20]):
    scores_len = 0
    y_prob=[]
    y=[]
    for i in range(gold.shape[0]): # predict counts
        if gold[i]!=Constants.PAD:
            scores_len+=1.0
            y_prob.append(pred[i])
            y.append(gold[i])
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = hits_k(y_prob, y, k=k)
        scores['map@' + str(k)] = mapk(y_prob, y, k=k)

    return scores, scores_len

def compute_metric(y_prob, y_true, k_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    '''
		y_true: (#samples, )
		y_pred: (#samples, #users)
	'''
    scores_len = 0
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)

    scores = {'hits@' + str(k): [] for k in k_list}
    scores.update({'map@' + str(k): [] for k in k_list})
    for p_, y_ in zip(y_prob, y_true):
        # if y_ != self.PAD:
        if y_ != 0:
            scores_len += 1.0
            p_sort = p_.argsort()
            for k in k_list:
                topk = p_sort[-k:][::-1]
                scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
                scores['map@' + str(k)].extend([apk([y_], topk, k)])

    scores = {k: np.mean(v) for k, v in scores.items()}
    return scores, scores_len