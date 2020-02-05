import os
import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset="cora", path="../data/", origin=False, split="equal"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    raw_data = np.genfromtxt(os.path.join(path, dataset, "%s.content" % dataset),
                            dtype=np.str)
    raw_idxs, raw_features, raw_labels = raw_data[:, 0], raw_data[:, 1:-1], raw_data[:, -1]
    features = sp.csr_matrix(raw_features, dtype=np.float32)
    labels = encode_onehot(raw_labels)

    # build graph
    edges_unordered = np.genfromtxt(os.path.join(path, dataset, "%s.cites" % dataset),
                                    dtype=np.str)
    idx_map = {j: i for i, j in enumerate(raw_idxs)}
    idx_edges = []
    invalid = 0
    for u, v in edges_unordered:
        if u not in idx_map or v not in idx_map:
            invalid += 1
            continue
        idx_edges.append([idx_map[u], idx_map[v]])
    print("%d / %d invalid edges" % (invalid, len(edges_unordered)))
    edges = np.array(idx_edges, dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    if origin:
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    else:
        # this normalization method introduces gains
        adj = normalize(adj + sp.eye(adj.shape[0]))

    if split == "fix":
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    elif split == "equal":
        idx_train = []
        for label in set(raw_labels):
            idxes = np.where(raw_labels == label)[0]
            idx_train += list(np.random.choice(idxes, 20, replace=False))
        rest = list(set(range(len(raw_labels))) - set(idx_train))
        idxes = np.random.choice(rest, 300 + 1000, replace=False)
        idx_val, idx_test = idxes[:300], idxes[300:]
    elif split == "random":
        all = range(len(raw_labels))
        idxes = np.random.choice(all, 140 + 300 + 1000)
        idx_train, idx_val, idx_test = idxes[:140], idxes[140:-1000], idxes[-1000:]
    else:
        raise ValueError("Unknown split type `%s`" % split)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def accuracy(outputs, labels):
    preds = outputs.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def binary_accuracy(outputs, labels):
    preds = outputs.gt(0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
