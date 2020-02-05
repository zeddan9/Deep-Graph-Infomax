import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import numpy as np
import scipy.sparse as sp
from utils import to_sparse
from utils import sparse_mx_to_torch_sparse_tensor

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, origin=False):
        # origin: keep original structure as the paper
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        ###
        self.origin = origin

    def forward(self, x, adj):
        if self.origin:
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class DGI(nn.Module):
    def __init__(self, num_feat, num_hid, dropout, rho=0.1, readout="average", corruption="node_shuffle"):
        super(DGI, self).__init__()

        self.gc = GraphConvolution(num_feat, num_hid)
        self.fc = nn.Linear(num_hid, num_hid, bias=False)
        self.dropout = dropout
        self.prelu = nn.PReLU()

        self.rho = rho

        self.readout = getattr(self, "_%s" % readout)
        self.corruption = getattr(self, "_%s" % corruption)

    def forward(self, X, A):
        x = F.dropout(X, self.dropout, training=self.training)
        H = self.prelu(self.gc(x, A))
        if not self.training:
            return H

        neg_X, neg_A = self.corruption(X, A)
        x = F.dropout(neg_X, self.dropout, training=self.training)
        neg_H = self.prelu(self.gc(x, neg_A))

        s = self.readout(H)
        x = self.fc(s)
        x = torch.mv(torch.cat((H, neg_H)), x)
        labels = torch.cat((torch.ones(X.size(0)), torch.zeros(neg_X.size(0))))
        return x, labels

    def _average(self, features):
        x = features.mean(0)
        return F.sigmoid(x)

    def _node_shuffle(self, X, A):
        perm = torch.randperm(X.size(0))
        neg_X = X[perm]
        return neg_X, A

    def _adj_corrupt(self, X, A):
        rho = self.rho
        [n, m] = A.shape
        neg_A = A.clone()
        p = np.random.rand(n, m)
        d_A = np.zeros((n, m))
        d_A[p < rho] = 1
        neg_A = np.logical_xor(neg_A.to_dense().data.cpu().numpy(), d_A)
        idx = np.nonzero(neg_A)
        d_A = torch.sparse.FloatTensor(torch.LongTensor(np.array(idx)), torch.FloatTensor(np.ones(len(idx[0]))) , \
                                       torch.Size([n, m])).cuda()
        return X, d_A
