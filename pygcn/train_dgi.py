from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, binary_accuracy
from models import DGI

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--origin', action='store_true', default=True,
                    help='Keep the original implementation as the paper.')
parser.add_argument('--test_only', action="store_true", default=False,
                    help='Test on existing model')
parser.add_argument('--repeat', type=int, default=50,
                    help='number of experiments')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Show training process')
parser.add_argument('--split', type=str, default='equal',
                    help='Data split method')
parser.add_argument('--rho', type=float, default=0.1,
                    help='Adj matrix corruption rate')
parser.add_argument('--corruption', type=str, default='node_shuffle',
                    help='Corruption method')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj: sparse tensor, symmetric normalized Laplication
# A = D^(-1/2)*A*D^(1/2)
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, origin=args.origin, split=args.split)

if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(num_epoch, patience=30, verbose=False):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_epoch = -1
    for epoch in range(num_epoch):
        t = time.time()
        optimizer.zero_grad()
        outputs, labels = model(features, adj)
        if args.cuda:
            labels = labels.cuda()
        loss_train = F.binary_cross_entropy_with_logits(outputs, labels)
        acc_train = binary_accuracy(outputs, labels)
        loss_train.backward()
        optimizer.step()

        loss = loss_train.item()
        accuracy = acc_train.item()
        if verbose:
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss),
                  'acc_train: {:.4f}'.format(accuracy),
                  'time: {:.4f}s'.format(time.time() - t))

        # early stop
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
        if epoch == best_epoch + patience:
            break


def test(verbose=False):
    with torch.set_grad_enabled(False):
        model.eval()
        outputs = model(features, adj)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-2)
    for epoch in range(2000):
        optimizer.zero_grad()
        predictions = F.log_softmax(classifier(outputs), dim=1)
        loss_train = F.nll_loss(predictions[idx_train], labels[idx_train])
        acc_train = accuracy(predictions[idx_train], labels[idx_train])
        if verbose and epoch % 100 == 0:
            print("loss: {:.4f}".format(loss_train.item()),
                  "accuracy: {:.4f}".format(acc_train.item()))
        loss_train.backward()
        optimizer.step()

        predictions = F.log_softmax(classifier(outputs), dim=1)
    loss_test = F.nll_loss(predictions[idx_test], labels[idx_test])
    acc_test = accuracy(predictions[idx_test], labels[idx_test])
    print("Test set results:",
          "loss: {:.4f}".format(loss_test.item()),
          "accuracy: {:.4f}".format(acc_test.item()))
    return acc_test.item()


if __name__ == "__main__":
    results = []

    for i in range(args.repeat):
        # model
        model = DGI(num_feat=features.shape[1],
                    num_hid=args.hidden,
                    dropout=args.dropout,
                    rho=args.rho,
                    corruption=args.corruption)
        classifier = nn.Linear(args.hidden, labels.max().item() + 1)
        if args.cuda:
            model.cuda()
            classifier = classifier.cuda()

        print("----- %d / %d runs -----" % (i+1, args.repeat))
        # Train model
        t_total = time.time()
        if args.test_only:
            model = torch.load("model")
        else:
            train(args.epochs, verbose=args.verbose)
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            #torch.save(model, "model")

        # Test
        results.append(test(verbose=args.verbose))
    print("%d runs, mean: %g, var: %g" % (args.repeat, np.mean(results), np.std(results)))