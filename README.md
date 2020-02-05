Deep Graph Infomax Replica

##

please create your own branch, and make merge request if you have new features to add.

## Requirements

  * PyTorch
  * Python 3.6

## Usage

```
cd pygcn
python train_gcn.py
python train_dgi.py
```

## Experiments

Numbers in parenthesis denotes the performance reported in the original paper.

### Equally sampled training set

| method  | cora         |
|---------|--------------|
| GCN     | 80.60 (81.5) |
| GCN*    | 78.90        |
| DGI     | 81.60 (82.3) |
| DGI*    | 80.20        |

### Randomly sampled training set

| method  | cora         |
|---------|--------------|
| GCN     | 80.30 (80.1) |
| GCN*    | 81.60        |
| DGI     | 83.70        |
| DGI*    | 83.30        |

* \* indicates a different normalization strategy for adjacency matrix.

***Zhu: I don't think the different normalization strategy works.***

## tasks
- [x] gcn replica
- [x] infomax loss
- [ ] ablations 

