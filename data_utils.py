import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score, f1_score
from torch_sparse import SparseTensor
from google_drive_downloader import GoogleDriveDownloader as gdd


def adj_ablation_node_degree(feat, labels, adj, soft_threshold):
    num_nodes =  labels.shape[0]
    num_labels = labels.unique().shape[0]
    preds = (adj.to_dense()>0).int().sum(dim=1)
    abla_adj = (preds.repeat(num_nodes,1)==preds.repeat(num_nodes,1).t())
    abla_adj = abla_adj.int().fill_diagonal_(1)
    return abla_adj

def adj_ablation_feature_cos(feat, labels, adj, soft_threshold):
    num_nodes =  labels.shape[0]
    num_labels = labels.unique().shape[0]
    output = feat.cpu().clone()
    output = output / output.norm(dim=1)[:, None]
    similarity_matrix = torch.mm(output, output.t())
    abla_adj = (similarity_matrix>=soft_threshold).int()
    abla_adj = abla_adj.fill_diagonal_(1)
    return abla_adj

def adj_ablation_feature_kmeans(feat, labels, adj, n_cluster):
    from sklearn.cluster import KMeans
    num_nodes =  labels.shape[0]
    num_labels = labels.unique().shape[0]
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(feat.cpu().clone().numpy())
    preds = torch.tensor(kmeans.labels_)
    abla_adj = (preds.repeat(num_nodes,1)==preds.repeat(num_nodes,1).t())
    abla_adj = abla_adj.int().fill_diagonal_(1)
    return abla_adj

def adj_ablation_feature_kmeans(feat, labels, adj, n_cluster):
    from sklearn.cluster import KMeans
    num_nodes =  labels.shape[0]
    num_labels = labels.unique().shape[0]
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(feat.cpu().numpy())
    preds = torch.tensor(kmeans.labels_)
    abla_adj = (preds.repeat(num_nodes,1)==preds.repeat(num_nodes,1).t())
    abla_adj = abla_adj.int().fill_diagonal_(1)
    return abla_adj

def adj_ablation_label_soft(output, labels, adj, soft_threshold):
    num_nodes =  labels.shape[0]
    num_labels = labels.unique().shape[0]
    output = output / output.norm(dim=1)[:, None]
    similarity_matrix = torch.mm(output, output.t())
    abla_adj = (similarity_matrix>=soft_threshold).int()
    abla_adj = abla_adj.fill_diagonal_(1)
    return abla_adj

def adj_ablation_label_hard(output, labels, adj, het_threshold):
    num_nodes =  labels.shape[0]
    num_labels = labels.unique().shape[0]
    preds = output.max(1)[1]+1 # Avoid 0 in the next matrix mul
    abla_adj = (preds.repeat(num_nodes,1)==preds.repeat(num_nodes,1).t())
    abla_adj = abla_adj.int().fill_diagonal_(1)
    return abla_adj

def bin_adj_neighbor_dist(output, labels, adj, bin_topk=0.9, bin_span=0.1):
    num_nodes =  labels.shape[0]
    num_labels = labels.unique().shape[0]
    # 1. Define hetero graph
    hetero = torch.zeros((num_nodes,num_labels))
    preds = output.max(1)[1]+1 # Avoid 0 in the next matrix mul
    adj_pred = adj.clone().to_dense()
    adj_pred[adj_pred>0]=1
    adj_pred[adj_pred==0]=-1
    adj_pred = adj_pred * preds.repeat(num_nodes, 1)
    for i_label in range(num_labels):
        hetero[:,i_label] = (adj_pred==(i_label+1)).sum(dim=1)
    hetero = hetero / hetero.norm(dim=1)[:, None]
    similarity_matrix = torch.mm(hetero, hetero.t())
    similarity_matrix = similarity_matrix.fill_diagonal_(0)
    rank = similarity_matrix.sort(dim=1)[1][:,int(num_nodes*(bin_topk)):int(num_nodes*(bin_topk+bin_span))]
    rank_row = torch.arange(num_nodes).unsqueeze(1).repeat(1,rank.shape[1])
    CL_adj = torch.zeros((num_nodes,num_nodes))
    CL_adj[rank_row.reshape(-1),rank.reshape(-1)]=1
    CL_adj = ((CL_adj + CL_adj.t())>0).int()
    CL_adj = CL_adj.fill_diagonal_(1)
    ##################!!!!!!!!!!!!!!!!!!!!!!
    # CL_adj = (torch.rand(num_nodes,num_nodes)>0.9).int()
    return CL_adj

def adj_neighbor_dist(output, labels, adj, het_threshold, drop_edge):
    num_nodes =  labels.shape[0]
    num_labels = labels.unique().shape[0]
    # 1. Define hetero graph
    hetero = torch.zeros((num_nodes,num_labels))
    preds = output.max(1)[1]+1 # Avoid 0 in the next matrix mul
    adj_pred = adj.clone().to_dense()
    adj_pred[adj_pred>0]=1
    adj_pred[adj_pred==0]=-1
    adj_pred = adj_pred * preds.repeat(num_nodes, 1)
    for i_label in range(num_labels):
        hetero[:,i_label] = (adj_pred==(i_label+1)).sum(dim=1)
    hetero = hetero / hetero.norm(dim=1)[:, None]
    similarity_matrix = torch.mm(hetero, hetero.t())
    #### Get topk in 2d matrix
    CL_adj = (similarity_matrix>=het_threshold).int()
    CL_adj = CL_adj.fill_diagonal_(1)
    # num_edge = int((adj.to_dense()>0).sum())
    # v, i = torch.topk(similarity_matrix.flatten(), num_edge)
    # res = np.array(np.unravel_index(i.numpy(), similarity_matrix.shape)).T
    # CL_adj = torch.zeros_like(similarity_matrix)
    # CL_adj[res[:,0],res[:,1]]=1
    if drop_edge>0:
        scale = CL_adj.sum()/torch.sparse.sum(adj)
        drop_rate = drop_edge/scale
        if drop_rate<1:
            mask = torch.bernoulli(torch.ones_like(CL_adj)*drop_rate)
            CL_adj = mask*CL_adj
    return CL_adj

def sim_neighbor_dist(output, labels, adj):
    num_nodes =  labels.shape[0]
    num_labels = labels.unique().shape[0]
    # 1. Define hetero graph
    hetero = torch.zeros((num_nodes,num_labels))
    preds = output.max(1)[1]+1 # Avoid 0 in the next matrix mul
    adj_pred = adj.clone().to_dense()
    adj_pred[adj_pred>0]=1
    adj_pred[adj_pred==0]=-1
    adj_pred = adj_pred * preds.repeat(num_nodes, 1)
    for i_label in range(num_labels):
        hetero[:,i_label] = (adj_pred==(i_label+1)).sum(dim=1)
    hetero = hetero / hetero.norm(dim=1)[:, None]
    similarity_matrix = torch.mm(hetero, hetero.t())
    return similarity_matrix

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def to_planetoid(dataset):
    """
        Takes in a NCDataset and returns the dataset in H2GCN Planetoid form, as follows:

        x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ty => the one-hot labels of the test instances as numpy.ndarray object;
        ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        split_idx => The ogb dictionary that contains the train, valid, test splits
    """
    split_idx = dataset.get_idx_split('random', 0.25)
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    graph, label = dataset[0]

    label = torch.squeeze(label)

    print("generate x")
    x = graph['node_feat'][train_idx].numpy()
    x = sp.csr_matrix(x)

    tx = graph['node_feat'][test_idx].numpy()
    tx = sp.csr_matrix(tx)

    allx = graph['node_feat'].numpy()
    allx = sp.csr_matrix(allx)

    y = F.one_hot(label[train_idx]).numpy()
    ty = F.one_hot(label[test_idx]).numpy()
    ally = F.one_hot(label).numpy()

    edge_index = graph['edge_index'].T

    graph = defaultdict(list)

    for i in range(0, label.shape[0]):
        graph[i].append(i)

    for start_edge, end_edge in edge_index:
        graph[start_edge.item()].append(end_edge.item())

    return x, tx, allx, y, ty, ally, graph, split_idx


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t


def normalize(edge_index):
    """ normalizes the edge_index
    """
    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def gen_normalized_adjs(dataset):
    """ returns the normalized adjacency matrix
    """
    row, col = dataset.graph['edge_index']
    N = dataset.graph['num_nodes']
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0

    DAD = D_isqrt.view(-1, 1) * adj * D_isqrt.view(1, -1)
    DA = D_isqrt.view(-1, 1) * D_isqrt.view(-1, 1) * adj
    AD = adj * D_isqrt.view(1, -1) * D_isqrt.view(1, -1)
    return DAD, DA, AD


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

@torch.no_grad()
def evaluate_higcn(model, x, adj_low, adj_high, adj_nd_low, adj_nd_high, dataset, split_idx, eval_func, result=None, sampling=False, subgraph_loader=None):
    if result is not None:
        out = result
    else:
        model.eval()
        if not sampling:
            out = model(x, adj_low, adj_high, adj_nd_low, adj_nd_high)
        else:
            out = model.inference(dataset, subgraph_loader)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    return train_acc, valid_acc, test_acc, out

def load_fixed_splits(dataset, sub_dataset):
    """ loads saved fixed splits for dataset
    """
    name = dataset
    if sub_dataset and sub_dataset != 'None':
        name += f'-{sub_dataset}'

    if not os.path.exists(f'./data/splits/{name}-splits.npy'):
        assert dataset in splits_drive_url.keys()
        gdd.download_file_from_google_drive(
            file_id=splits_drive_url[dataset],
            dest_path=f'./data/splits/{name}-splits.npy', showsize=True)

    splits_lst = np.load(f'./data/splits/{name}-splits.npy', allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst


dataset_drive_url = {
    'twitch-gamer_feat': '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges': '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',  # Wiki 1.9M
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',  # Wiki 1.9M
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'  # Wiki 1.9M
}

splits_drive_url = {
    'snap-patents': '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
    'pokec': '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}
