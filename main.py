import argparse
import os
from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sk_normalize
from datetime import datetime
import pandas as pd
from logger import Logger
from dataset import load_nc_dataset
from data_utils import adj_neighbor_dist, evaluate_higcn, eval_acc
from parse import parse_method, parser_add_main_args
import faulthandler
faulthandler.enable()

Hi_path = "Hi_features"

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

device = torch.device(args.device)

### Load and preprocess data ###
dataset = load_nc_dataset(args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

split_idx_lst = [dataset.get_idx_split(split_type = args.split_type, train_prop=args.train_prop, valid_prop=args.valid_prop, run = i)
                    for i in range(args.runs)]

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def row_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm='l1', axis=1)
    return sp.coo_matrix(adj_normalized)


def precompute_degree_s(adj):
    adj_i = adj._indices()
    adj_v = adj._values()
    adj_diag_ind = (adj_i[0, :] == adj_i[1, :])
    adj_diag = adj_v[adj_diag_ind]
    v_new = torch.zeros_like(adj_v)
    for i in tqdm(range(adj_i.shape[1])):
        v_new[i] = adj_diag[adj_i[0, i]]/adj_v[i]-1
    degree_precompute = torch.sparse.FloatTensor(
        adj_i, v_new, adj.size())
    return degree_precompute


def get_adj_high(adj_low):
    adj_high = -adj_low + sp.eye(adj_low.shape[0])
    return adj_high


num_relations = None

x = dataset.graph['node_feat']
edge_index = dataset.graph['edge_index']
adj_low_pt = Hi_path + '/' + args.dataset + '_adj_low.pt'
adj_high_pt = Hi_path + '/' + args.dataset + '_adj_high.pt'
if os.path.exists(adj_low_pt) and os.path.exists(adj_high_pt):
    adj_low = torch.load(adj_low_pt)
    adj_high = torch.load(adj_high_pt)
else:
    adj_low = to_scipy_sparse_matrix(edge_index)
    adj_low = row_normalized_adjacency(adj_low)
    # print(adj_low)
    adj_high = get_adj_high(adj_low)
    # print(adj_high)
    adj_low = sparse_mx_to_torch_sparse_tensor(adj_low)
    adj_high = sparse_mx_to_torch_sparse_tensor(adj_high)
    # adj_high = (torch.eye(n) - adj_low).to_sparse()
    torch.save(adj_low, adj_low_pt)
    torch.save(adj_high, adj_high_pt)
x = x.to(device)
adj_low = adj_low.to(device)
adj_high = adj_high.to(device)

output_path = f'{Hi_path}/output/{args.dataset}/'
if args.method in ['higcn','sgc','sage','gat','mixhop','gcn']:
    assert(os.path.exists(output_path))
    adj_nd_low_list = []
    adj_nd_high_list = []
    for i in range(args.runs):
        output = torch.load(output_path+f'{i}_{args.runs}.pt')
        adj_nd = adj_neighbor_dist(output, dataset.label, adj_low.cpu(), het_threshold = args.het_threshold,drop_edge = args.drop_edge)
        adj_nd_low = adj_nd
        adj_nd_low = row_normalized_adjacency(adj_nd_low)
        adj_nd_high = get_adj_high(adj_nd_low)
        adj_nd_low = sparse_mx_to_torch_sparse_tensor(adj_nd_low)
        adj_nd_high = sparse_mx_to_torch_sparse_tensor(adj_nd_high)
        adj_nd_low_list.append(adj_nd_low)
        adj_nd_high_list.append(adj_nd_high)
        adj_ratio = (adj_low.to_dense()>0).sum()/(adj_nd_low.to_dense()>0).sum()
        print("{:.2f}".format(adj_ratio),end='|')

train_loader, subgraph_loader = None, None

print(f"num nodes {n} | num classes {c} | num node feats {d}")


### Load method ###

model = parse_method(args, dataset, n, c, d, args.device, num_relations)
# print('model', next(model.parameters()).device)


# using rocauc as the eval function
criterion = nn.NLLLoss()
eval_func = eval_acc

logger = Logger(args.runs, args)


model.train()
print('MODEL:', model)
### Training loop ###
last_time = time.time()

for run in range(args.runs):
    adj_nd_low = adj_nd_low_list[run].to(device)
    adj_nd_high = adj_nd_high_list[run].to(device)
    adj_low = adj_low.coalesce()
    adj_nd_low = adj_nd_low.coalesce()
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    if args.sampling:
        if args.num_layers == 2:
            sizes = [15, 10]
        elif args.num_layers == 3:
            sizes = [15, 10, 5]
        train_loader = NeighborSampler(dataset.graph['edge_index'], node_idx=train_idx,
                                       sizes=sizes, batch_size=1024,
                                       shuffle=True, num_workers=12)
        subgraph_loader = NeighborSampler(dataset.graph['edge_index'], node_idx=None, sizes=[-1],
                                          batch_size=4096, shuffle=False,
                                          num_workers=12)

    model.reset_parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('-inf')
    cost_val = []
    for epoch in range(args.epochs):
        model.train()

        optimizer.zero_grad()
        out = model(x, adj_low, adj_high, adj_nd_low, adj_nd_high)
        out = F.log_softmax(out, dim=1)
        loss = criterion(
            out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
        
        result = evaluate_higcn(model, x, adj_low, adj_high, adj_nd_low, adj_nd_high, dataset, split_idx, eval_func,
                                     sampling=args.sampling, subgraph_loader=subgraph_loader)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            if args.dataset != 'ogbn-proteins':
                best_out = F.softmax(result[-1], dim=1)
            else:
                best_out = result[-1]

        now_time = time.time()
        time_elapsed = now_time - last_time
        last_time = now_time

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Time: {time_elapsed:.4f}')
            if args.print_prop:
                pred = out.argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred.unique(
                    return_counts=True)[1].float()/pred.shape[0])
                
        loss_val = F.nll_loss(result[-1][split_idx['valid']],dataset.label.squeeze(1)[split_idx['valid']])
        cost_val.append(loss_val.item())
        if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping+1):-1]):
            print("Early stopping...")
            break

    logger.print_statistics(run)
    if args.save_output:
        if not os.path.exists(output_path): os.mkdir(output_path)
        torch.save(best_out.cpu(), output_path+f'{run}_{args.runs}.pt')

best_val, best_test = logger.print_statistics()

if args.save_result:
    result={'acc_val_mean':best_val.mean().item(),
            'acc_val_std':best_val.std().item(),
            'acc_test_mean':best_test.mean().item(),
            'acc_test_std':best_test.std().item(),
            'time_elapsed':time_elapsed,
            'adj_ratio':adj_ratio}
    result.update(vars(args))
    result['datetime'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    df = pd.DataFrame(columns=result.keys())
    df = df.append(result, ignore_index=True)
    if args.save_result_filename!="": save_path = f'results/{args.save_result_filename}.csv'
    elif args.method  in ['acmgcn2','higcn']: save_path = f'results/{args.method}.csv'
    else: save_path = f'results/{args.method}.csv'
    if os.path.exists(save_path):
        df.to_csv(save_path,mode='a',header=False) 
    else:
        df.to_csv(save_path,mode='w',header=True) 