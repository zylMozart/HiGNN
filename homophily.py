import numpy as np
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops

def attribute_homophily(edge_index, feat, node_nums):
    edge_index = torch.cat([edge_index,edge_index[[1,0],:]],dim=1)
    hom = ((feat[edge_index[0]]==feat[edge_index[1]])).sum(dim=1)/feat.shape[1]
    return float(hom.mean())

def neighbor_distrbution_homophily(edge_idx, labels, het_threshold=1.0, khop=1):
    num_nodes =  labels.shape[0]
    num_labels = labels.unique().shape[0]
    # Get new adj
    hetero = torch.zeros((num_nodes,num_labels))
    preds = labels+1 # Avoid 0 in the next matrix mul
    adj_pred = torch.zeros((num_nodes,num_nodes))
    adj_pred[edge_idx[0,:],edge_idx[1,:]]=1
    adj_pred[edge_idx[1,:],edge_idx[0,:]]=1
    adj_pred = adj_pred.fill_diagonal_(1)
    for i in range(khop-1):
        adj_pred = torch.matmul(adj_pred,adj_pred)
    adj_pred[adj_pred>0]=1
    adj_pred[adj_pred==0]=-1
    adj_pred = adj_pred.fill_diagonal_(-1)
    adj_pred = adj_pred * preds.repeat(num_nodes, 1)
    for i_label in range(num_labels):
        hetero[:,i_label] = (adj_pred==(i_label+1)).sum(dim=1)
    hetero = hetero / hetero.norm(dim=1)[:, None]
    similarity_matrix = torch.mm(hetero, hetero.t())
    CL_adj = (similarity_matrix>=het_threshold).int()
    CL_adj = CL_adj.fill_diagonal_(0)

    edge_idx = CL_adj.nonzero()
    src_label = labels[edge_idx[:,0]]
    targ_label = labels[edge_idx[:,1]]
    labeled_edges = (src_label >= 0) * (targ_label >= 0)
    return torch.mean((src_label[labeled_edges] == targ_label[labeled_edges]).float())
    pass

def edge_homophily(A, labels, ignore_negative=False):
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    if ignore_negative = True, then only compute for edges where nodes both have
        nonnegative class labels (negative class labels are treated as missing
    """
    src_node, targ_node = A.nonzero()
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    if ignore_negative:
        edge_hom = np.mean(matching[labeled_mask])
    else:
        edge_hom = np.mean(matching)
    return edge_hom

def compat_matrix(A, labels):
    """ c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     See Zhu et al. 2020
    """
    c = len(np.unique(labels))
    H = np.zeros((c,c))
    src_node, targ_node = A.nonzero()
    for i in range(len(src_node)):
        src_label = labels[src_node[i]]
        targ_label = labels[targ_node[i]]
        H[src_label, targ_label] += 1
    H = H / np.sum(H, axis=1, keepdims=True)
    return H

def node_homophily(A, labels):
    """ average of homophily for each node
    """
    src_node, targ_node = A.nonzero()
    edge_idx = torch.tensor(np.vstack((src_node, targ_node)), dtype=torch.long).contiguous()
    labels = torch.tensor(labels)
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)

def edge_homophily_edge_idx(edge_idx, labels):
    """ edge_idx is 2x(number edges) """
    edge_index = remove_self_loops(edge_idx)[0]
    # treat negative edges
    src_label = labels[edge_index[0,:]]
    targ_label = labels[edge_index[1,:]]
    labeled_edges = (src_label >= 0) * (targ_label >= 0)
    return torch.mean((src_label[labeled_edges] == targ_label[labeled_edges]).float())

def class_homophily_edge_idx(edge_idx, labels):
    """ edge_idx is 2x(number edges) """
    num_nodes = labels.shape[0]
    edge_index = remove_self_loops(edge_idx)[0]
    # treat negative edges
    src_label = labels[edge_index[0,:]]
    targ_label = labels[edge_index[1,:]]
    A = torch.zeros((num_nodes,num_nodes))
    A[edge_index[0,:],edge_index[1,:]]=1
    A[edge_index[1,:],edge_index[0,:]]=1
    labeled_edges = (src_label >= 0) * (targ_label >= 0)
    sum = 0
    for c in range(len(labels.unique())):
        hetero_ratio = (torch.logical_and(src_label==c,targ_label==c)).sum()/(torch.logical_or(src_label==c,targ_label==c)).sum()
        class_ratio = (labels==c).sum()/num_nodes
        sum+=max(0,hetero_ratio-class_ratio)
        pass
    return sum/(len(labels.unique())-1)

def edge_homophily_edge_idx2(edge_idx, labels):
    """ edge_idx is 2x(number edges) """
    # treat negative edges
    src_label = labels[edge_idx[:,0]]
    targ_label = labels[edge_idx[:,1]]
    labeled_edges = (src_label >= 0) * (targ_label >= 0)
    return torch.mean((src_label[labeled_edges] == targ_label[labeled_edges]).float())

def node_homophily_edge_idx(edge_idx, labels, num_nodes):
    """ edge_idx is 2 x(number edges) """
    edge_index = remove_self_loops(edge_idx)[0]
    hs = torch.zeros(num_nodes)
    degs = torch.bincount(edge_index[0,:]).float()
    matches = (labels[edge_index[0,:]] == labels[edge_index[1,:]]).float()
    hs = hs.scatter_add(0, edge_index[0,:], matches) / degs
    return hs[degs != 0].mean()
    
def compat_matrix_edge_idx(edge_idx, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    edge_index = remove_self_loops(edge_idx)[0]
    src_node, targ_node = edge_index[0,:], edge_index[1,:]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max()+1
    H = torch.zeros((c,c)).to(edge_index.device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    label_idx = torch.cat((src_label.unsqueeze(0), targ_label.unsqueeze(0)), axis=0)
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx).to(H.dtype), add_idx, out=H[k,:], dim=-1)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H

def our_measure(edge_index, label):
    """ 
    our measure \hat{h}
    treats negative labels as unlabeled 
    """
    label = label.squeeze()
    c = label.max()+1
    H = compat_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k,k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c-1
    return val

