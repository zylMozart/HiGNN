from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_sparse import masked_select_nnz
from torch import Tensor
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter as Param
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import SGConv, GCNConv, GATConv, SAGEConv, JumpingKnowledge, APPNP, GCN2Conv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy.sparse
from tqdm import tqdm
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math
import sys
from typing import Union, Tuple, Optional
from sklearn.preprocessing import normalize as sk_normalize
import scipy.sparse as sp
from torch.nn.modules.module import Module
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing



class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    
class SGC(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nd_lambda,
                 num_layers=2,dropout=0.5, save_mem=False, use_bn=True):
        """ takes 'hops' power of the normalized adjacency"""
        super(SGC, self).__init__()
        self.nd_lambda = nd_lambda
        self.convs = nn.ModuleList()
        self.convs.append(SGConv(in_channels, hidden_channels))
        # self.convs.append(SGConv(hidden_channels, out_channels))
        self.hiconvs = nn.ModuleList()
        self.hiconvs.append(SGConv(in_channels, hidden_channels))
        # self.hiconvs.append(SGConv(hidden_channels, out_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.nd_lambda = nd_lambda
        self.num_layers = num_layers
        self.activation = F.relu
        self.use_bn = use_bn
        self.lin = Linear(hidden_channels, out_channels, bias=True)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for conv in self.hiconvs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_low, adj_high, adj_nd_low, adj_nd_high):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers-1):
            if self.nd_lambda==0:
                x = self.convs[i](x, adj_low)
            else:
                x = self.convs[i](x, adj_low) + self.nd_lambda*self.hiconvs[i](x, adj_nd_low)
            # if self.use_bn:
            #     x = self.bns[i](x)
            # x = self.activation(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, adj_low) + self.nd_lambda*self.hiconvs[-1](x, adj_nd_low)
        x = self.lin(x)
        return x
    
class Sage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nd_lambda):
        super(Sage, self).__init__()
        self.nd_lambda = nd_lambda
        self.conv = SAGEConv(in_channels, hidden_channels)
        self.hiconv = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.hiconv2 = SAGEConv(hidden_channels, out_channels)
        
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.hiconv.reset_parameters()

    def forward(self, x, adj_low, adj_high, adj_nd_low, adj_nd_high):
        out = self.conv(x, adj_low) + self.nd_lambda*self.hiconv(x, adj_nd_low)
        out = self.conv2(out, adj_low) + self.nd_lambda*self.hiconv2(out, adj_nd_low)
        return out

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nd_lambda,
                 dropout=0.5, heads=2, sampling=False, add_self_loops=True):
        super(GAT, self).__init__()

        self.nd_lambda = nd_lambda
        self.convs = nn.ModuleList()
        self.hiconvs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))
        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))
        self.hiconvs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))
        self.hiconvs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))


        self.dropout = dropout
        self.activation = F.elu
        self.sampling = sampling

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for conv in self.hiconvs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_low, adj_high, adj_nd_low, adj_nd_high):
        if self.nd_lambda==0:
            out = self.convs[0](x, adj_low)
        else:
            out = self.convs[0](x, adj_low) + self.nd_lambda*self.hiconvs[0](x, adj_nd_low)
        out = self.bns[0](out)
        out = self.activation(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        if self.nd_lambda==0:
            out = self.convs[1](out, adj_low)
        else:
            out = self.convs[1](out, adj_low) + self.nd_lambda*self.hiconvs[1](out, adj_nd_low)
        
        return out

class MixHopLayer(nn.Module):
    """ Our MixHop layer """

    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops+1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x)]
        for j in range(1, self.hops+1):
            # less runtime efficient but usually more memory efficient to mult weight matrix first
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = adj_t@x_j
            xs += [x_j]
        return torch.cat(xs, dim=1)


class MixHop(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nd_lambda,
                  num_layers=2, dropout=0.5, hops=2):
        super(MixHop, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))
        self.hiconvs = nn.ModuleList()
        self.hiconvs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))
        for _ in range(num_layers - 2):
            self.convs.append(
                MixHopLayer(hidden_channels*(hops+1), hidden_channels, hops=hops))
            self.hiconvs.append(
                MixHopLayer(hidden_channels*(hops+1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))

        self.convs.append(
            MixHopLayer(hidden_channels*(hops+1), out_channels, hops=hops))
        self.hiconvs.append(
            MixHopLayer(hidden_channels*(hops+1), out_channels, hops=hops))

        # note: uses linear projection instead of paper's attention output
        self.final_project = nn.Linear(out_channels*(hops+1), out_channels)
        self.num_layers = num_layers
        self.dropout = dropout
        self.nd_lambda = nd_lambda
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for conv in self.hiconvs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, x, adj_low, adj_high, adj_nd_low, adj_nd_high):
        n = x.shape[0]

        for i in range(self.num_layers-1):
            x = self.convs[i](x, adj_low) + self.nd_lambda*self.convs[i](x, adj_nd_low)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_low) + self.nd_lambda*self.convs[-1](x, adj_nd_low)

        x = self.final_project(x)
        return x
    
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nd_lambda,
                 num_layers=2,dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()

        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
        self.hiconvs = nn.ModuleList()
        self.hiconvs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.hiconvs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
        self.hiconvs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.nd_lambda = nd_lambda
        self.num_layers = num_layers
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for conv in self.hiconvs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_low, adj_high, adj_nd_low, adj_nd_high):
        for i in range(self.num_layers-1):
            x = self.convs[i](x, adj_low) + self.nd_lambda*self.hiconvs[i](x, adj_nd_low)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_low) + self.nd_lambda*self.hiconvs[-1](x, adj_nd_low)
        return x