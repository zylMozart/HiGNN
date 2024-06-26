from numpy import select
from models import HIGCN
from models_baseline_hi import SGC, Sage, GAT, MixHop, GCN, MLP
from data_utils import normalize
import math
from os import path

DATAPATH = path.dirname(path.abspath(__file__)) + '/data/'

def parse_method(args, dataset, n, c, d, device, num_relations):
    if args.method == 'gcn':
        model = GCN(in_channels=d,hidden_channels=args.hidden_channels,
                    out_channels=c, nd_lambda=args.nd_lambda,
                    num_layers=args.num_layers,dropout=args.dropout,use_bn=not args.no_bn).to(device)
    elif args.method == 'mlp':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=args.num_layers,
                    dropout=args.dropout).to(device)
    elif args.method == 'sgc':
        model = SGC(in_channels=d,hidden_channels=args.hidden_channels,
                        out_channels=c, nd_lambda=args.nd_lambda,
                        num_layers=args.num_layers,dropout=args.dropout,use_bn=not args.no_bn).to(device)
    elif args.method == 'sage':
        model = Sage(in_channels=d, hidden_channels=args.hidden_channels,
                     out_channels=c, nd_lambda=args.nd_lambda).to(device)
    elif args.method == 'gat':
        model = GAT(in_channels=d, hidden_channels=args.hidden_channels,
                     out_channels=c, nd_lambda=args.nd_lambda,
                    dropout=args.dropout, heads=args.gat_heads).to(device)
    elif args.method == 'mixhop':
        model = MixHop(in_channels=d, hidden_channels=args.hidden_channels,
                     out_channels=c, nd_lambda=args.nd_lambda,
                    dropout=args.dropout).to(device)
    elif args.method == 'higcn':
        model = HIGCN(nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout,
                       model_type=args.model_type, nlayers=args.num_layers, variant=False, nd_lambda=args.nd_lambda, abla_type=args.abla_type).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    '''Cora CiteSeer PubMed chameleon cornell film squirrel texas wisconsin'''
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--method', '-m', type=str, default='higcn')
    parser.add_argument('--split_type', type=str, default='public')
    parser.add_argument('--runs', type=int, default=10,help='number of distinct runs')
    parser.add_argument('--early_stopping', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)

    # acmgcn2
    parser.add_argument('--save_output', type=int, default=0, help="acmgcn method only. Save output for calculating neighbor disctribution")
    parser.add_argument('--save_result', type=int, default=0)
    parser.add_argument('--save_result_filename', type=str, default="")
    parser.add_argument('--abla_type', type=str, default="HIGCN_wo_ori")
    parser.add_argument('--het_threshold', type=float, default=0.999, help='threshold for nd adj')
    parser.add_argument('--nd_lambda', type=float, default=0.01, help='lambda for fusing adj_nd and adj')
    parser.add_argument('--drop_edge', type=int, default=0, help='drop edge to lower density')

    parser.add_argument('--mlpkhop', type=int, default=2)

    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--display_step', type=int,
                        default=50, help='how often to print')
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers in h2gcn')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--no_bn', action='store_true',
                        help='do not use batchnorm')
    parser.add_argument('--sampling', action='store_true',
                        help='use neighbor sampling')

    # used for acmgcn
    parser.add_argument('--model_type', type=str, default='acmgcn')