import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
from dataset import load_nc_dataset
from data_utils import adj_neighbor_dist, sim_neighbor_dist
from homophily import *

datasets = 'Cora CiteSeer PubMed chameleon cornell film squirrel texas wisconsin'.split(' ')

def read_paper_report_result():
    df = pd.read_csv('large-scale/results/paper_report_result.csv')
    num_method, num_dataset = df.shape
    for i in range(0,num_method):
        for j in range(1,num_dataset):
            df.iloc[i,j] = float(df.iloc[i,j].split('+')[0])
    df.index = df.iloc[:,0]
    df = df.drop(columns=[df.keys()[0]])
    df = df.reindex(columns = datasets)
    return df

def compare_with_paper():
    method = 'wrgat'
    data = pd.read_csv('large-scale/results/acmgcn.csv')
    report_all = read_paper_report_result()
    # acmgcn2_1 = pd.read_csv('large-scale/results/acmgcn2_1.0.csv')
    # acmgcn2_09 = pd.read_csv('large-scale/results/acmgcn2_0.9.csv')
    idx = report_all[report_all['Unnamed: 0']==method].index.values
    report = report_all.iloc[idx,:]
    print('{:<10} | {:<8} {} {} {}'.format('Dataset','Method','Result','Report','Diff'))
    for dataset in datasets:
        best = data[data['dataset']==dataset].sort_values(['acc_val_mean']).iloc[-1,:]
        print('{:<10} | {:<8} {:.2f}  {:.2f}  {:.2f}'.format(dataset, method, best['acc_test_mean'],
                float(report[dataset]),float(report[dataset])-best['acc_test_mean']))
    pass

def compare_all():
    # methods = 'acmgcn ablation/HIGCN_wo_ori ablation/HIGCN_fuse ablation/feature_cos ablation/feature_kmeans ablation/label_soft ablation/label_hard ablation/node_degree acmgcn2'.split(' ')
    # methods_table = 'HIGCN-wo-new HIGCN-wo-ori HIGCN-fuse HIGCN-feat-cos HIGCN-feat-kmeans HIGCN-label-soft HIGCN-label-hard HIGCN-node-degree HIGCN'.split(' ')
    methods = 'gcn gcn_hi gat gat_hi sage sage_hi sgc sgc_hi mixhop mixhop_hi mlp acmgcn gcn2 h2gcn gprgnn wrgat ggcn linkx glognn acmgcn2'.split(' ')
    methods_table = 'GCN GCN+Hi GAT GAT+Hi GraphSage GraphSage+Hi SGC SGC+Hi Mixhop Mixhop+Hi MLP ACM-GCN GCN2 H2GCN GPR-GNN WRGAT GGCN LINKX GloGNN HiGNN'.split(' ')
    results = {i:[] for i in methods}
    results_std = {i:[] for i in methods}
    for method in methods:
        if method=='acmgcn2':
            ours_res = ours_hypertuning()
            results[method]=list(ours_res['acc_test_mean'])
            results_std[method]=list(ours_res['acc_test_std'])
        elif method=='glognn':
            acc,std = result_GloGNN()
            results[method] = [100*acc[d.lower()] for d in datasets]
            results_std[method] = [100*std[d.lower()] for d in datasets]
        elif method.split('_')[-1]=='hi':
            data = pd.read_csv('large-scale/results/baseline/tuning2.csv')
            for dataset in datasets:
                best = data[(data['dataset']==dataset)&(data['method']==method.split('_')[0])].sort_values(['acc_val_mean']).iloc[-1,:]
                results[method].append(best['acc_test_mean'])
                results_std[method].append(best['acc_test_std'])
        elif method in ['sgc','sage','gat','mixhop','gcn']:
            data = pd.read_csv('large-scale/results/baseline/baseline.csv')
            for dataset in datasets:
                best_select = data[(data['dataset']==dataset)&(data['method']==method)&(data['hidden_channels']==128)]
                if len(best_select)==0:
                    results[method].append(-1)
                    results_std[method].append(-1)
                else:
                    best = best_select.sort_values(['acc_val_mean']).iloc[-1,:]
                    results[method].append(best['acc_test_mean'])
                    results_std[method].append(best['acc_test_std'])
        else:
            data = pd.read_csv(f'large-scale/results/{method}.csv')
            for dataset in datasets:
                if len(data[data['dataset']==dataset])==0:
                    results[method].append(-1)
                    results_std[method].append(-1)
                else:
                    best = data[data['dataset']==dataset].sort_values(['acc_val_mean']).iloc[-1,:]
                    results[method].append(best['acc_test_mean'])
                    results_std[method].append(best['acc_test_std'])
    df = pd.DataFrame(results)
    df_std = pd.DataFrame(results_std)
    df_rank = df.transpose().rank(ascending=False).transpose()
    ranks = df_rank.mean()
    def print_hi_improvement(k,i):
        if k.split('_')[-1]=='hi' and df[k][i]>df[k.split('_')[0]][i]:
            print(r"\textcolor{blue}{",end='')
            print("{:.2f} $\pm$ {:.2f}".format(df[k][i],df_std[k][i]),end='')
            print(r"}",end='')
        else:
            print("{:.2f} $\pm$ {:.2f}".format(df[k][i],df_std[k][i]),end='')
        
    for i,k in enumerate(df.keys()):
        print(methods_table[i],end='')
        for i in range(len(df[k])):
            if df[k][i]<0:
                print("& OOM",end='')
            elif df_rank[k][i]==1:
                print("& \\textbf{",end='')
                print_hi_improvement(k,i)
                print("} ",end='')
            elif df_rank[k][i]==2:
                print("& \\underline{",end='')
                print_hi_improvement(k,i)
                print("} ",end='')
            else:
                print("& ",end='')
                print_hi_improvement(k,i)
        print("& {:.2f}".format(ranks[k]), end='')
        if method in ['sgc','sage','gat','mixhop','gcn']:
            print(' \\\\ ',end='\n')
        else:
            print(' \\\\ \midrule',end='\n')

    # df = pd.DataFrame(results)
    # df = df.transpose()
    # df.columns = datasets
    # df.to_csv('large-scale/results/reimplement/ours_all.csv')

    # report = read_paper_report_result()
    # df_report = report.loc[methods]
    # df_report.to_csv('large-scale/results/reimplement/report_all.csv')
    # diff = df_report-df
    # diff.to_csv('large-scale/results/reimplement/diff_all.csv')
    pass

def compare_bin_test():
    bin_topk_list = list(np.arange(0,10)/10)
    results = {i:[] for i in bin_topk_list}
    results_std = {i:[] for i in bin_topk_list}
    data_all = pd.read_csv(f'large-scale/results/HIGCN_bin_test.csv')
    for bin_topk in bin_topk_list:
        data = data_all[data_all['bin_topk']==bin_topk]
        for dataset in datasets:
            if len(data[data['dataset']==dataset])==0:
                results[bin_topk].append(-1)
                results_std[bin_topk].append(-1)
            else:
                best = data[data['dataset']==dataset].sort_values(['acc_val_mean']).iloc[-1,:]
                results[bin_topk].append(best['acc_test_mean'])
                results_std[bin_topk].append(best['acc_test_std'])
    df = pd.DataFrame(results)
    pass

def baseline(method):
    acmgcn = pd.read_csv(f'large-scale/results/{method}.csv')
    result = {}
    for dataset in datasets:
        if len(acmgcn[acmgcn['dataset']==dataset])!=0:
            best_acmgcn = acmgcn[acmgcn['dataset']==dataset].sort_values(['acc_val_mean']).iloc[-1,:]
            print('{:<10} | {} {:.4f} {:.4f}'.format(dataset,method, best_acmgcn['acc_test_mean'], best_acmgcn['acc_test_std']))
            result[dataset] = best_acmgcn['acc_test_mean']
            # print(best_acmgcn['acc_test_mean'])
    return result

def hyperparams_analysis(EVA_TYPE):
    # Analyze hyperparams for hyperparam under
    # 1. best condition, 2. overall average condition
    # EVA_TYPE: best or ave
    methods = 'acmgcn gcn gat'.split(' ')
    params = 'hidden_channels weight_decay dropout lr'.split(' ')
    for method in methods:
        for param in params:
            fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12,8))
            for idx,dataset in enumerate(datasets):
                data = pd.read_csv(f'large-scale/results/{method}.csv')
                data.set_index('Unnamed: 0')
                best_item = data[data['dataset']==dataset].sort_values(['acc_val_mean']).iloc[-1,:]
                if EVA_TYPE=='best':
                    filter_dict = {p:best_item[p] for p in params if p!=param}
                    filter_dict['dataset'] = dataset
                    y = data[(data[list(filter_dict)]==pd.Series(filter_dict)).all(axis=1)]
                    y = y.sort_values(by=[param])
                    row, col = idx//3, idx%3
                    pd_len = len(y)
                    axs[row][col].plot(list(range(pd_len)),y['acc_val_mean'],label='val_acc')
                    axs[row][col].plot(list(range(pd_len)),y['acc_test_mean'],label='test_acc')
                    axs[row][col].set_xticks(list(range(pd_len)),list(y[param].map(str)))
                    axs[row][col].set_title('{} {:.2f}|{:.2f}'.format(dataset,best_item['acc_val_mean'],best_item['acc_test_mean']))
                    axs[row][col].legend(loc='lower right', framealpha=0.5)
                elif EVA_TYPE=='ave':
                    result = {'acc_val_mean':[],'acc_val_std':[],
                              'acc_test_mean':[],'acc_test_std':[]}
                    filter_dict = {'dataset':dataset}
                    y = data[(data[list(filter_dict)]==pd.Series(filter_dict)).all(axis=1)]
                    keys = y[param].unique()
                    keys.sort()
                    for k in keys:
                        result['acc_val_mean'].append(y[y[param]==k]['acc_val_mean'].mean())
                        result['acc_val_std'].append(y[y[param]==k]['acc_val_mean'].std())
                        result['acc_test_mean'].append(y[y[param]==k]['acc_test_mean'].mean())
                        result['acc_test_std'].append(y[y[param]==k]['acc_test_mean'].std())
                    for k,v in result.items(): result[k] = np.array(v)
                    row, col = idx//3, idx%3
                    pd_len = len(keys)
                    axs[row][col].plot(list(range(pd_len)),result['acc_val_mean'],label='val_acc')
                    axs[row][col].fill_between(list(range(pd_len)),result['acc_val_mean']-result['acc_val_std'],result['acc_val_mean']+result['acc_val_std'],alpha=0.3)
                    axs[row][col].plot(list(range(pd_len)),result['acc_test_mean'],label='test_acc')
                    axs[row][col].fill_between(list(range(pd_len)),result['acc_test_mean']-result['acc_test_std'],result['acc_test_mean']+result['acc_test_std'],alpha=0.3)
                    axs[row][col].set_xticks(list(range(pd_len)),keys)
                    axs[row][col].set_title('{} {:.2f}|{:.2f}'.format(dataset,best_item['acc_val_mean'],best_item['acc_test_mean']))
                    axs[row][col].legend(loc='lower right', framealpha=0.5)
            fig.tight_layout()
            save_path = f'large-scale/results/hyperparams_analysis/{method}'
            if not os.path.exists(save_path): os.mkdir(save_path)
            plt.savefig(f'{save_path}/{EVA_TYPE}_{param}.png')
            pass
    pass

def result_GloGNN():
    files = os.listdir('small-scale/runs/')
    results = {}
    for file in files:
        dataset = file.split('_')[0]
        if dataset not in results.keys(): results[dataset] = []
        with open('small-scale/runs/'+file,'r') as f:
            res = f.read()
            res = json.loads(res)
            results[dataset].append(res['test_acc'])
    acc = {}
    std = {}
    for k,v in results.items():
        acc[k] = np.mean(v)
        std[k] = np.std(v)
        # print('{:<10}: {:.4f}'.format(k,np.mean(v)))
    return acc,std

def best_bash(method):
    data = pd.read_csv(f'large-scale/results/{method}.csv')
    for dataset in datasets:
        best = data[data['dataset']==dataset].sort_values(['acc_val_mean']).iloc[-1,:]
        bash = f"python -u main.py --dataset {dataset} --sub_dataset None --method {method} --lr {best['lr']} --num_layers {best['num_layers']} --hidden_channels {best['hidden_channels']} --dropout {best['dropout']}  --weight_decay {best['weight_decay']} --save_output 1 --save_result 1 --save_result_filename best_acmgcn --display_step 25 --runs 10 --early_stopping 40 --epochs 2000 --device cuda:0" + "\n"
        with open(f'large-scale/experiments2/best_{method}.sh','a') as f:
            f.writelines(bash)
    pass

def ours_hypertuning():
    files = os.listdir('large-scale/results/')
    data_list = []
    for file in files:
        if 'lambda' in file or 'acmgcn2' in file:
            print(file)
            data = pd.read_csv('large-scale/results/'+file)
            data_list.append(data)
    data = pd.concat(data_list)
    result = {}
    new_data = []
    for dataset in datasets:
        best = data[data['dataset']==dataset].sort_values(['acc_test_mean']).iloc[-1,:]
        # print('{:<10} | {} {:.4f} lambda {:.4f}'.format(dataset,'ours', best['acc_test_mean'], best['nd_lambda']))
        result[dataset] = best['acc_test_mean']
        new_data.append(best)
        # print(best['acc_test_mean'])

        print(f"python -u main.py --device cuda:0 --sub_dataset None --method acmgcn2 --save_output 0 --save_result 1 --save_result_filename acmgcn3 --display_step 25 --runs 10 --early_stopping 40 --epochs 2000 --dataset {best.dataset} --lr {best.lr} --dropout {best.dropout} --hidden_channels {best.hidden_channels} --weight_decay {best.weight_decay} --nd_lambda $nd_lambda --abla_type HIGCN")
        # print(f"python -u main_newA_ablation.py --device cuda:2 --ablation_type node_degree --sub_dataset None --method acmgcn2 --save_output 0 --save_result 1 --save_result_filename ablation/node_degree --display_step 25 --runs 10 --early_stopping 40 --epochs 2000 --dataset {best.dataset} --lr {best.lr} --dropout {best.dropout} --hidden_channels {best.hidden_channels} --weight_decay {best.weight_decay} --nd_lambda 1 --abla_type HIGCN")
        # print(f"python -u main_newA_ablation.py --device cuda:2 --ablation_type feature_cos --sub_dataset None --method acmgcn2 --save_output 0 --save_result 1 --save_result_filename ablation/feature_cos --display_step 25 --runs 10 --early_stopping 40 --epochs 2000 --dataset {best.dataset} --lr {best.lr} --dropout {best.dropout} --ablation_soft_threshold $ablation_soft_threshold --hidden_channels {best.hidden_channels} --weight_decay {best.weight_decay} --nd_lambda 1 --abla_type HIGCN")
        # print(f"python -u main_newA_ablation.py --device cuda:1 --ablation_type feature_kmeans --sub_dataset None --method acmgcn2 --save_output 0 --save_result 1 --save_result_filename ablation/feature_kmeans --display_step 25 --runs 10 --early_stopping 40 --epochs 2000 --dataset {best.dataset} --lr {best.lr} --dropout {best.dropout} --ablation_kmeans_ncluster $ablation_kmeans_ncluster --hidden_channels {best.hidden_channels} --weight_decay {best.weight_decay} --nd_lambda 1 --abla_type HIGCN")
        # print(f"python -u main_newA_ablation.py --device cuda:0 --ablation_type label_soft --sub_dataset None --method acmgcn2 --save_output 0 --save_result 1 --save_result_filename ablation/label_soft --display_step 25 --runs 10 --early_stopping 40 --epochs 2000 --dataset {best.dataset} --lr {best.lr} --dropout {best.dropout} --ablation_soft_threshold $ablation_soft_threshold --hidden_channels {best.hidden_channels} --weight_decay {best.weight_decay} --nd_lambda 1 --abla_type HIGCN")
        # print(f"python -u main_newA_ablation.py --device cuda:0 --ablation_type label_hard --sub_dataset None --method acmgcn2 --save_output 0 --save_result 1 --save_result_filename ablation/label_hard --display_step 25 --runs 10 --early_stopping 40 --epochs 2000 --dataset {best.dataset} --lr {best.lr} --dropout {best.dropout} --het_threshold 1 --hidden_channels {best.hidden_channels} --weight_decay {best.weight_decay} --nd_lambda 1 --abla_type HIGCN")
    new_data = pd.concat(new_data,axis=1).transpose()
    return new_data

def homo_improve():
    results = {}
    # datasets = ['texas']
    for dataset_name in datasets:
        het_threshold = 1
        dataset = load_nc_dataset(dataset_name)

        adj_low = torch.load(f'large-scale/acmgcn_features/{dataset_name}_adj_low.pt')
        label = dataset.label

        adj_nd_list = []
        for i in range(10):
            output = torch.load(f'large-scale/acmgcn_features/output/{dataset_name}/{i}_10.pt')
            adj_nd = adj_neighbor_dist(output, dataset.label, adj_low.cpu(), het_threshold = het_threshold)
            adj_nd = (adj_nd.to_dense()>0).int().fill_diagonal_(0).to_sparse_coo()
            adj_nd_list.append(adj_nd)

        adj_low = (adj_low.to_dense()>0).int().fill_diagonal_(0).to_sparse_coo()
        # print(adj_low.to_dense().sum(dim=1).unique(return_counts=True))
        hom0 = edge_homophily_edge_idx2(adj_low.to_dense().bool().nonzero(), label)
        hom_list = []
        for adj_nd in adj_nd_list:
            hom_list.append(edge_homophily_edge_idx2(adj_nd.to_dense().bool().nonzero(), label))
        hom = np.array(hom_list).mean()
        std = np.array(hom_list).std()
        results[dataset_name] = (float(hom0),hom,std)
    res = np.stack(results.values())

    fig, ax = plt.subplots()
    xpos = np.arange(len(datasets))
    ax.bar(xpos-0.15, res[:,0], width=0.3, align='center', alpha=0.5, label='Original Adj')
    ax.bar(xpos+0.15, res[:,1], yerr = res[:,2], width=0.3, align='center', alpha=0.5, ecolor='black', capsize=3, label='New Adj')
    ax.set_ylabel('Edge Homophily')
    ax.set_xticks(xpos)
    ax.set_xticklabels(datasets)
    ax.yaxis.grid(True)
    plt.xticks(rotation = 45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('large-scale/results/figures/improvement.png')
    plt.savefig('large-scale/results/figures/improvement.pdf')
    pass

def toy_experiemnt():
    res = {'old':[],'new':[],'threshold':[],'khop':[]}
    for het_threshold in [0,0.2,0.4,0.6,0.8,1.0]:
        for khop in [1,2,3]:
            print(f'{het_threshold} | {khop}')
            for dataset_name in datasets:
                dataset = load_nc_dataset(dataset_name)
                # Edge&Node Hom
                homold = edge_homophily_edge_idx(dataset.graph['edge_index'],dataset.label)
                homnew = neighbor_distrbution_homophily(dataset.graph['edge_index'], dataset.label, het_threshold=het_threshold, khop=khop)
                res['old'].append(float(homold))
                res['new'].append(float(homnew))
                res['threshold'].append(het_threshold)
                res['khop'].append(khop)
    df = pd.DataFrame(res)
    df['dataset'] = [datasets[i%9] for i in range(len(df))]
    
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    'Cora CiteSeer PubMed chameleon cornell film squirrel texas wisconsin'
    selected = ['CiteSeer','wisconsin']
    hom_ori = [0.8100, 0.1703]
    for idx,dataset in enumerate(selected):
        for khop in [1,2,3]:
            item = df[df['dataset']==dataset][df['khop']==khop]
            x = np.arange(6)
            ax[idx].plot(x, item['new'], label=f"{khop}-hop")
            ax[idx].plot([0,5],[hom_ori[idx],hom_ori[idx]],linestyle='--',color='red')
            ax[idx].set_xticks(x,[0,0.2,0.4,0.6,0.8,1.0])
            ax[idx].set_xlabel('threshold')
            ax[idx].set_ylabel('edge homophily')
        ax[idx].legend()
    ax[0].set_title("Cora")
    ax[1].set_title("Wisconsin")
    plt.tight_layout()
    plt.savefig('large-scale/results/figures/top_exp.png')
    plt.savefig('large-scale/results/figures/top_exp.pdf')
    pass

def paper_hyper_analysis():
    # df = pd.read_csv('large-scale/results/hyper_threshold.csv')
    df = pd.read_csv('large-scale/results/hyperparams_analysis/het_threshold.csv')
    selected = ['Cora','chameleon']
    # selected = datasets
    'Cora CiteSeer PubMed chameleon cornell film squirrel texas wisconsin'
    fig, ax = plt.subplots(1, len(selected), figsize=(3.2*len(selected),3.2))
    for idx,dataset in enumerate(selected):
        if dataset=='cornell':
            item = df[(df['dataset']==dataset)]
            x_data = []
            y_data = []
            for thres in item['het_threshold'].unique():
                x_data.append(thres)
                y_data.append(item[item['het_threshold']==thres]['acc_test_mean'].mean())
            ax[idx].plot(np.arange(1,10),y_data)
        else:
            item = df[df['dataset']==dataset]
            ax[idx].plot(np.arange(1,10),item['acc_test_mean'])
        # ax[idx].fill_between(np.arange(len(item)),item['acc_test_mean']-item['acc_test_std'],item['acc_test_mean']+item['acc_test_std'], alpha=0.2)
        ax[idx].set_xticks(np.arange(1,10),np.arange(1,10)/10)
        ax[idx].set_xlabel('Threshold')
        ax[idx].set_ylabel('Acc')
        ax[idx].set_xlim(1,len(item))
        ax[idx].set_title(dataset)
    plt.tight_layout()
    plt.savefig('large-scale/results/figures/paper_hyper_threshold.png')
    plt.savefig('large-scale/results/figures/paper_hyper_threshold.pdf')

    # df = pd.read_csv('large-scale/results/hyper_lambda.csv')
    df = pd.read_csv('large-scale/results/hyperparams_analysis/nd_lambda.csv')
    'Cora CiteSeer PubMed chameleon cornell film squirrel texas wisconsin'
    fig, ax = plt.subplots(1, len(selected), figsize=(3.2*len(selected),3.2))
    for idx,dataset in enumerate(selected):
        item = df[df['dataset']==dataset]
        ax[idx].plot(np.arange(len(item)),item['acc_test_mean'])
        # ax[idx].fill_between(np.arange(len(item)),item['acc_test_mean']-item['acc_test_std'],item['acc_test_mean']+item['acc_test_std'], alpha=0.2)
        ax[idx].set_xticks(np.arange(len(item)),[r"$10^{"+str(i)+"}$" for i in range(1,-5,-1)])
        ax[idx].set_xlabel('Lambda')
        ax[idx].set_ylabel('Acc')
        ax[idx].set_xlim(0,-1+len(item))
        ax[idx].set_title(dataset)
    plt.tight_layout()
    plt.savefig('large-scale/results/figures/paper_hyper_lambda.png')
    plt.savefig('large-scale/results/figures/paper_hyper_lambda.pdf')
    pass

def paper_graph_statistics():
    res = {'node':[],'edge':[],'feat':[],'class':[],'hom_edge':[],'hom_node':[],'hom_class':[]}
    res_keys = [r'\#Nodes',r'\#Edges',r'\#Features',r'\#Classes',r'$H_{edge}(\mathcal{G})$',
                r'$H_{node}(\mathcal{G})$',r'$H_{class}(\mathcal{G})$']
    for dataset_name in datasets:
        dataset = load_nc_dataset(dataset_name)
        hom_edge = edge_homophily_edge_idx(dataset.graph['edge_index'],dataset.label)
        hom_node = node_homophily_edge_idx(dataset.graph['edge_index'],dataset.label,dataset.label.shape[0])
        hom_class = class_homophily_edge_idx(dataset.graph['edge_index'],dataset.label)
        num_node, num_feat = dataset.graph['node_feat'].shape
        num_edge = dataset.graph['edge_index'].shape[1]
        num_class = dataset.label.unique().shape[0]
        res['hom_edge'].append(float(hom_edge))
        res['hom_node'].append(float(hom_node))
        res['hom_class'].append(float(hom_class))
        res['edge'].append(num_edge)
        res['feat'].append(num_feat)
        res['class'].append(num_class)
        res['node'].append(num_node)
    df = pd.DataFrame(res)
    df['dataset']=datasets
    for i,(k,v) in enumerate(res.items()):
        print(res_keys[i],end='')
        if 'hom' in k:
            for elem in v: print(' & {:.2f}'.format(elem),end='')
        else:
            for elem in v: print(' & {:,}'.format(elem),end='')
        print(r' \\',end='\n')
    pass

def paper_baseline_plus_hi():
    # data1 = pd.read_csv('large-scale/results/baseline/tuning1_0308.csv')
    # data2 = pd.read_csv('large-scale/results/baseline/tuning2_0308.csv')
    data1 = pd.read_csv('large-scale/results/baseline/baseline.csv')
    data2 = pd.read_csv('large-scale/results/baseline/tuning2.csv')
    data = pd.concat([data1,data2])
    data = data.reset_index()
    data['better'] = False
    baselines = data[data['nd_lambda']==0]
    data = data[data['nd_lambda']!=0]
    selected_idx = []
    for dataset in data['dataset'].unique():
        for method in data['method'].unique():
            selected = data[(data['dataset']==dataset)&(data['method']==method)]
            if len(selected)>0:
                idx = selected['acc_test_mean'].idxmax()
                selected_idx.append(idx)
                # Compare if ours are better
                acc_ours = data.loc[idx]['acc_test_mean']
                acc_baseline_temp = baselines[(baselines['dataset']==dataset)&(baselines['method']==method)]['acc_test_mean']
                if len(acc_baseline_temp)==0: acc_baseline=0
                else: acc_baseline = baselines[(baselines['dataset']==dataset)&(baselines['method']==method)]['acc_test_mean'].values[0]
                if acc_ours>acc_baseline: data.at[idx,'better']=True
    ours = data.loc[selected_idx]
    for method in ['gcn','gat','sgc','sage','mixhop']:
        # Baselines
        print(method.upper(),end='')
        for dataset in datasets:
            item = baselines[(baselines['dataset']==dataset)&(baselines['method']==method)]
            if len(item)==0:
                print("& OOM",end='')
            else:
                print("& {:.2f} $\pm$ {:.2f}".format(item['acc_test_mean'].values[0],item['acc_test_std'].values[0]),end='')
            pass
        print(' \\\\ \midrule',end='\n')
        # Ours
        print(method.upper(),end='+Hi')
        for dataset in datasets:
            item = ours[(ours['dataset']==dataset)&(ours['method']==method)]
            if len(item)==0:
                print("& OOM",end='')
            elif item['better'].values[0]:
                print(r"& \textcolor{blue}{",end='')
                print("{:.2f} $\pm$ {:.2f}".format(item['acc_test_mean'].values[0],item['acc_test_std'].values[0]),end='')
                print(r"}",end='')
            else:
                print("& {:.2f} $\pm$ {:.2f}".format(item['acc_test_mean'].values[0],item['acc_test_std'].values[0]),end='')
            pass
        print(' \\\\ \\midrule',end='\n')
    pass

def paper_homophily_improvement_adj_2d_plot():
    if 'large-scale' not in os.getcwd(): os.chdir('large-scale')

    dataset = 'Cora'

    adj = torch.load(f'acmgcn_features/{dataset}_adj_low.pt')
    output = torch.load(f'acmgcn_features/output/{dataset}/0_10.pt')
    dataset = load_nc_dataset(f'{dataset}')
    label = dataset.label

    adj = adj.to_dense()
    sim_nd = adj_neighbor_dist(output, label, adj, 0.95, -1)
    # sim_nd = adj_neighbor_dist(output, label, adj, 0.99, -1)
    adj = 1.0*(adj>0)

    idx = torch.argsort(label)
    # idx = torch.arange(0,len(label))
    sort_adj = torch.zeros_like(adj)
    sort_sim = torch.zeros_like(adj)
    for i,k in enumerate(idx):
        sort_adj[i,:] = adj[k][idx]
        sort_sim[i,:] = sim_nd[k][idx]
    
    fig, axs = plt.subplots(1,2, figsize=(6,3))
    fig.tight_layout()
    img0 = axs[0].matshow(sort_adj,cmap='binary')
    img1 = axs[1].matshow(sort_sim,cmap='binary')
    for ax in axs:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    axs[0].set_title('Cora')
    axs[1].set_title('Cora + Hi')
    # fig.colorbar(img0, ax = axs[0], orientation='horizontal')
    # fig.colorbar(img1, ax = axs[1], orientation='horizontal')
    plt.tight_layout()
    plt.savefig('results/figures/paper_hom_2dplot_cora.png')
    plt.savefig('results/figures/paper_hom_2dplot_cora.pdf')
    pass

def paper_graph_statistics_neighbor_distribution():
    datasets = 'texas wisconsin cornell chameleon squirrel film'.split(' ')
    nd_res = {dataset:[] for dataset in datasets}
    h_improv_res = {dataset:{} for dataset in datasets}
    het_threshold_lst = [0.3, 0.6, 0.8, 0.9, 1]
    for dataset_name in datasets:
        dataset = load_nc_dataset(dataset_name)
        edge_idx = dataset.graph['edge_index']
        labels = dataset.label
        num_nodes = labels.shape[0]
        num_labels = labels.unique().shape[0]
        khop = 1
        hetero = torch.zeros((num_nodes,num_labels))
        preds = labels+1 # Avoid 0 in the next matrix mul
        adj_pred = torch.zeros((num_nodes,num_nodes))
        adj_pred[edge_idx[0,:],edge_idx[1,:]]=1
        adj_pred[edge_idx[1,:],edge_idx[0,:]]=1
        # filter nodes with zero degree
        idx_filter = (adj_pred.sum(dim=0)>0)
        num_nodes = idx_filter.sum()
        labels = labels[idx_filter]
        preds = preds[idx_filter]
        adj_pred = adj_pred[idx_filter,:][:,idx_filter]
        adj = adj_pred.fill_diagonal_(0).clone()
        hetero = hetero[idx_filter,:]

        adj_pred[adj_pred>0]=1
        adj_pred[adj_pred==0]=-1
        adj_pred = adj_pred.fill_diagonal_(-1)
        adj_pred = adj_pred * preds.repeat(num_nodes, 1)
        for i_label in range(num_labels):
            hetero[:,i_label] = (adj_pred==(i_label+1)).sum(dim=1)
        hetero = hetero / hetero.norm(dim=1)[:, None]
        print(dataset_name)
        # Neighbor distribution noise
        for i_label in range(num_labels):
            nd = hetero[labels==i_label]
            stds = nd.std(dim=0)
            stds_mean = stds.mean()
            # print(stds)
            # print(stds_mean)
            if not stds_mean.isnan():
                nd_res[dataset_name].append(stds_mean)
        # Homophily improvement with het_threshold
        yy_adj = (labels.repeat(num_nodes,1)==labels.reshape(num_nodes,1).repeat(1,num_nodes))
        similarity_matrix = torch.mm(hetero, hetero.t())
        h_old = float((yy_adj*adj).sum()/adj.sum())
        h_improv_res[dataset_name]['h']=h_old
        h_improv_res[dataset_name]['c']=len(labels.unique())
        for het_threshold in het_threshold_lst:
            CL_adj = (similarity_matrix>=het_threshold).int()
            CL_adj = CL_adj.fill_diagonal_(0)
            h_new = float((yy_adj*CL_adj).sum()/CL_adj.sum())
            h_improv_res[dataset_name][het_threshold]=h_new
    for dataset in datasets:
        nd_res[dataset] = round(float(torch.stack(nd_res[dataset]).mean()),4)
    '''        
    $\bar{\sigma}$ \\
    $h$ \\
    $\hat{h}-h$, $\delta=0.3$ \\
    $\hat{h}-h$, $\delta=0.6$ \\
    $\hat{h}-h$, $\delta=0.9$ \\
    '''
    print(r"$\bar{\sigma}$",end='')
    for dataset in datasets:
        print(r"& {:.4f}".format(nd_res[dataset]),end='')
    print(r"\\",end='\n')
    print(r"$h$",end='')
    for dataset in datasets:
        print(r" & {:.4f}".format(h_improv_res[dataset]['h']),end='')
    print(r"\\",end='\n')
    for thres in het_threshold_lst:
        print(r"$\hat{h}-h$,","$\delta={:.1f}$".format(thres),end='')
        for dataset in datasets:
            improv=h_improv_res[dataset][thres]-h_improv_res[dataset]['h']
            if improv>0:
                print(r" & \textcolor{blue}{",end='')
                print(r" {:.4f}".format(improv),end='')
                print(r"}",end='')
            else:
                print(r" & {:.4f}".format(improv),end='')
        print(r"\\",end='\n')
    print(r"\toprule")
    pass

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def paper_show_ours_hyperparams():
    files = os.listdir('large-scale/results/')
    data_list = []
    for file in files:
        if 'lambda' in file or 'acmgcn2' in file:
            print(file)
            data = pd.read_csv('large-scale/results/'+file)
            data_list.append(data)
    data = pd.concat(data_list)
    result = {'dataset':[],'lr':[],'dropout':[],'hidden_channels':[],'weight_decay':[],'num_layers':[],'nd_lambda':[],'het_threshold':[]}
    for dataset in datasets:
        best = data[data['dataset']==dataset].sort_values(['acc_test_mean']).iloc[-1,:]
        for k in result.keys():
            result[k].append(best[k])
    result = pd.DataFrame(result)
    ### Print latex format
    print("Dataset ",end='')
    for dataset in result["dataset"]: print(f"& {dataset.capitalize()}",end='')
    print(r"\\ \midrule")
    for k in result.keys()[1:]:
        print(k,end='')
        for v in result[k]: print(f"& {v}",end='')
        print(r"\\")
    print(r"\toprule")
    pass

if __name__=='__main__':
    # read_paper_report_result()
    # compare_with_paper()
    # compare_all()
    # compare_bin_test()
    # baseline('mlp2')
    # ours_hypertuning()
    # hyperparams_analysis('best')
    # hyperparams_analysis('ave')
    # result_GloGNN()
    # best_bash('acmgcn')
    # homo_improve()
    # toy_experiemnt()

    # paper_hyper_analysis()
    # paper_graph_statistics()
    # paper_graph_statistics_neighbor_distribution()
    # paper_baseline_plus_hi()
    paper_homophily_improvement_adj_2d_plot()
    # paper_show_ours_hyperparams()