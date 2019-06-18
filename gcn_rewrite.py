#!/usr/bin/env python3

from argparse import ArgumentParser as AP
p = AP()
p.add_argument('--workdir', '-w')
p.add_argument('--npred', '-n', default=1, const=16, type=int, nargs='?')
p.add_argument('--plot', '-p', default=None)
args = p.parse_args() 

import os
import sys
import glob 
import math

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm , trange
import scipy.sparse as sp
import scipy.io as scpio
from scipy.linalg import sqrtm, block_diag  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchviz

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

np.set_printoptions(threshold=np.inf)

DEVICE = 'cpu'
HIDW = 32 
HIDD = 10
HIDB = 1 
NEPOCH = 100 

def t2n(x):
    if isinstance(x, np.ndarray):
        return x
    x = x.data
    if DEVICE == 'cuda':
        x = x.cpu()
    return x.numpy()

def n2t(x, device=DEVICE):
    return torch.from_numpy(x).to(device)

# largely inspired by tkipf's pygcn implementation
# https://github.com/tkipf/pygcn

class GraphConv(nn.Module):
    def __init__(self, n_in, n_out, powers=(1,), activ=torch.relu):
        super(GraphConv, self).__init__()
        self.powers = powers 
        self.kernels = nn.ParameterList([nn.Parameter(torch.Tensor(n_in, n_out)) for _ in powers])
        self.bias = nn.Parameter(torch.Tensor(n_out))
        self.activ = activ
        self.reset_parameters()
    def reset_parameters(self):
        for k in self.kernels:
            stdv = 1. / math.sqrt(k.size(1))
            k.data.uniform_(-stdv, stdv)
            # nn.init.xavier_normal_(k.data)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x, adj):
        support = None 
        for p,k in zip(self.powers, self.kernels):
            h = torch.mm(adj.matrix_power(p), x)
            h = torch.mm(h, k)
            if support is None:
                support = h 
            else:
                support += h 
        '''
        last_adj = None 
        support = None
        for j,k in enumerate(self.kernels):
            if last_adj is not None:
                last_adj = torch.mm(last_adj, adj)
                h = torch.mm(last_adj, x)
            else:
                last_adj = adj 
                h = x 
            h = torch.mm(h, k)
            if support is None:
                support = h 
            else:
                support += h 
        h = torch.mm(adj, x)
        support = torch.mm(h, self.kernels[0])
        '''
        h = self.activ(support)
        return h

class GCN(nn.Module):
    def __init__(self, n_in=HIDW, n_out=1, powers=(0,1), latent=[HIDW]*(HIDD-HIDB), branch_latent=[HIDW]*HIDB):
        super(GCN, self).__init__()
        self._mods = nn.ModuleList()
        
        latent = [n_in] + latent
        self.convs = [GraphConv(i, o, powers, torch.relu) for i,o in zip(latent[:-1], latent[1:])]
        self._mods += self.convs 

        bl = latent[-1:] + branch_latent 
        self.outs = [[GraphConv(i, o, powers, torch.relu) for i,o in zip(bl[:-1], bl[1:])] +
                        [GraphConv(bl[-1], 2, powers, lambda h : torch.log_softmax(h, dim=1))]
                     for _ in range(n_out)]
        for o in self.outs:
            self._mods += o 

    def print_me(self):
        return 
        for i,c in enumerate(self.convs):
            print(i, [t2n(k).flatten() for k in c.kernels])

    def forward(self, x, adj):
        h = x
        for c in self.convs:
            h = c(h, adj)
        logprobs = []
        for o in self.outs:
            hh = h 
            for oo in o:
                hh = oo(hh, adj)
            logprobs.append(hh)
        return logprobs 

class MatDir(object):
    def __init__(self, path):
        self.files = [path+'/'+f for f in os.listdir(path)]
    def __len__(self):
        return len(self.files)*5
    def iter(self, stop=None):
        s = slice(None, stop)
        np.random.shuffle(self.files)
        Xs, Ys, As = [], [], []
        for f in self.files[s]:
            fo = scpio.loadmat(f)
            A = n2t(fo['adj'].toarray().astype(np.float32))
            try:
                YY = fo['indset_label'].astype(np.int)
            except:
                YY = fo['sol'].astype(np.int).T
            X = n2t(np.ones((YY.shape[0], HIDW)).astype(np.float32)) 
            for i in range(YY.shape[1]):
                Xs.append(X)
                As.append(A)
                Ys.append(n2t(YY[:,i]))
        idcs = np.random.permutation(len(Xs))
        for i in idcs:
            yield Xs[i], As[i], Ys[i]
        return 

def hindsight_nll(lpss, Y, loss_only=True):
    losses = [F.nll_loss(lps, Y) for lps in lpss]
    if not loss_only:
        return min(losses), losses 
    else:
        return min(losses)

if __name__ == '__main__':
    model = GCN(n_out=args.npred).to(DEVICE)
    data = MatDir(args.workdir + '/mat/')

    opt = optim.Adam(model.parameters())
    model.print_me()
    for epoch in trange(NEPOCH, desc='outer'):
        model.train()
        avg_acc = 0.
        avg_avgacc = 0.
        avg_loss = 0.
        freq = {i:0 for i in range(args.npred)}
        node_total = 0
        stop = 20
        for X, A, Y in tqdm(data.iter(stop), total=5*stop, desc='inner'):
            opt.zero_grad()
            logprobs = model(X, A)
            if args.plot is not None:
                stacked = torch.stack(logprobs, dim=0)
                dg = torchviz.make_dot(stacked, params=dict(model.named_parameters()))
                dg.format = 'png'; dg.render(f'{args.plot}/model')
                dg.format = 'pdf'; dg.render(f'{args.plot}/model')
                args.plot = None
            if args.npred > 1:
                loss, losses = hindsight_nll(logprobs, Y, loss_only=False)
            else:
                loss = F.nll_loss(logprobs[0], Y)
                losses = [loss]
            loss.backward()
            opt.step()
            i_best = np.argmin(losses)
            freq[i_best] += 1
            probs = [np.exp(t2n(lp)) for lp in logprobs]
            preds = [np.argmax(p, axis=1) for p in probs]
            nY = t2n(Y)
            avg_acc += (preds[i_best] == nY).sum()
            try:
                avg_avgacc += sum([p == nY for p in preds]).sum() / len(preds)  
            except:
                print(preds[0].shape, nY.shape, ) # sum([p == nY for p in preds]))
            avg_loss += loss.item()
            node_total += X.shape[0]
        avg_acc /= node_total; avg_loss /= node_total; avg_avgacc /= node_total
        # print()
        # print('P',probs)
        # print('Y',t2n(Y))
        # print('X', t2n(X).flatten()) 
        # print('A', '\n',t2n(A))
        # model.print_me()
        print()
        print(f'Average training acc={avg_acc:.4f}, average acc={avg_avgacc:.4f}, loss={avg_loss:.3f}, summmary={str(freq)}')
        if avg_acc == 1:
            break
