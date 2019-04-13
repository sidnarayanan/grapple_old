#!/usr/bin/env python

print 'stdlib imports'
import os
import sys
import glob 
import math

print 'numpy etc'
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc,\
            average_precision_score,\
            roc_auc_score,accuracy_score
from tqdm import tqdm , trange
import scipy.sparse as sp

print 'torch...'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print '...done'

np.set_printoptions(threshold=np.inf)

NNODES = 1000
NFEATS = 5
NSCATTER = 1+1
DEVICE = 'cuda'
NEPOCH = 50
NBATCH = 100
PATH = '/local/snarayan/grapple_0/*npz'


def t2n(x):
    if isinstance(x, np.ndarray):
        return x
    x = x.data
    if DEVICE == 'cuda':
        x = x.cpu()
    return x.numpy()

def n2t(x, device=DEVICE):
    return torch.from_numpy(x).to(device)

def rocauc(path,y,y_score):
    fpr, tpr, thresholds = roc_curve(y, y_score)#,pos_label=2)
    roc_auc = auc(fpr, tpr)
    # print("roc_auc_score (%s) : "%path + str(roc_auc_score(y, y_score)))
    # print y[:10]
    # print y_score[:10]
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('/home/snarayan/public_html/figs/dgcnn/pu_1/roc_%s.png'%path)
    plt.savefig('/home/snarayan/public_html/figs/dgcnn/pu_1/roc_%s.pdf'%path)
    
class Standardizer(object):
    def __init__(self):
        self._mu = None
        self._sigmainv = None
    def __call__(self, x):
        if self._mu is None:
            self._mu = x.mean(axis=0)
            self._sigmainv = np.divide(1, x.std(axis=0))
        return  (x - self._mu) * self._sigmainv 

class Dataset(object):
    def __init__(self, path, st=None):
        self._path = path 
        self._files = glob.glob(path)
        self.st = st
    def gen(self, refresh=False):
        while True:
            np.random.shuffle(self._files)
            for f_ in self._files:
                d = np.load(f_)
                As = d['adj']
                xs = d['x']
                ys = d['y']
                
                N = xs.shape[0]

                for i in xrange(N):
                    A = As[i] # NB: differs from pygcn impl in that this already contains self-loops
                    x = xs[i]
                    y = ys[i]

                    if st is not None:
                        x = st(x)

                    D = np.array(A.sum(1))
                    D = np.power(D,-1).flatten()
                    D[np.isinf(D)] = 0
                    D = np.sqrt(D)
                    D = sp.diags(D)
                    Atilde = D.dot(A).dot(D)

                    Atilde = Atilde.tocoo().astype(np.float32)
                    idx = n2t(np.vstack([Atilde.row, Atilde.col]).astype(np.int64))
                    data = n2t(Atilde.data)
                    Atilde = torch.sparse.FloatTensor(
                            idx, data, torch.Size(Atilde.shape)
                        )

                    yield n2t(x), n2t(y), Atilde

            if not refresh:
                return
            print 'Exhausted %s, reloading'%self._pattern

# largely inspired by tkipf's pygcn implementation
# https://github.com/tkipf/pygcn

class GraphConv(nn.Module):
    def __init__(self, n_in, n_out, activ=torch.relu):
        super(GraphConv, self).__init__()
        self.kernel = nn.Parameter(torch.Tensor(n_in, n_out))
        self.bias = nn.Parameter(torch.Tensor(n_out))
        self.activ = activ
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.kernel.size(1))
        self.kernel.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x, Atilde):
        h = torch.mm(x, self.kernel)
        h = torch.mm(Atilde, h)
        h = h + self.bias
        h = self.activ(h)
        return h

class GCN(nn.Module):
    def __init__(self, n_in, n_out, latent=[10, 10]):
        super(GCN, self).__init__()
        
        latent = [n_in] + latent
        self.convs = nn.ModuleList()
        for i,o in zip(latent[:-1], latent[1:]):
            self.convs.append(GraphConv(i, o, torch.relu))
        self.convs.append(GraphConv(latent[-1], n_out, 
                                    lambda h : torch.log_softmax(h, axis=1)))

    def forward(self, x, Atilde):
        h = x
        for c in self.convs:
            h = c(h, Atilde)
        return h


if __name__ == '__main__':

    print 'data'
    st = Standardizer()
    data = Dataset(PATH, st)
    gen = data.gen(refresh=True)

    print 'model'
    model = GCN(NFEATS, NSCATTER).to(DEVICE)

    print 'train'
    opt = optim.Adam(model.parameters())
    for epoch in trange(NEPOCH):
        model.train()
        count = 0
        train_acc = 0.
        for batch in trange(NBATCH):
            x, y, Atilde = next(gen)
            count += x.shape[0]
            opt.zero_grad()
            logprobs = model(x, Atilde)
#            for l in logprobs:
#                print t2n(l)
            # print t2n(logprobs)[:10]
            loss = F.nll_loss(logprobs, y)
            # loss = sum([F.nll_loss(logprobs[i], ys[i]) for i in xrange(NNODES)])
            loss.backward()
            opt.step()
#            train_acc += (preds == data['labels']).sum().item()
#            print '='*20
        preds = np.stack([t2n(l).argmax(axis=1) for l in logprobs], axis=-1)
        pred = preds[0]
        mask0 = y == 0
        mask1 = y !=0
        print '%i/%i PV, %i/%i PU are correct'%(
                    np.sum(pred[mask0] == 0), np.sum(mask0),
                    np.sum(pred[mask1] != 0), np.sum(mask1),
                )

#        ys = []
#        yscores = []
#        tau32s = []
#        for batch in xrange(50):
#            data = shuffle(DEVICE, gtop, gqcd)
#            tau32s.append(1-t2n(data['tau32']))
#            ys.append(t2n(data['labels']))
#            logprobs = model(data['DA'], data['X'])
#            yscores.append(t2n(logprobs)[:,1])
#        ys = np.concatenate(ys)
#        yscores = np.concatenate(yscores)
#        tau32s = np.concatenate(tau32s)
#        print
#        print
#        rocauc('nn',ys,yscores)
#        rocauc('tau32',ys,tau32s)
#        print 'Epoch %i has loss %.6g and acc %.3f'%(epoch, loss.item()/count, train_acc/count)
        print 'Epoch %i has loss %.6g'%(epoch, loss.item()/BATCH)
