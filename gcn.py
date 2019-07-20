
import os
import sys
import glob 
import math

import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.metrics import roc_curve,auc,\
#             average_precision_score,\
#             roc_auc_score,accuracy_score
from tqdm import tqdm , trange
import scipy.sparse as sp
import scipy.linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torchviz

np.set_printoptions(threshold=np.inf)

NNODES = 200
NFEATS = 5
DEVICE = 'cpu'
NEPOCH = 5
NBATCH = 1000
HIDW = NFEATS 
HIDD = 2
#PATH = '/local/snarayan/noneut_0/*npz'
PATH = '/local/snarayan/grapple_1/*npz'


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

class FakeDataset(object):
    def __init__(self, st=None):
        self.st = st
    def gen(self, refresh=False):
        while True:
            xs, ys, As = [], [], []
            N = 0
            partition = np.random.randint(1, NNODES-1)
            for n in (partition, NNODES-partition):
                x = np.ones((n,1)) * len(xs)
                y = np.ones(n) * len(ys)
                A = np.random.binomial(n=1, p=.3, size=(n*n)).reshape(n,n).astype(bool)
                A += A.T 
                A += np.eye(n).astype(bool)
                A = A.astype(float)
                As.append(A); xs.append(x); ys.append(y)
            x = np.concatenate(xs).astype(np.float32)
            y = np.concatenate(ys).astype(np.int64)
            A = scipy.linalg.block_diag(*As)
            # A = sp.coo_matrix(A).tocsr()

            D = np.array(A.sum(1))
            D = np.power(D,-1).flatten()
            D[np.isinf(D)] = 0
            D = np.sqrt(D)
            #D = sp.diags(D)
            D = np.diag(D)
            Atilde = D.dot(A).dot(D)
            Atilde = Atilde.astype(np.float32)

            '''
            Atilde = Atilde.tocoo().astype(np.float32)
            idx = n2t(np.vstack([Atilde.row, Atilde.col]).astype(np.int64))
            data = n2t(Atilde.data)
            Atilde = torch.sparse.FloatTensor(
                    idx, data, torch.Size(Atilde.shape)
                )
            '''
            Atilde = n2t(Atilde)

            yield n2t(x.astype(np.float32)), n2t(y.astype(np.int64)), Atilde

class Dataset(object):
    def __init__(self, path, st=None):
        self._path = path 
        self._files = glob.glob(path)
        self.st = st
    def gen(self, refresh=False):
        while True:
            np.random.shuffle(self._files)
            for f_ in self._files:
                d = np.load(f_, allow_pickle=True)
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

                    y = (y > 0).astype(np.int64)

                    D = np.array(A.sum(1))
                    D = np.power(D,-1).flatten()
                    D[np.isinf(D)] = 0
                    D = np.sqrt(D)
                    D = sp.diags(D)
                    Atilde = D.dot(A).dot(D)

                    Atilde = n2t(Atilde.toarray().astype(np.float32))
                    '''
                    Atilde = Atilde.tocoo().astype(np.float32)
                    idx = n2t(np.vstack([Atilde.row, Atilde.col]).astype(np.int64))
                    data = n2t(Atilde.data)
                    Atilde = torch.sparse.FloatTensor(
                            idx, data, torch.Size(Atilde.shape)
                        )
                    '''

                    yield n2t(x.astype(np.float32)), n2t(y.astype(np.int64)), Atilde

            if not refresh:
                return
            print('Exhausted %s, reloading'%self._path)

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
    def __init__(self, n_in=HIDW, powers=(0,1), latent=[HIDW]*(HIDD)):
        super(GCN, self).__init__()
        
        latent = [n_in] + latent 
        self.convs = nn.ModuleList(
                [GraphConv(i, o, powers, torch.relu) for i,o in zip(latent[:-1], latent[1:])]
            )
        self.convs.append(GraphConv(latent[-1], 2, powers, 
                                    lambda h : torch.log_softmax(h, dim=1)))

    def print_me(self):
        return 
        for i,c in enumerate(self.convs):
            print(i, [t2n(k) for k in c.kernels])

    def forward(self, x, adj):
        h = x
        for c in self.convs:
            h = c(h, adj)
        return h 
        logprobs = []


#trange = lambda x, **kwargs : range(x)

if __name__ == '__main__':

    st = Standardizer()
    #data = Dataset(PATH, st)
    print('data...')
    data = FakeDataset()
    gen = data.gen(refresh=True)

    print('model...')
    model = GCN(n_in=1).to(DEVICE)

    opt = optim.Adam(model.parameters())
    for epoch in trange(NEPOCH):
        model.train()
        count = 0
        train_acc = 0.
        model.print_me()
        for batch in trange(NBATCH, leave=False):
            x, y, Atilde = next(gen)
            count += x.shape[0]
            opt.zero_grad()
            logprobs = model(x, Atilde)
#            for l in logprobs:
#                print t2n(l)
            # print t2n(logprobs)[:10]
            loss = F.nll_loss(logprobs, y)
            #loss = sum([F.nll_loss(logprobs[i], y[i]) for i in xrange(NNODES)])
            loss.backward()
            opt.step()
#            train_acc += (preds == data['labels']).sum().item()
#            print '='*20
        pred = t2n(logprobs).argmax(axis=1)
        # preds = np.stack([t2n(l).argmax(axis=1) for l in logprobs], axis=-1)
        ny = t2n(y)
        nx = t2n(x)
        mask0 = ny == 0
        mask1 = ~mask0
        print('%i/%i charged PV'%(np.sum(pred[mask0]==0), np.sum(mask0)))
        print('%i/%i charged PU'%(np.sum(pred[mask1]==1), np.sum(mask1)))
        print(' -- Epoch %i has loss %.6g'%(epoch, loss.item()))

    for xx,yy,ll in zip(nx, ny, t2n(logprobs)):
        break 
        print(xx,yy,np.exp(ll))

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
