#!/usr/bin/env python

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--minbias')
parser.add_argument('--hard')
parser.add_argument('--output')
parser.add_argument('--npu', type=int)
parser.add_argument('--nmax', type=int, default=-1)
args = parser.parse_args()

import ROOT as root
root.PyConfig.IgnoreCommandLineOptions = True 

from tqdm import tqdm, trange
import numpy as np
import scipy.linalg
import scipy.sparse
import sys 


MAXPARTICLES = 1000
NEUTRALS = set([22, 2112])
FILESIZE = 5e3
IFILE = 0
NGRID = 500
MINDR = 0.025

def feta(pt, pz):
    return -np.log(np.tan(0.5 * np.arctan(np.abs(pt / pz)))) * np.sign(pz)

def ftheta(eta):
    return 2 * np.exp(np.exp(-eta))

def fphi(px, py):
    return np.arctan(py / px)

def fpxyz(pt, eta, phi):
    px = np.cos(phi) * pt
    py = np.sin(phi) * pt
    pz = np.tan(ftheta(eta)) * pt
    return px, py, pz


class Record(object):
    def __init__(self, fpath):
        self.fpath = fpath
        self.f = open(fpath)
    def get_event(self):
        lines = []
        for l_ in self.f:
            l = l_.strip()
            if l.startswith('#'):
                continue
            if 'end' in l:
                break
            lines.append(l)
        if not lines:
            raise Exception('Input file (%s) is exhausted!'%self.fpath)
        return lines

class Particle(object):
    __slots__ = ['y', 'eta', 'phi', 'pt', 'm', 'pdgid', 'q']
    def __init__(self, line=None, vidx=None, eta=None, phi=None, pt=None, pdgid=None, q=None):
        self.y = vidx
        if line is not None:
            px, py, pz, self.m, self.pdgid = [float(l) for l in line.split()]
            self.pt = np.sqrt(px*px + py*py)
            self.eta = feta(self.pt, pz)
            self.phi = fphi(px, py)
        else:
            self.pt = pt
            self.eta = eta
            self.phi = phi
            self.pdgid = pdgid
            self.q = q
    @property
    def x(self):
        vlabel = self.y if self.q else -1 
        return np.array([self.pt, self.eta, self.phi, self.pdgid, vlabel]) 

class Grid(object):
    def __init__(self, nidx):
        self.idxs = range(nidx)
        self._h = [root.TH2F('','',NGRID,-5,5,NGRID,-np.pi,np.pi) for _ in self.idxs]
        self._x = self._h[0].GetXaxis() 
        self._y = self._h[0].GetYaxis()
        self._p = set([])
    def clear(self):
        for h in self._h:
            h.Reset()
        self._p.clear()
    def add(self, p):
        ieta, iphi, idx = self._x.FindBin(p.eta), self._y.FindBin(p.phi), self._h[0].FindBin(p.eta,p.phi)
        self._p.add((ieta,iphi,idx))
        self._h[p.y].SetBinContent(idx, self._h[p.y].GetBinContent(idx) + p.pt)
    def get_particles(self):
        pss = [[] for _ in self.idxs]
        for ieta,iphi,idx in self._p:
            eta = self._x.GetBinCenter(ieta)
            phi = self._y.GetBinCenter(iphi)
            pts = [h.GetBinContent(idx) for h in self._h]
            pt = sum(pts)
            px,py,pz = fpxyz(sum(pts), eta, phi)
            vidx = np.argmax(pts)
            pss[vidx].append(Particle(vidx=vidx, eta=eta, phi=phi, pt=pt, pdgid=0, q=0))
        return pss 
    def run(self, ps):
        for p in ps:
            self.add(p)
        return self.get_particles()

class Interaction(object):
    __slots__ = ['vidx', 'charged','neutral']
    def __init__(self, rec=None, vidx=-1, npu=1):
        self.charged = []
        self.neutral = []
        self.vidx = vidx
        if rec is None:
            return 
        for _ in xrange(npu):
            for l in rec.get_event():
                p = Particle(line=l, vidx=vidx)
                if p.pdgid in NEUTRALS:
                    p.q = 0
                    self.neutral.append(p)
                else:
                    p.q = 1
                    self.charged.append(p)
    @property
    def particles(self):
        return self.charged + self.neutral
    @property
    def x(self):
        return np.array([p.x for p in self.particles])
    @property
    def y(self):
        return np.array([p.y for p in self.particles])
    def get_neutrals(self):
        n = self.neutral
        self.neutral = []
        return n

def get_adj(ep, k):
    e,p = ep[:,0].reshape(-1,1), ep[:,1].reshape(-1,1)
    N = ep.shape[0]
    de = np.square((e - e.T))
    dp = np.square(np.arcsin(np.sin((p - p.T))))
    dr = de + dp 
    nbors = np.argpartition(dr, np.arange(k), axis=0)[:k].reshape(-1)
    dist = np.sqrt(np.partition(dr, np.arange(k), axis=0)[:k]).reshape(-1)
    invdist = np.minimum(1./MINDR,
                         np.divide(1, dist+MINDR) * MINDR)
    i = np.repeat(np.arange(N).reshape(1,-1), k, axis=0).reshape(-1)
    adj = np.zeros_like(dr)
    adj[i, nbors] = invdist
    adj[nbors, i] = invdist
    return adj 


class Event(object):
    __slots__ = ['x','adj','y','N']
    def __init__(self, hard_rec, pu_rec, npu, grid=None):
        hard = Interaction(hard_rec, 0)
        pus = [Interaction(pu_rec, i+1, 1) for i in xrange(npu)]

        ints = [hard] + pus

        if grid is not None:
            grid.clear()
            neutrals = []
            for i in ints:
                neutrals += i.get_neutrals()
            ns = grid.run(neutrals)
            hard.neutral = ns[0]
            for i,p in enumerate(pus):
                p.neutral = ns[i+1]
        
        x_list = [i.x for i in ints]
        self.x = np.concatenate(x_list, axis=0)

        # pt-ordering
        idx = np.argsort(self.x[:,0], axis=0)
        idx = np.flip(idx, axis=0)
        
        y_list = [i.y for i in ints]
        self.y = np.concatenate(y_list, axis=0)

        self.adj = get_adj(self.x[:,1:3], k=5)

        # now sort everything

        N = self.x.shape[0]
        self.N = N
#        idx = np.arange(self.N)
        idx_full = idx
        if N > MAXPARTICLES:
            idx = idx[:MAXPARTICLES]
            N = MAXPARTICLES

        self.x = self.x[idx]
        self.y = self.y[idx]
        self.adj = self.adj.reshape(-1)[np.add.outer(
                        self.N*idx_full, idx_full
                    ).reshape(-1)].reshape(self.N, self.N)
        self.adj = self.adj[:MAXPARTICLES,:MAXPARTICLES]

        self.adj = scipy.sparse.coo_matrix(self.adj).tocsr()

        if N < MAXPARTICLES:
            pad = MAXPARTICLES - N
            embed = self.x
            self.x = np.zeros((MAXPARTICLES, embed.shape[1]))
            self.x[:N,:] = embed 

            embed = self.y
            self.y = np.zeros((MAXPARTICLES,))
            self.y[:N] = embed 

            self.adj = np.pad(self.adj, pad_width=(0, pad), mode='constant', constant_values=0)

def saveto(xs, ys, adjs, Ns):
    global IFILE
    outpath = args.output.replace('.npz','_%i.npz'%IFILE)

    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    N = np.stack(Ns, axis=0)
    adj = np.array(adjs)

    print 'x', x.shape
    print 'y', y.shape
    print 'adj', adj.shape
    print '-->', outpath
    sys.stdout.flush()

    if MAXPARTICLES < 150:
        np.set_printoptions(threshold=np.inf)
        print adj[0]
        print np.concatenate([x[0], y[0][:,np.newaxis]], axis=-1)

    np.savez(outpath,
             x=x,
             N=N,
             y=y,
             adj=adj)

    IFILE += 1


if __name__ == '__main__':

    hard_rec = Record(args.hard)
    mb_rec = Record(args.minbias)
    grid = Grid(args.npu+1)

    xs = []
    ys = []
    adjs = []
    Ns = []
    n_total = 0
    for n_total in trange(args.nmax):
#        try:
            e = Event(hard_rec, mb_rec, args.npu, grid)
            xs.append(e.x)
            ys.append(e.y)
            adjs.append(e.adj)
            Ns.append(e.N)
            n_total += 1
            if n_total == args.nmax:
                break
            if len(xs) == FILESIZE:
                saveto(xs, ys, adjs, Ns)
                xs, ys, adjs, Ns = [], [], [], []
                print
    saveto(xs, ys, adjs, Ns)
#        except Exception as e:
#            print str(e)
#            break

