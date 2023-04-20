import sys
import os
import time
import string

from itertools import product,permutations,chain
from sortedcontainers import SortedDict
import copy

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection,PatchCollection
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.collections as mc

from sklearn.metrics import euclidean_distances
import numba

from sklearn.manifold import smacof
#from _mds import *

#segements from br_adj


def nn_neighbors(csr_adj,nn):
    """
    finds the graph neighbors of each
    vertex out to radius nn, inclusive
    """
    tot = csr_adj

    for i in range(1,nn):
        tot = csr_adj + csr_adj @ tot

    tot = tot.tolil()
    tot.setdiag(0)

    dk = tot.todok()

    #get indices of non-zero elements in dk
    irow,icol = list(zip(*dk.keys()))

    #set all non-zero elements to 1
    #so now have simple adjacency matrix for nn neighbor
    #graph of csr_adj
    dk[irow,icol] = 1.0
    
    dk = dk.tocsr()
    rsums = dk.sum(axis=1)

    deg = rsums.copy()

    rsums = np.array(rsums)

    rsums = rsums**(-1)
    N = tot.shape[0]

    tot2 = sp.lil_matrix((N,N))
    tot2.setdiag(rsums)
    tot2 = tot2.tocsr()

    #tot2 is now averaging matrix
    tot2 = tot2 @ dk


    return deg,tot2


def do_avg(X,A,navg):
    """
    #A is the one time averaging matrix
    #final X is (A^navg) @ X
    """

    for i in range(navg):
        X = A @ X
    return X


def mk_df_var(csk,tree_radius,navg):
    """
    compute variance for each avg cell
    """

    deg,A = nn_neighbors(csk.csr_mst,tree_radius)

    X0 = csk.df.loc[:,csk.traj_markers].values
    X = do_avg(X0,A,navg)

    """
    check average is same as in project
    Xavg = csk.df_avg.loc[:,csk.traj_markers].values

    all_ok = np.allclose(X,Xavg)
    if not all_ok:
        print("incompatible average")
        return
    else:
        print("all ok")
    """
    
    X2_0 = X0 * X0
    X2 = do_avg(X2_0,A,navg)
    Xvar = X2 - X*X

    #just in case some elements are slightly negative
    sel = Xvar < 0    
    if np.sum(sel) > 0:
        Xvar[sel] = 0.0

    df_var = csk.df_avg.copy()
    df_var.loc[:,csk.traj_markers] = Xvar

    return df_var





def avg_weights(A,cells0,navg):
    """
    """
    Alil = A.tolil()
    v0lay = cells0.copy()
    v0lay.sort()
    mat_list = []

    print("loop")

    t0 = time.time()

    #we will use A itself for the last factor
    for n_lev in range(navg-1):
        B1 = sp.lil_matrix(A.shape)
        B1[v0lay,:] = Alil[v0lay,:]

        v1lay = set(v0lay)

        for row in B1.rows:
            v1lay.update(row)

        v1lay = list(v1lay)
        v1lay.sort()

        print("nnz",B1.nnz)

        mat_list.append(B1.tocsr())        
        v0lay = v1lay

    t1 = time.time()
    print("loop time",t1 - t0)

    mat_list.append(A)
    W = mat_list[0]

    for i in range(1,len(mat_list)):
        W = W @ mat_list[i]

    #W.data **= 2

    wmax = W.max()
    wmin = W.min()
    print(wmin,wmax)

    t2 = time.time()

    print("W time",t2-t1)

    print("W",W.nnz)

    return W


def mk_W(csk,tree_radius,navg):
    deg,A = nn_neighbors(csk.csr_mst,tree_radius)

    br_cells = list(csk.cg.br_adj)

    W = avg_weights(A,br_cells,navg)
    return W
    



def fends(adj,start):
    """
    adj is a tree
    start is any end
    """

    degree = SortedDict()
    parent = SortedDict()
    stack = [start]
    parent[start] = -1

    far_ends = []
    while len(stack) > 0:
        v = stack.pop()
        count = 0

        #look at all the neighbors of the latest vertex
        for u in adj[v]:
            #don't want to go backward in tree
            if u != parent[v]:
                parent[u] = v
                stack.append(u)
                count += 1
        #count == 0 means that we are at a branch end
        if count == 0:
            far_ends.append(v)
        degree[v] = count +1

    #have to set degree[start] here
    degree[start] = 1

    return parent,far_ends,degree


def get_seg_lengths(segments):
    seg_lengths = {}
    for vp in segments:
        seg = segments[vp]
        slen = len(seg)
        seg_lengths[vp] = slen
        seg_lengths[(vp[1],vp[0])] = slen

    return seg_lengths


def get_segments(br_adj):
    adj = br_adj

    ends = []
    bpts = []

    for v in br_adj:
        vadj = br_adj[v]
        alen = len(vadj)

        if alen == 1:
            ends.append(v)
        elif alen > 2:
            bpts.append(v)

    #stack = [start]
    #we use a dict for the parent points
    #so every node in the tree has a parent back towards start
    parent = {}
    degree = {}

    parent,far_ends,degree = fends(adj,ends[0])

    #now construct reduced graph by working backwards
    reduced_adj = SortedDict()
    segments = {}
    edges = set([])
    fw_edges = set()

    #construct segments_sym  and reduced_adj
    for v in far_ends:
        v0 = v
        v1 = parent[v0]
        seg = [v0]
        while v1 != -1:
            seg.append(v1)
            if degree[v1] != 2:
                if (v0,v1) in segments:
                    #edge already done
                    break
                if (v0,v1) in segments:
                    print("duplicating",v0,v1)
                edges.add((v0,v1))
                edges.add((v1,v0))
                fw_edges.add((v1,v0))                    
                segments[(v0,v1)] = seg
                segments[(v1,v0)] = seg[::-1]
                v0 = v1
                seg = [v0]
            v1 = parent[v1]

    for e in edges:
        if e[0] in reduced_adj:
            reduced_adj[e[0]].append(e[1])
        else:
            reduced_adj[e[0]] = [e[1]]


    return segments


#frechet code

#fill c and spokes
@numba.njit()
def do_spokes(dist,c,spokes):
    n0 = dist.shape[0]
    n1 = dist.shape[1]
    I = min(n0,n1)
    
    for i0 in range(1,n0):
        spokes[i0,0] = [i0-1,0]
        c[i0,0] = max( dist[i0,0],c[i0-1,0])

    for i1 in range(1,n1):
        spokes[0,i1] = [0,i1-1]
        c[0,i1] = max( dist[0,i1], c[0,i1-1])

    for ii in range(1,I):
        iprev = [ [ii-1,ii-1], [ii,ii-1], [ii-1,ii] ]
        cprev= np.array( [ c[ii-1,ii-1], c[ii,ii-1], c[ii-1,ii] ] )
        amin = np.argmin(cprev)
        spokes[ii,ii] = iprev[amin]
        xmin = cprev[amin]
        c[ii,ii] = max(dist[ii,ii],xmin)

        for i0 in range(ii+1,n0):
            iprev = [ [i0-1,ii], [i0-1,ii-1], [i0,ii-1] ] 
            cprev = np.array([ c[i0-1,ii], c[i0-1,ii-1], c[i0,ii-1] ] )
            amin = np.argmin(cprev)
            spokes[i0,ii] = iprev[amin]
            xmin = cprev[amin]
            c[i0,ii] = max(dist[i0,ii],xmin)

        for i1 in range(ii+1,n1):
            iprev = [ [ii,i1-1], [ii-1,i1-1], [ii-1,i1 ] ]
            cprev = np.array([ c[ii,i1-1], c[ii-1,i1-1], c[ii-1,i1 ] ] )
            amin = np.argmin(cprev)
            spokes[ii,i1] = iprev[amin]
            xmin = cprev[amin]  
            c[ii,i1] = max(dist[ii,i1],xmin)


def frechet(x0,x1):

    n0 = x0.shape[0]
    n1 = x1.shape[0]

    idx0 = np.arange(x0.shape[0])
    idx1 = np.arange(x1.shape[0])

    #get all pairwise distances
    dist = np.zeros( (n0, n1) )
    edges = list(product(idx0,idx1))
    edges = np.array(edges)
    dX = x0[edges[:,0]] -x1[edges[:,1]]
    dist = la.norm(dX , axis=1)
    dist = dist.reshape( (n0,n1) )

    c = np.zeros(dist.shape)
    I = min(n0,n1)

    c[0,0] = dist[0,0]
    spokes = np.full( (n0,n1,2),-1)
    spokes[0,0] = [0,0]


    do_spokes(dist,c,spokes)    

    s = [n0-1,n1-1]
    back = [s]

    while s[0] != 0 and s[1] != 0:
        s = spokes[s[0],s[1]]
        back.append(s)

    back = np.array(back)
    back = back[::-1]

    return spokes,back,c


def get_perms(adj):

    #alist is a list of lists
    #each element list is a list of
    #forward vertices of a tree vertex
    alist = list(adj.values())

    import math

    num = 1

    for a in alist:
        na = len(a)
        num = num * math.factorial(na)
        #print(na)

    #num is the number of possible permuations
    print("num",num)


    #for each element of alist
    #add to plist the list of all its
    #permuations
    plist = []
    for a in alist:
        p = permutations(a)
        #p is an iterator, need to make list
        plist.append(list(p))

    #each element of glist is a permuation
    #of each adjaccency list in adj
    glist = list( product(*plist) )

    keys =list( adj.keys())

    #adj_list is list of permuted tree graphs
    adj_list = []
    for g in glist:
        adj0 = SortedDict( zip(keys,g) )
        adj_list.append(adj0)

    return adj_list



#mds code

def _smacof(
        _d,
        _w,
        n_components=2,
        init=None,
        max_iter=300,
        eps=1e-3):
    n_samples = _d.shape[0]

    if init is None:
        # Randomly choose initial configuration
        X = np.random.rand(n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError("init matrix should be of shape (%d, %d)" %
                             (n_samples, n_components))
        X = init

    wrow = np.sum(_w,axis=1).ravel()

    U = np.diag(wrow) - _w
    Upinv = la.pinv(U)
    t0 = _w * _d

    old_stress = None
    for it in range(max_iter):
        dis = euclidean_distances(X)
        stress = ((dis.ravel() - _d.ravel()) ** 2).sum() / 2
            
        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        T = t0/dis
        trow = np.sum(T,axis=1).ravel()
        T = np.diag(trow) - T

        X = Upinv @ (T @ X)

        dis = np.sqrt((X ** 2).sum(axis=1)).sum()

        if old_stress is not None:
            if(old_stress - stress / dis) < eps:
                break
        old_stress = stress / dis

    xdist = euclidean_distances(X)

    return X, stress, it + 1, xdist


def smacof2(_d, *,n_components=2,
                init=None,
                weights = None,
           n_init=8, max_iter=300,eps=1e-3,
           random_state=None, return_n_iter=False):

    print("smacof2 n_init",n_init)

    if hasattr(init, '__array__'):
        init = np.asarray(init).copy()

        print("init",init[0])


    if hasattr(weights, '__array__'):
        _w = np.asarray(weights).copy()
    else:
        _w = np.ones(_d.shape)

    best_pos, best_stress = None, None

    for it in range(n_init):
        pos, stress, n_iter_ ,xdist = _smacof(
            _d,
            _w,
            n_components=n_components, init=init,
            max_iter=max_iter,
            eps=eps)
        
        if best_stress is None or stress < best_stress:
            best_stress = stress
            best_pos = pos.copy()
            best_iter = n_iter_

    
    if return_n_iter:
        return best_pos, best_stress, xdist, best_iter
    else:
        return best_pos, best_stress, xdist



def add_arch(A,B):
    #find point in A with furthest sum from B points
    N = A.shape[0] ; p = A.shape[1]
    NB = B.shape[0]

    clist = []
    for i in range(p):
        C = np.subtract.outer(A[:,i],B[:,i])
        clist.append(C)

    D = np.array(clist)
    D = np.swapaxes(D,0,2)

    #D now has shape (NB,N,p)
    dist = la.norm(D,axis=2)
    dist = np.sum(dist,axis = 0)
    i1 = np.argmax(dist)

    return i1


def add_one_arch(X):
    #just add n_arch more

    N = X.shape[0]
    n_arch = 1
    ilist = [0,N-1]


    sel = np.full(N,True,dtype=np.bool)
    for i in ilist:
        sel[i] = False

    nsel = np.logical_not(sel)

    idx = np.arange(N)

    not_idx = idx[nsel]
    idx = idx[sel]

    A = X[idx]
    B = X[not_idx]
    i1 = idx[add_arch(A,B)]

    return i1



def harch(X,nlev):
    N = X.shape[0]

    ilist = [0,N-1]
    ilist2 = [0]

    for ilev in range(nlev):
        n_list = len(ilist)
        for i in range(n_list-1):
            j0 = ilist[i]
            j1 = ilist[i+1]
            X01 = X[j0:j1+1,:]
            i1 = add_one_arch(X01)
            i1 = j0 + i1

            ilist2.append(i1)
            ilist2.append(j1)

        ilist = ilist2
        ilist2 = [0]

    return ilist




def get_edges(br_adj):
    edges = []

    for v0 in br_adj:
        links = br_adj[v0]
        for v1 in links:
            if v1 < v0: continue
            edges.append((v0,v1))

    edges = np.array(edges,dtype=int)
    return edges




class mdsxx:
    def __init__(self,df_avg,df_seg0,icells,edges,init=None,seed=137,xfac=1.0,wfac=1.0):
        map0,rmap0 = mk_maps(df_avg,icells)
        edges_0 = rmap0[edges]

        print("wfac",wfac)

        self.e0 = edges_0[:,0]
        self.e1 = edges_0[:,1]

        X = df_seg0.loc[icells,:].values
        dist = euclidean_distances(X)

        edist = dist[self.e0,self.e1]

        print("edist",edist[:5])

        weights = np.ones(dist.shape)
        if wfac != 1.0:
            print("weights")
            weights[self.e0,self.e1] = wfac
            weights[self.e1,self.e0] = wfac        

        if init is None:
            np.random.seed(seed)

        #n_init should not be 1 for random seed case
        if wfac == 1.0:
            ux2,stress,n_iter = smacof(dist,init=init,n_init=1,return_n_iter=True)
        else:
            ux2,stress,xdist,n_iter = smacof2(dist,init=init,n_init=1,return_n_iter=True,weights=weights)
        #ux2,stress,xdist,n_iter = smacof2(dist,init=init,n_init=1,return_n_iter=True,weights=None)

        print("n_iter",n_iter)

        """
        exdist = xdist[self.e0,self.e1]

        eratio = exdist/edist

        print(np.amax(eratio), np.amin(eratio))


        plt.hist(eratio,bins=50,log=True)
        plt.show()
        """

        
        #ux2,stress,n_iter = smacof2(dist,return_n_iter=True,weights=weights) #here init is None
        ux2[:,0] = xfac*ux2[:,0]
        

        segs = ux2[edges_0]
        segs0 = np.mean(segs,axis=1,keepdims=True)
        dsegs = segs - segs0
        segs = .95*dsegs + segs0

        self.df_avg = df_avg
        self.ux2 = ux2
        self.segs = segs
        self.edges_0 = edges_0

        self.map0 = map0 #really same as icells
        self.rmap0 = rmap0


    def plot0(self,nrow,ncol,clist,show_ax = False):

        df_avg = self.df_avg
        ux2 = self.ux2
        segs = self.segs
        e0 = self.e0
        e1 = self.e1

        map0 = self.map0

        """
        #previous method
        pfac = 2.8

        ww = pfac*ncol
        hh = pfac*nrow

        ww = 12
        hh = 9
        """

        dh = 8.5/nrow
        dw = 17.0/ncol
        
        
        pfac = min(dh,dw)

        ww = pfac*ncol
        hh = pfac*nrow

        fig,axes = plt.subplots(nrow,ncol,figsize=(ww,hh))
        if nrow*ncol > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        for i,ax in enumerate(axes):

            if i >= len(clist):

                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])        
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                continue

            color = df_avg.loc[map0,clist[i]].values
            vmax = np.amax(color)
            ecolor0 = color[e0]
            ecolor1 = color[e1]
            ecolor = .5*(ecolor0 + ecolor1)

            ecolor = ecolor/vmax

            pnts = ax.scatter(ux2[:,0],ux2[:,1],s=15,c=color,cmap=mpl.cm.jet,vmin=0.0)

            rgba = mpl.cm.jet(ecolor)

            seg_col = mc.LineCollection(segs,color=rgba)
            #seg_col = mc.LineCollection(segs)
            fig.colorbar(pnts,ax=ax)
            ax.add_collection(seg_col)
            ax.set_xlabel(clist[i])

            #ax.axis('equal')

            if not show_ax:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)

        #plt.savefig("markers1.pdf",format='pdf',bbox_inches='tight')
        #plt.savefig("receptors_homing.pdf",format='pdf',bbox_inches='tight')
        plt.tight_layout()
        plt.show()


    def plot1(self,fig,ax,mcol0):
        #mcol is categorical here

        df_avg = self.df_avg
        ux2 = self.ux2
        segs = self.segs
        e0 = self.e0
        e1 = self.e1

        map0 = self.map0

        cat = df_avg.loc[map0,mcol0].values
        cats = list(set(cat))
        cats.sort()

        print("cats",cats)

        for cat0 in cats:
            sel_cat = cat == cat0
            print(np.sum(sel_cat),cat0)

            ux_sel = ux2[sel_cat]

            pnts = ax.scatter(ux_sel[:,0],ux_sel[:,1],s=20,label=cat0)

        #pnts = ax.scatter(ux2[:,0],ux2[:,1],s=15,c=color,cmap=mpl.cm.jet,vmin=0.0,marker='^')

        #rgba = mpl.cm.jet(ecolor)

        #seg_col = mc.LineCollection(segs,color=rgba)
        seg_col = mc.LineCollection(segs,color=(0,0,0,.3))
        #seg_col = mc.LineCollection(segs)
        #fig.colorbar(pnts,ax=ax)
        ax.add_collection(seg_col)
        ax.set_xlabel(mcol0)

        ax.legend()

        #ax.axis('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)

        
    def plot1_0(self,fig,ax,mcol):

        df_avg = self.df_avg
        ux2 = self.ux2
        segs = self.segs
        e0 = self.e0
        e1 = self.e1

        map0 = self.map0

        color = df_avg.loc[map0,mcol].values
        vmax = np.amax(color)
        ecolor0 = color[e0]
        ecolor1 = color[e1]
        ecolor = .5*(ecolor0 + ecolor1)

        ecolor = ecolor/vmax

        pnts = ax.scatter(ux2[:,0],ux2[:,1],s=15,c=color,cmap=mpl.cm.jet,vmin=0.0)

        rgba = mpl.cm.jet(ecolor)

        #seg_col = mc.LineCollection(segs,color=rgba)
        seg_col = mc.LineCollection(segs,color=(0,0,0,.3))
        #seg_col = mc.LineCollection(segs)
        fig.colorbar(pnts,ax=ax)
        ax.add_collection(seg_col)
        ax.set_xlabel(mcol)

        #ax.axis('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)


def cpoints0(csk):
    br_adj = csk.cg.br_adj

    branchp = []
    endp = []

    for v in br_adj:
        adj = br_adj[v]
        alen = len(adj)
        if alen == 1:
            endp.append(v)
        if alen > 2:
            branchp.append(v)

    return endp,branchp

def mk_maps(df,pcells):

    map0 = np.array(pcells,dtype=int)
    npcells = len(pcells)
    idx0 = np.arange(npcells)

    N = df.shape[0]

    rmap0 = np.full((N,),-1,dtype=np.int)
    rmap0[map0] = idx0

    return map0,rmap0

"""
def crit_segs_dx(tj1):
    N = tj1.csk.df.shape[0]    
    crit,crit_segs = tj1.do_critical(tj1.csk.cg.segments)
    #crit,crit_segs = tj1.do_critical(crit_segs,do_all=True)
    crit = crit - N

    return crit,crit_segs
"""


def add_critical(df_seg0,segments):
    """
    df_seg0 is data frame only for dist markers
    """
    ind = set()

    crit_segs = {}

    for vp in segments:
        seg = np.array(segments[vp],dtype=int)
        X = df_seg0.loc[seg,:].values
        N = X.shape[0]

        if len(seg) >= 3:
            ilist = harch(X,1)
            imid = ilist[1]
        else:
            crit_segs[vp] = seg
            continue

        xlen = la.norm(X[-1] - X[0])
        xlen_m1 = la.norm(X[-1] - X[imid])
        xlen_0m = la.norm(X[imid] - X[0])
        xfac = (xlen_0m + xlen_m1) / xlen

        if xfac > 1.25:
            seg0 = seg[:ilist[1]]
            seg1 = seg[ilist[1]-1:]

            vp0 = tuple(seg0[ [0,-1] ])
            vp1 = tuple(seg1[ [0,-1] ])

            crit_segs[vp0] = seg0
            crit_segs[vp1] = seg1
        else:
            crit_segs[vp] = seg

    return crit_segs


class critical0:
    """
    gathers critical points:
    end points, branch points
    and turning points
    
    adds turning points if n_iter > 0

    Also creates adjacency list for
    critical point graph
    """

    def __init__(self,csk,n_iter=1):
        crit_segs0 = {}

        for vp in csk.cg.segments:
            if vp[0] > vp[1]: continue
            crit_segs0[vp] = csk.cg.segments[vp]


        len0 = len(crit_segs0)

        df_seg0 = csk.df_avg.loc[:,csk.traj_markers]
        crit_segs1 = copy.deepcopy(crit_segs0)
        for i in range(n_iter):
            crit_segs1 = add_critical(df_seg0,crit_segs1)
            len1 = len(crit_segs1)
            if len1 == len0: break
            len0 = len1

        crit_pnts = set()
        crit_adj = SortedDict()

        for vp in crit_segs1:
            crit_pnts.update(vp)

        for v in crit_pnts:
            crit_adj[v] = set()

        for vp in crit_segs1:
            crit_adj[vp[0]].add(vp[1])
            crit_adj[vp[1]].add(vp[0])


        for v in crit_adj:
            crit_adj[v] = list(crit_adj[v])

        self.crit_pnts = np.array( list(crit_adj.keys()) )

        self.crit_adj,self.crit_segs1 = crit_adj,crit_segs1

        #get the end pnts
        endp = []
        turnp = []
        branchp = []
        for v in crit_adj:
            alen = len(crit_adj[v])
            if alen == 1:
                endp.append(v)
            elif alen == 2:
                turnp.append(v)
            elif alen > 2:
                branchp.append(v)

        self.endp = endp
        self.turnp = turnp
        self.branchp = branchp


    def in_segs(self):
        """
        creates list of inward segments from
        all endpoints
        in_segments
        """
        br_adj = self.crit_adj
        endp = self.endp
        segments = self.crit_segs1

        in_segments = {}

        for vp in segments:
            seg = segments[vp]
            if vp[0] in endp:
                in_segments[vp] = seg
            if vp[1] in endp:
                in_segments[(vp[1],vp[0])] = seg[::-1]

        print("in",list(in_segments.keys()))

        self.in_segments = in_segments


class mds00:
    def __init__(self,csk,n_iter,seed=3738,crit_iter=0):
        
        crit0 = critical0(csk,crit_iter)
        crit_adj,crit_segs = crit0.crit_adj,crit0.crit_segs1
        crit_pnts = crit0.crit_pnts

        br_adj = csk.br_adj
        tedges = get_edges(br_adj)
        br_cells = list(br_adj.keys())
        df_avg = csk.df_avg

        X = df_avg.loc[br_cells,csk.traj_markers].values
        idx0 = np.arange(X.shape[0])
        map0 = np.array(br_cells,dtype=int)
        rmap0 = np.full((df_avg.shape[0],),-1,dtype=int)
        rmap0[map0] = idx0

        dist = euclidean_distances(X)


        #this seed gives result used for Roshni paper
        #np.random.seed(3738)
        np.random.seed(seed)

        t0 = time.time()
        #this is my version of smacof in mds_tree_1
        self.ux2,stress,xdist,n_iter = smacof2(dist,return_n_iter=True)

        t1 = time.time()

        self.tedges = tedges
        self.tedges0 = rmap0[tedges]
        self.segs = self.ux2[self.tedges0]

        crit_pnts0 = rmap0[crit_pnts]
        self.crit_pnts0 = crit_pnts0
        self.xcrit = self.ux2[crit_pnts0]

        self.df_color = df_avg.loc[br_cells,:]

        self.rmap0 = rmap0

        

    def mplot(self,clist,nrow,ncol,do_xcrit=None):
        ux2 = self.ux2
        tedges0 = self.tedges0
        segs = self.segs
        xcrit = self.xcrit
        df_color = self.df_color
        mds_plot(df_color,ux2,segs,tedges0,nrow,ncol,clist,xcrit,do_xcrit)        
    

    def gray(self,clist,nrow,ncol):
        ux2 = self.ux2
        tedges0 = self.tedges0
        segs = self.segs
        xcrit = self.xcrit
        df_color = self.df_color
        crit_pnts0 = self.crit_pnts0
        gray_plot(df_color,ux2,segs,tedges0,nrow,ncol,clist,xcrit,crit_pnts0)

        


class mds_tree:
    def __init__(self,csk):

        br_adj = csk.cg.br_adj
        crit_segs0 = {}
        self.csk = csk

        self.traj_markers = csk.traj_markers

        for vp in csk.cg.segments:
            if vp[0] > vp[1]: continue
            crit_segs0[vp] = csk.cg.segments[vp]

        self.csk = csk
        self.br_adj = br_adj
        self.crit_segs0 = crit_segs0
        

        self.df_avg = csk.df_avg
        self.tree_cells = np.array( list( br_adj.keys() ), dtype=int)
        self.endp,self.branchp = cpoints0(csk)


    def mk_tree(self,seed=137,level=1,xfac=1.0):
        df_seg0 = self.df_avg.loc[:,self.traj_markers]
        crit_segs1 = copy.deepcopy(self.crit_segs0)

        for i in range(level):
            crit_segs1 = add_critical(df_seg0,crit_segs1)
        #crit_segs1 = self.crit_segs0

        crit_edges = list(crit_segs1.keys())
        crit_edges = np.array(crit_edges,dtype=int)
        crit_cells = list(set(crit_edges.flatten()))
        crit_cells.sort()

        mds = mdsxx(self.df_avg,df_seg0,crit_cells,crit_edges,seed=137)
        #mds.plot0(nrow,ncol,clist)

        #tscale = 1
        #ux4_0,td,tedges4 = ux_init2(self.csk,tscale)

        ux2 = mds.ux2
        crit_edges_0 = mds.edges_0
        #set up the init for the full mds
        map0,rmap0 = mk_maps(df_seg0,self.tree_cells)


        ux4_0 = np.zeros((len(self.tree_cells),2))

        
        for i,edge in enumerate(crit_edges_0):
            x0 = ux2[edge[0]]
            x1 = ux2[edge[1]]

            dx = (x1 - x0).reshape((1,2))
            vp = tuple(crit_edges[i])

            iseg = crit_segs1[vp]
            nseg = len(iseg)

            iseg0 = rmap0[iseg]

            s = np.linspace(0.0,1.0,nseg).reshape( (nseg,1) )
            dux = s @ dx
            dux = dux + x0
            ux4_0[iseg0] = dux


        tedges = get_edges(self.br_adj)
        mds = mdsxx(self.df_avg,df_seg0,self.tree_cells,tedges,init=ux4_0,xfac=xfac,wfac=1.0)

        """
        #look for long edges

        usegs = mds.ux2[mds.edges_0]

        print(usegs.shape)

        dux = usegs[:,1,:] - usegs[:,0,:]

        seglens = la.norm(dux,axis=1)

        idx = np.argsort(seglens)

        print(np.amax(seglens),np.mean(seglens))

        print(seglens[idx[-1]])

        print(mds.edges_0[idx[-1]])

        print(tedges[idx[-1]])

        exit()
        """
        self.mds= mds

        print("mk_tree")

    def plot(self,clist,nrow,ncol):
        nrow = int(nrow); ncol = int(ncol)
        tedges = get_edges(self.br_adj)
        self.mds.plot0(nrow,ncol,clist)        

    def plot2(self,clist,nrow,ncol):
        nrow = int(nrow); ncol = int(ncol)
        tedges = get_edges(self.br_adj)
        self.mds.plot0(nrow,ncol,clist)        


    def run0(self,clist,nrow,ncol,seed=137):
        df_seg0 = self.df_avg.loc[:,self.traj_markers]
        crit_segs1 = add_critical(df_seg0,self.crit_segs0)

        crit_edges = list(crit_segs1.keys())
        crit_edges = np.array(crit_edges,dtype=int)
        crit_cells = list(set(crit_edges.flatten()))
        crit_cells.sort()

        mds = mdsxx(self.df_avg,df_seg0,crit_cells,crit_edges,seed=137)
        #mds.plot0(nrow,ncol,clist)

        ux2 = mds.ux2
        crit_edges_0 = mds.edges_0
        #set up the init for the full mds
        map0,rmap0 = mk_maps(df_seg0,self.tree_cells)       
        ux4_0 = np.zeros((len(self.tree_cells),2))

        for i,edge in enumerate(crit_edges_0):
            x0 = ux2[edge[0]]
            x1 = ux2[edge[1]]

            dx = (x1 - x0).reshape((1,2))
            vp = tuple(crit_edges[i])

            iseg = crit_segs1[vp]
            nseg = len(iseg)

            iseg0 = rmap0[iseg]

            s = np.linspace(0.0,1.0,nseg).reshape( (nseg,1) )
            dux = s @ dx
            dux = dux + x0
            ux4_0[iseg0] = dux

        tedges = get_edges(self.br_adj)
        #mds = mdsxx(self.df_avg,df_seg0,self.tree_cells,tedges,init=ux4_0)
        mds = mdsxx(self.df_avg,df_seg0,self.tree_cells,tedges)
        mds.plot0(nrow,ncol,clist)




def gray_plot(df_colors,ux2,segs,tedges0,nrow,ncol,clist,xcrit,crit_points0):
    pfac = 2.8

    ww = pfac*ncol
    hh = pfac*nrow

    e0 = tedges0[:,0]
    e1 = tedges0[:,1]

    print("mds plot",nrow,ncol)

    lstr = list(string.ascii_uppercase)


    fig,axes = plt.subplots(nrow,ncol,figsize=(ww,hh))

    axes = axes.flatten()

    for i,ax in enumerate(axes):

        if i >= len(clist):

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])        
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            continue

        color = df_colors.loc[:,clist[i]].values
        vmax = np.amax(color)
        ecolor0 = color[e0]
        ecolor1 = color[e1]
        ecolor = .5*(ecolor0 + ecolor1)

        ecolor = ecolor/vmax

        rgba = mpl.cm.jet(ecolor)

        crit_color = color[crit_points0]/vmax
        crit_color = mpl.cm.jet(crit_color)

        #pnts = ax.scatter(ux2[:,0],ux2[:,1],s=25,c=color,cmap=mpl.cm.jet,vmin=0.0)

        ax.scatter(xcrit[:,0],xcrit[:,1],c='w')

        for j in range(xcrit.shape[0]):
            ax.text(xcrit[j,0],xcrit[j,1],lstr[j],ha='center',va='center',fontsize=12,weight='bold',c=crit_color[j])



        seg_col = mc.LineCollection(segs,color=rgba)
        #seg_col = mc.LineCollection(segs)
        #fig.colorbar(pnts,ax=ax)
        ax.add_collection(seg_col)
        ax.set_xlabel(clist[i])

        #ax.axis('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)

    #fig.suptitle("Chemo Cyto Rectors",fontsize=16)

    #plt.savefig("markers1.pdf",format='pdf',bbox_inches='tight')
    #plt.savefig("receptors_homing.pdf",format='pdf',bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def crit_plot(X,segs,ilist,nlist):
    #originally from tsg/zcrit_basa1_0.py
    tvecs = X.T


    acolors = ['r','g','b','c','tab:blue', 'tab:brown','m','y', 'k','tab:orange']

    #markers = ["." , "," , "o" , "v" , "^" , "<", ">"]    
    markers = ["o" , "v" , "^" , "<", ">"]

    mtypes = list(product(markers,acolors))

    fig,ax = plt.subplots(figsize=(9,9))

    ax.scatter(tvecs[0,:],tvecs[1,:],c='darkgrey',s=10)  

    for i,name in enumerate(nlist):
        print("name",name)
        ii = ilist[i]
        mtyp = mtypes[i]
        ax.scatter(tvecs[0,ii],tvecs[1,ii],edgecolor=mtyp[1],marker=mtyp[0],facecolor='none',s=80,label=name)
        print(mtyp[0])
        #ax.scatter(tvecs[0,csel],tvecs[1,csel],facecolors='none',edgecolors=acolors[i],s=20,label=scol)

    #ax.set_xlabel(m,fontsize=16,fontweight='bold')
    ax.axis('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])        
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)                

    seg_coll = mc.LineCollection(segs,color='k',alpha=.7)
    ax.add_collection(seg_coll)

    ax.legend()
    plt.tight_layout()
    plt.show()



def mds_plot(df_colors,ux2,segs,tedges0,nrow,ncol,clist,xcrit,do_xcrit):
    pfac = 2.8

    ww = pfac*ncol
    hh = pfac*nrow

    e0 = tedges0[:,0]
    e1 = tedges0[:,1]

    fig,axes = plt.subplots(nrow,ncol,figsize=(ww,hh))

    if nrow*ncol > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i,ax in enumerate(axes):

        if i >= len(clist):

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])        
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            continue

        color = df_colors.loc[:,clist[i]].values
        vmax = np.amax(color)
        ecolor0 = color[e0]
        ecolor1 = color[e1]
        ecolor = .5*(ecolor0 + ecolor1)

        ecolor = ecolor/vmax

        #pnts = ax.scatter(ux2[:,0],ux2[:,1],s=25,c=color,cmap=mpl.cm.jet,vmin=0.0)
        pnts = ax.scatter(ux2[:,0],ux2[:,1],s=5,c=color,cmap=mpl.cm.jet,vmin=0.0)

        if do_xcrit: ax.scatter(xcrit[:,0],xcrit[:,1],c='k')

        rgba = mpl.cm.jet(ecolor)

        seg_col = mc.LineCollection(segs,color=rgba)
        #seg_col = mc.LineCollection(segs)
        fig.colorbar(pnts,ax=ax)
        ax.add_collection(seg_col)
        ax.set_xlabel(clist[i])

        #ax.axis('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)

    #fig.suptitle("Chemo Cyto Rectors",fontsize=16)

    #plt.savefig("markers1.pdf",format='pdf',bbox_inches='tight')
    #plt.savefig("receptors_homing.pdf",format='pdf',bbox_inches='tight')
    plt.tight_layout()
    plt.show()




class ftree:
    """
    recursive callable to propagate
    sideways positions of nodes
    from far ends to other nodes
    """
    
    def __init__(self,adj,w):
        self.adj = adj
        self.w = w

    def __call__(self,v0):
        vadj = self.adj[v0]
        nchild = len(vadj)

        if nchild == 0:
            return self.w[v0]
        else:
            self.w[v0] = 0.0
            for v1 in self.adj[v0]:
                self.w[v0] += self.__call__(v1)
            self.w[v0] /= nchild

            return self.w[v0]

class rtree0:
    def __init__(self,adj,start,seg_lengths=None):

        #dictionary of backward parent
        #of each vertex
        parent = {}

        #level of start is 0
        #this is number of edges from
        #start to each vertex
        level = SortedDict()

        #forward adjacency lists
        #for each vertex
        fwd = SortedDict()

        #list of the forward edges
        edges = []

        stack = [start]
        parent[start] = -1
        level[start] = 0
        max_level = 0
        wpos = SortedDict()
        #forward degree
        degree = SortedDict()
        degree[start] = len(adj[start])

        for v in adj:
            fwd[v] = []

        far_ends = []

        #depth first walk thru the adj tree
        while len(stack) > 0:
            v = stack.pop()
            count = 0

            #look at all the neighbors of the latest vertex
            for u in adj[v][::-1]:
                #don't want to go backward in tree
                if u != parent[v]:
                    parent[u] = v
                    edges.append( [v,u] )
                    if seg_lengths:
                        level[u] = level[v] + seg_lengths[(v,u)]
                    else:
                        level[u] = level[v] + 1

                    max_level = max(max_level,level[u])

                        
                    stack.append(u)
                    fwd[v].append(u)
                    count += 1
            #count == 0 means that we are at a branch end
            if count == 0:
                far_ends.append(v)
            degree[v] = count

        scale = 50/max_level

        #done walking the tree, fwd tree in fwd

        if seg_lengths:
            for v in level:
                print("scaling levels")
                level[v] *= scale

        ne = len(far_ends)
        we = .7*np.linspace(0,ne-1,ne)

        for v in adj:
            wpos[v] = 0.0

        for i,v in enumerate(far_ends):
            wpos[v] = we[i]


        f = ftree(fwd,wpos)

        f(start)

        X = np.array([list(wpos.values()),list(level.values()) ] )

        map0 = np.array( list(adj.keys()) )
        N = np.amax(map0) + 1
        rmap0 = np.full( (N,),-1)
        rmap0[map0] = np.arange(map0.shape[0])


        edges = np.array(edges)

        self.edges = edges
        edges0 = rmap0[edges]

        self.map0 = map0
        self.far_ends,self.X,self.edges0,self.rmap0 = far_ends,X,edges0,rmap0
        self.fwd = fwd

        #this gives layout coords for each critical point
        #given its original index
        Xdict = SortedDict()
        ux = X.T
        for i,ix in enumerate(map0):
            Xdict[ix] =ux[i]

        self.Xdict = Xdict


class sdict(SortedDict):
    def __init__(self,fstart):
        super().__init__()

        self.fstart = fstart

class crit_velocities:
    def __init__(self,csk,crit_iter=0):
        c0 = critical0(csk,n_iter = crit_iter)
        self.c0 = c0
        self.csk = csk

    def vtree(self,vstart):
        #construct the forward graph from vstart

        adj = self.c0.crit_adj
        fwd = sdict(vstart)
        parent = SortedDict()
        parent[vstart] = -1

        for v in adj:
            fwd[v] = []
        
        
        stack = [vstart]

        while len(stack) > 0:
            v = stack.pop()
            count = 0

            #look at all the neighbors of the latest vertex
            for u in adj[v][::-1]:
                #don't want to go backward in tree
                if u != parent[v]:
                    parent[u] = v
                    stack.append(u)
                    fwd[v].append(u)
                    count += 1
        self.fwd = fwd


    def vel(self):
        fwd = self.fwd
        csegs = self.c0.crit_segs1

        self.crit_segs_fwd = {}

        df_avg = self.csk.df_avg
        tcols = self.csk.traj_markers
        acols = self.csk.avg_markers



        X = df_avg.loc[:,tcols].values
        Xa = df_avg.loc[:,acols].values

        acols = np.array(acols)

        all_sig = np.zeros(len(acols),dtype=bool)

        Vdict = SortedDict()
        dx_dict = SortedDict()
        self.Xa = Xa


        for v0 in fwd:
            for v1 in fwd[v0]:

                ds = la.norm(X[v1] - X[v0])
                dx = Xa[v1] - Xa[v0]
                avg = .5*(Xa[v1] + Xa[v0]) +1e-12
                #V = dx/ds
                V = dx/avg
                #V = dx

                Vdict[(v0,v1)] = V

                dx_dict[(v0,v1)] = dx


                vp0 = (v0,v1)
                vp1 = (v1,v0)

                #fix the segments
                if vp0 in csegs:
                    seg = csegs[vp0]
                elif vp1 in csegs:
                    seg = csegs[vp1][::-1]                    
                else:
                    print("not found")

                self.crit_segs_fwd[vp0] = seg

        self.Vdict = Vdict
        self.dx_dict = dx_dict

        self.br_err = br_error(self.csk)


    def one_seg(self,vp,table_name=None,csv_name=None):
        df_err = self.br_err.df_err
        df_expr = self.br_err.df_expr

        mcols = df_expr.columns[:-1]

        #print(mcols)

        df_err.set_index('br_cells',inplace=True)
        df_expr.set_index('br_cells',inplace=True)        

        err0 = df_err.loc[vp[0],:]
        err1 = df_err.loc[vp[1],:]

        expr0 = df_expr.loc[vp[0],:]
        expr1 = df_expr.loc[vp[1],:]

        dx = expr1 - expr0

        adx = np.abs(dx)

        sel = adx > .05
        #print("sel",np.sum(sel))

        

        err = np.sqrt(err0 + err1)

        avg = .5*(expr1 + expr0)

        dx2 = dx/(avg + 1e-10)

        sdx = dx[sel]
        scols = mcols[sel]

        #print(scols)

        #print(sdx)

        sdx2 = dx2[sel]

        serr = err[sel]

        savg = avg[sel]

        idx = np.argsort(sdx2)
        sdx = sdx[idx]
        sdx2 = sdx2[idx]
        serr = serr[idx]
        scols = np.array(scols[idx]).ravel()

        savg = savg[idx]


        dfp = pd.DataFrame(scols,columns=['marker'])


        #dfp = pd.DataFrame([scols,sdx,sdx2,serr])

        #need to do the numerical part separately
        #so it is not treated as dytype == object
        data = np.array([sdx,savg,sdx2,serr]).T
        dcols = ['dx','avg','dx/avg','err']
        dfp.loc[:,dcols] = data


        dfp = dfp.round(4)

        print(dfp)
        dfp.to_csv(csv_name,index=False)


        df_err.reset_index(inplace=True)
        df_expr.reset_index(inplace=True)                

        #dfp.to_csv("dfp.csv",float_format='%.4f',index=False)
        #dfp.set_index('m',inplace=True)
        #dfp.to_latex(table_name,index=False)





    def vel0(self):
        fwd = self.fwd
        csegs = self.c0.crit_segs1

        df_avg = self.csk.df_avg
        tcols = self.csk.traj_markers
        acols = self.csk.avg_markers



        X = df_avg.loc[:,tcols].values
        Xa = df_avg.loc[:,acols].values

        acols = np.array(acols)

        print(tcols[:12])

        all_sig = np.zeros(len(acols),dtype=bool)

        Vdict = SortedDict()
        dx_dict = SortedDict()
        self.Xa = Xa


        for v0 in fwd:
            for v1 in fwd[v0]:


                ds = la.norm(X[v1] - X[v0])
                dx = Xa[v1] - Xa[v0]
                avg = .5*(Xa[v1] + Xa[v0]) +1e-12
                #V = dx/ds
                V = dx/avg
                #V = dx

                Vdict[(v0,v1)] = V

                dx_dict[(v0,v1)] = dx

                absV = np.abs(V)

                vmax = np.amax(absV)
                vmin = np.amin(absV)

                n_avg = avg > .2

                vg = absV > .5

                sig = np.logical_and(vg,n_avg)


                sig2 = acols[sig]

                

                #if (v0,v1) != (24707,3853):
                all_sig = np.logical_or(all_sig,sig)

                

                nsum = np.sum(sig)

                print("ds",v0,v1,ds,vmax,vmin,nsum)

                print(sig2)

                print()


                vp0 = (v0,v1)
                vp1 = (v1,v0)

                if vp0 in csegs:
                    seg = csegs[vp0]
                elif vp1 in csegs:
                    seg = csegs[vp1]                    
                else:
                    print("not found")

                #print("len",len(seg))

        print("all_sig",np.sum(all_sig))

        sig2 = acols[all_sig]

        self.Vdict = Vdict
        self.dx_dict = dx_dict

        #print(sig2)

                    
            
                    
class fgraph:
    def __init__(self,csk,start,do_lengths=False,crit_iter=0):

        #for frechet
        Xa = csk.df_avg.loc[:,csk.traj_markers].values

        #find all critical points
        c0 = critical0(csk,n_iter = crit_iter)

        #get the inward pointing segments in c0.in_segments
        #which is dictionary indexed by vertex pairs vp
        #vp[0] -> vp[1]
        c0.in_segs()

        isegs = c0.in_segments
        keys = list(isegs.keys())
        n_keys = len(keys)

        ends = []

        N = csk.df_avg.shape[0]

        #for frechet distances of in_segments
        dk = sp.dok_matrix((N,N))

        for i in range(n_keys):
            vp0 = keys[i]
            seg0 = isegs[vp0]
            v0 = vp0[0]
            #xseg0 is traj coords of the vp0 segment
            xseg0 = Xa[seg0,:]

            #now get frechet distance of each in seg
            #to other in segs
            #store distances in dk labeled by start vertices
            #of the in segs
            for j in range(i,n_keys):
                vp1 = keys[j]
                seg1 = isegs[vp1]
                xseg1 = Xa[seg1,:]
                v1 = vp1[0]

                spokes2,back2,fdist = frechet(xseg0,xseg1)
                dk[v0,v1] = fdist[-1,-1]
                dk[v1,v0] = dk[v0,v1]

        print("end frechet done")

        #adj = csk.cg.reduced_adj
        adj = c0.crit_adj


        rt0 = rtree0(adj,start)

        #adj_list is list of permuted tree_graph adjaceny matrices
        #for adj in adj_list, the forward adjacent vertices of each
        #vertex are listed in left to right order in the adj as drawn
        #with the root(start) at the bottom
        adj_list = get_perms(rt0.fwd)


        adj_costs = []

        count = 1
        for adj in adj_list:
            #print("count",count)
            count += 1
            rt = rtree0(adj,start)
            far_ends,X,edges0,rmap0 = rt.far_ends,rt.X,rt.edges0,rt.rmap0
            n = len(far_ends) + 1
            far_ends.append(start)

            cost = 0
            for i in range(n):
                i1 = (i + 1) % n
                v = far_ends[i]
                v1 = far_ends[i1]
                cost += dk[v,v1]
            adj_costs.append(cost)

        adj_costs = np.array(adj_costs)
        amin = np.argmin(adj_costs)

        #the best adj
        adj = adj_list[amin]


        #do rtree0 on best adj
        if do_lengths:
            seg_lengths = get_seg_lengths(c0.crit_segs1)
            rt = rtree0(adj,start,seg_lengths)
        else:
            rt = rtree0(adj,start)        

        far_ends,X,edges0,rmap0 = rt.far_ends,rt.X,rt.edges0,rt.rmap0

        self.tplot_X = X
        self.tplot_edges0 = edges0
        self.tplot_adj = adj
        self.tplot_rmap0 = rmap0


        #tree with cell numbers
        #tplot0(X,edges0,adj,rmap0)


        crit_segs = c0.crit_segs1
        br_adj = csk.br_adj
        br_cells = list(br_adj.keys())
        tedges = get_edges(br_adj)    

        df_avg = csk.df_avg

        idx0 = np.arange(len(br_cells))
        map0 = np.array(br_cells,dtype=int)
        rmap0 = np.full((df_avg.shape[0],),-1,dtype=int)
        rmap0[map0] = idx0


        #ux2 is for refplot and mds_plot
        ux2 = np.zeros((map0.shape[0],2))
        for vp in crit_segs:
            print(vp)
            vp0 = rmap0[np.array(vp)]

            x0 = rt.Xdict[vp[0]]
            x1 = rt.Xdict[vp[1]]
            dx = (x1 - x0).reshape((1,2))

            iseg = crit_segs[vp]
            iseg0 = rmap0[iseg]
            nseg = len(iseg)

            s = np.linspace(0.0,1.0,nseg).reshape( (nseg,1) )
            dux = s @ dx
            dux = dux + x0

            ux2[iseg0] = dux


        df_color = df_avg.loc[br_cells,:]
        crit_pnts = c0.crit_pnts    

        crit_pnts0 = rmap0[crit_pnts]

        xcrit = ux2[crit_pnts0]

        tedges0 = rmap0[tedges]
        segs = ux2[tedges0]

        self.df_color = df_color
        self.ux2 = ux2
        self.segs = segs
        self.tedges0 = tedges0
        self.xcrit = xcrit
        self.crit_pnts0 = crit_pnts0
        


    def refplot(self,nrow,ncol,clist):
        gray_plot(self.df_color,self.ux2,self.segs,self.tedges0,nrow,ncol,clist,self.xcrit,self.crit_pnts0)


    def mds_plot(self,nrow,ncol,clist):
        #should rename this
        mds_plot(self.df_color,self.ux2,self.segs,self.tedges0,nrow,ncol,clist,self.xcrit)            

        #mds_plot(df_color,ux2,segs,tedges0,nrow,ncol,clist,xcrit)    
        
    def tplot0(self):
        #best tree plot with cell numbers
        X = self.tplot_X.T
        edges0 = self.tplot_edges0
        adj = self.tplot_adj
        rmap0 = self.tplot_rmap0

        segs = X[edges0]
        mu_segs = np.mean(segs,axis=1,keepdims=True)
        dsegs = segs-mu_segs
        dsegs = .7 * dsegs
        segs = mu_segs + dsegs
        seg_col = mc.LineCollection(segs,color='r')

        fig,ax = plt.subplots()
        ax.add_collection(seg_col)
        ax.scatter(X[:,0],X[:,1],s=4,c='w')

        for v in adj.keys():
            i = rmap0[v]
            ax.text(X[i,0],X[i,1],str(v),ha='center',va='center',fontsize=8)

        plt.axis('equal')
        plt.show()



class br_error:
    def __init__(self,csk,tree_radius=5,navg=5):
        br_adj = csk.cg.br_adj
        br_cells = list(br_adj.keys())

        self.csk = csk

        have_W = os.path.exists(csk.project_dir+"W.npz")

        if have_W:
            print("loading W")
            W = sp.load_npz(csk.project_dir+"W.npz")
        else:
            W = mk_W(csk,tree_radius,navg)
            sp.save_npz(csk.project_dir+"W.npz",W)

        #compute avg for br_cells using W    
        X0 = csk.df.loc[:,csk.avg_markers].values
        Xavg1 = (W @ X0)[br_cells,:]

        #next do variance

        X2_0 = X0 * X0

        #X2avg = (W @ X2_0)[br_cells,:]
        #variance for comparison
        #Xvar = X2avg - Xavg1*Xavg1


        W2 = W.copy()
        W2.data **= 2
        W2row = np.array(W2.sum(axis=1))[br_cells,:]

        Xerr2 = (W2 @ X2_0)[br_cells,:]
        M = (W2 @ X0)[br_cells,:]
        err = Xerr2 -2*M*Xavg1 + W2row*Xavg1*Xavg1


        print("err",np.amin(err),np.amax(err))


        df_err = pd.DataFrame(err,columns=csk.avg_markers)
        df_expr = pd.DataFrame(Xavg1,columns=csk.avg_markers)

        df_err.loc[:,'br_cells'] = br_cells
        df_expr.loc[:,'br_cells'] = br_cells

        self.df_err = df_err
        self.df_expr = df_expr

    

