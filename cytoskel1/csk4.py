import os
#import logging
import shutil
import pickle

import numpy as np
import pandas as pd
import numpy.linalg as la
import threading

import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.collections import LineCollection,PatchCollection

#from sklearn.neighbors.ball_tree import BallTree as btree
#import sklearn.neighbors.ball_tree as ball_tree
from sklearn.neighbors import BallTree
import scipy.sparse as sp
from itertools import groupby

import time

import queue as Q
import heapq
from sortedcontainers import SortedDict,SortedList
import ast

from .util import *
from .cgraph import *

from .tmap import *

from .tumap import *

from .subway import *

from .ofun import *

from collections import OrderedDict
from matplotlib import cm

import multiprocessing as mp
mp.set_start_method('fork')

import traceback




def tumap_plot(X,segs,df,clist,skip=1,nrow=1,ncol=1):
    tvecs = X.T

    """
    #ntot = len(clist)
    ntot = np.min(nrow*ncol,len(clist))

    if ntot == 1:
        nrow = 1
        ncol = 1
    elif ntot == 2:
        nrow = 1
        ncol = 2
    elif ntot <= 4:
        nrow = 2
        ncol = 2
    else:
        nrow = 2
        ncol = 3
    """

    fig,axes = plt.subplots(nrow,ncol,figsize=(9,4))
    ntot = len(clist)

    if nrow*ncol == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i,ax in enumerate(axes):
        if i >= ntot: continue
        print(clist[i])
        color = df.loc[:,clist[i]].values
        #ax.scatter(tvecs[0,:], tvecs[1,:],c=df.loc[:,clist[i] ], s=.3,cmap=cm.jet,vmin=0.0,alpha=.2)
        ax.scatter(tvecs[0,::skip], tvecs[1,::skip],c=color[::skip], s=1,cmap=cm.jet,vmin=0.0,alpha=1.0)        
        ax.set_xlabel(clist[i],fontsize=16,fontweight='bold')
        ax.axis('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)                

        seg_coll = mc.LineCollection(segs,color='r',alpha=.9)
        ax.add_collection(seg_coll)

    plt.tight_layout()
    plt.show()


def mk_submatrix(V,A):
    #A should be a csr_matrix
    B = A[V,:]
    B = B.T
    B = B[V,:]
    #get the orignal vertex numbers
    B = B.tocoo()
    e0 = V[B.row]
    e1 = V[B.col]
    #make the submatrix with the original vertex numbers
    #transpose to restore original directions for directed case
    S = sp.coo_matrix( (B.data, (e0,e1)) , shape=A.shape).T
    S = S.tocsr()
    return S
    

def branch_csr(paths,N):
    #each path is SortedDict distance
    #along path as key and cell as value

    e0_tot = []
    e1_tot = []
    sdata_tot = []

    #or make directed first???

    #instead of using lists we could just add the individual sparse matrices
    #for each path

    for path in paths:
        pnodes = np.array( list(path.values()) ,dtype=int)
        
        e0 = pnodes[:-1]
        e1 = pnodes[1:]

        spath = np.array( list(path.keys()) )
        sdata = spath[1:] - spath[:-1]

        e0_sym = np.concatenate( (e0,e1),dtype=int)
        e1_sym = np.concatenate( (e1,e0),dtype=int)
        sdata_sym = np.concatenate( (sdata,sdata))

        e0_tot.append(e0_sym)
        e1_tot.append(e1_sym)
        sdata_tot.append(sdata_sym)

    e0_tot = np.concatenate(e0_tot)
    e1_tot = np.concatenate(e1_tot)
    sdata_tot = np.concatenate(sdata_tot)

    coo_br = sp.coo_matrix( (sdata_tot,(e0_tot,e1_tot)), shape=(N,N) )
    csr_br = coo_br.tocsr()
                              
    return csr_br



def all_edges(N1):
    #all edges

    vert = np.arange(N1)
    aedges = list(cprod(vert,vert))
    aedges = np.array(aedges,dtype=int)
    sel = aedges[:,0] < aedges[:,1]

    print(np.sum(sel))

    aedges = aedges[sel]

    print(aedges.shape)
    return aedges



def set_pos1(cg):
    #assign x coords of branching points
    #breadth first? walk thru forward graph
    start = cg.start
    radj = cg.for_adj
    lifo = Q.LifoQueue() 

    lifo.put(start)
    vpos = SortedDict()
    vpos[start] = np.array([0.0,0.0])

    #does dy here make sense
    dy = 1.0
    fac = .75

    vsegs = []
    edges = []

    while not lifo.empty():
        v0 = lifo.get()
        pos0 = np.array(vpos[v0]) #dont need to np.array here?
        nvert = len(radj[v0])

        if nvert == 0:
            ylevels = []
        elif nvert == 1:
            ylevels = [0.0]
        else:
            ylevels = 2*dy*np.arange(nvert)/(nvert-1) -dy

        #shift vertex postions to get gap
        for i,v1 in enumerate(radj[v0]):
            vp = (v0,v1)

            edges.append(vp)

            seg = cg.segments[vp]
            dxl = len(seg)
            vpos[v1] = pos0+(dxl,ylevels[i])
            lifo.put(v1)

            vsegs.append([pos0,vpos[v1]])

        #shrink the range for vertices deeper in graph
        #result is that we get ordering of far_ends by y
        dy = fac*dy
    return vpos,vsegs,edges



def nn_from_tree_2(csr_adj,nn):
    tot = csr_adj

    for i in range(2,nn+1):
        tot = csr_adj + csr_adj @ tot

    return tot


def branch_adj2(paths,trim=0):
    br_adj = SortedDict()

    for path in paths:
        vbr = path.values()
        n = len(vbr)
        for j in range(n-1):
            j_cell = vbr[j]
            j1_cell = vbr[j+1]
            if j_cell  not in br_adj:
                br_adj[j_cell] = []
            br_adj[j_cell].append(j1_cell)
            if j1_cell  not in br_adj:
                br_adj[j1_cell] = []
            br_adj[j1_cell].append(j_cell)

    #collect the ends

    while trim > 0:

        ends = []
        for v0 in br_adj:
            adj = br_adj[v0]
            if len(adj) == 1:
                ends.append((v0,adj[0]))

        for v0,v1 in ends:
            br_adj.pop(v0)
            br_adj[v1].remove(v0)

        trim -= 1

    return br_adj


class pq_vert:
    def __init__(self,v_,cost_):
        self.v = v_
        self.cost = cost_

    def __lt__(self,other):
        return self.cost < other.cost

    def __eq__(self,other):
        return self.cost == other.cost        


def get_parents(v0set,adj,dist):
    pq = Q.PriorityQueue()    
    nvert = len(adj)

    parent = np.full(nvert,-1,dtype=int)
    cost = np.full(nvert,-1.0,dtype=np.double)

    for v0 in v0set:
        parent[v0] = -1
        pq.put((pq_vert(v0,0.0)))
        cost[v0] = 0.0

    while not pq.empty():
        pqv = pq.get()
        v = pqv.v

        for i,u in enumerate(adj[pqv.v]):
            ucost = dist[v][i] + pqv.cost

            if cost[u] == -1 or ucost < cost[u]:
                cost[u] = ucost
                pq.put(pq_vert(u,ucost))
                parent[u] = v


    return cost,parent



def normalize1(data):
    ntdata = la.norm(data,ord=1,axis=1,keepdims=True)
    data1 = data/ntdata
    return data1


def sym_coo(cc):
    """
    cc should be a coo_matrix
    returns symmetrized version
    """
    elems = set(zip(cc.row,cc.col,cc.data))
    telems = set(zip(cc.col,cc.row,cc.data))
    all = list(elems.union(telems))
    all = list(zip(*all))

    #print("cc shape",cc.shape)

    ccs = sp.coo_matrix((all[2],(all[0],all[1])),shape=cc.shape)
    return ccs

#problem was not tuple sort so may not need this
def keyf(x): return x[0]


class knn_graph:
    """
    This class finds the neighbors of the Y points
    among the X points.
    """

    def worker(iseg,nl,X,Y,k,ires,dres):
        tree = BallTree(X, leaf_size=2)
        iseg1 = iseg+1
        dist, ind = tree.query(Y[iseg*nl:iseg1*nl],k=k)

        ind0 = np.frombuffer(ires, dtype=np.uint32 )
        ind0 = ind0.reshape((Y.shape[0],k))

        dist0 = np.frombuffer(dres, dtype=float)
        dist0 = dist0.reshape((Y.shape[0],k))        

        nlo = iseg*nl
        nhi = iseg1*nl

        ind0[nlo:nhi,:] = ind[:,:]
        dist0[nlo:nhi,:] = dist[:,:]        

    def __init__(self,X,Y,nsegs=8):
        nsegs = int(nsegs)
        self.nsegs = nsegs

        #print("using ",nsegs, "processes")

        self.X = X
        self.Y = Y

        self.ndim = X.shape[1]
        self.npnts = X.shape[0]
        self.nq = Y.shape[0]

        self.nl = self.nq//self.nsegs + 1
        self.imax = self.nsegs

        self.dlist = [None]*(self.imax)
        self.ilist = [None]*(self.imax)           



    def run(self,k):

        nraw = k*self.Y.shape[0]
        
        ires = mp.RawArray('i',nraw)
        dres = mp.RawArray('d',nraw)

        threads = []
        for iseg in range(self.imax):
            t = mp.Process(target=knn_graph.worker, args=(iseg,self.nl,self.X,self.Y,k,ires,dres))
            threads.append(t)
            t.start()
            
        for thread in threads:
            thread.join()

        ind0 = np.frombuffer(ires, dtype=np.uint32 )
        ind0 = ind0.reshape((self.Y.shape[0],k))

        dist0 = np.frombuffer(dres, dtype=float)
        dist0 = dist0.reshape((self.Y.shape[0],k))                

        self.dist = dist0
        self.ind = ind0

    def write(self,dir):
        np.savetxt(dir+"/mgraph.dist",self.dist)
        np.savetxt(dir+"/mgraph0.adj",self.ind,fmt="%d")




class sparse_adj:
    def __init__(self,ind1,dist1):
        """
        ind1,dist1 are numpy arrays
        with no explicity src vertices
        src # is just row index
        """
        #much better way of constructing csr_matrix than explicit looping

        #print(ind1.shape,dist1.shape)



        nrows = ind1.shape[0]
        nrows1 = ind1.shape[0]+1
        idim = ind1.shape[1]
        ind1 = ind1.flatten()
        dist1 = dist1.flatten()
        indptr = np.arange(0,nrows1*idim,idim)

        bb = sp.csr_matrix( (dist1,ind1,indptr),shape=(nrows,nrows))
        bb = sp.coo_matrix(bb)

        bb = sym_coo(bb)
        bb = sp.csr_matrix(bb)
        self.csr_adj = bb


class mst:
    def __init__(self,ind,dist):
        #cut off the src vertices
        #dist1 = dist[:,1:]
        #ind1 = ind[:,1:]

        print("starting mst")

        dist1 = dist.copy()
        ind1 = ind

        dist1 += 1e-8

        sadj = sparse_adj(ind1,dist1)
        bb = sadj.csr_adj

        self.csr_nn = bb


        ncomp, labels = sp.csgraph.connected_components(bb,directed=False)
        print("nn components",ncomp)
        
        #now do the mst
        tmp = sp.csgraph.minimum_spanning_tree(bb)

        print("tmp",type(tmp))

        ncomp, labels = sp.csgraph.connected_components(tmp,directed=False)

        print("tmp mst components",ncomp)                

        tmp = sp.coo_matrix(tmp)

        row = tmp.row

        self.coo_mst = sym_coo(tmp)

        self.csr_mst = self.coo_mst.tocsr()

        ncomp, labels = sp.csgraph.connected_components(self.csr_mst,directed=False)

        print("mst components",ncomp)        
        

        
#find the shortest path to the farthest cell from vsrc
def find_path(vsrc, mst_adj,mst_dist):
    cost,parent = get_parents(vsrc,mst_adj,mst_dist)
    pcell = np.argmax(cost)

    br = SortedDict()
    br[cost[pcell]] = pcell
    
    
    while parent[pcell] != -1:
        pcell = parent[pcell]
        br[cost[pcell]] = pcell

    #br is SortedDict of cells along path
    #index is cost from src
    return br


def pad_adj(a):
    alens = np.array([len(x) for x in a])
    amax = np.amax(alens)


def find_branches(mst_adj,mst_dist,branchings,start):
    """ mst0 is min span tree"""

    paths = []

    if start == -1:
        #find a good start cell
        v0set = [0]
        br0 = find_path(v0set,mst_adj,mst_dist)
        imax = br0.values()[-1]
    else:
        imax = start
        
    v0set = [imax]
    #br is a sortedDict of cells along path
    #index is cost from src    
    br = find_path(v0set,mst_adj,mst_dist)

    paths.append(br)
    #print("br peek",br.peekitem(0),br.peekitem())
    vbr = br.values()
    ns = set()

    while branchings > 0:
        ns.update(vbr)
        br = find_path(ns,mst_adj,mst_dist)
        #print(br.peekitem(0),br.peekitem())
        paths.append(br)
        vbr = br.values()
        branchings -= 1

    return paths


def level_nn(X,lev0,k=10,nsegs=8):

    N = X.shape[0]
    map0 = np.arange(N)
    map0 = map0[lev0]

    X0 = X[lev0]

    knn = knn_graph(X0,X0,nsegs)
    knn.run(k+1)
    #we need to drop the first col, since it is self
    adj = knn.ind[:,1:]
    dist = knn.dist[:,1:]


    #node numbers in adj need to be mapped to originals
    adj2 = map0[adj]
    dk = sp.dok_matrix( (N,N) )

    #try lil
    #dk = sp.lil_matrix( (N,N) )

    for i in range(k):
        dk[map0,adj2[:,i]] = dist[:,i]

    #for dok
    graph_sym(dk)

    return dk,map0,adj2,dist



    
def div_link(df,lev0,lev1,k=1):
    """
    lev0 is mitotic
    lev1 is g1
    cells already transformed
    """

    df0 = df[lev0]
    df1 = df[lev1]

    #mapping from df1.values indices to original indices
    map1 = df1.index
    #mapping from df0.values indices to original indices
    map0 = df0.index

    #find nearest point in lev1 of each
    #point in lev0

    tree = BallTree(df1.values, leaf_size=16)
    dist, adj = tree.query(df0.values,k=k)

    print("div link",np.amin(dist))

    dist += 5.0

    #now construct dok_matrix to describe graph with these edges
    #ok this works
    #here the df1.values
    adj2 = map1[adj]


    N = df.shape[0]

    
    dk = sp.dok_matrix( (N,N) )

    #for matrix need original df0 indices
    for i in range(k):
        dk[map0,adj2[:,i]] = dist[:,i]

    graph_sym(dk)
    return dk


def level_link(df,lev0,lev1,k=1):
    """
    df is data frame with original indexing
    and transformed markers containing
    both levels

    this function finds nearest neighbors of
    lev0 points among lev1 points
    """

    df0 = df[lev0]
    df1 = df[lev1]

    #mapping from df1.values indices to original indices
    map1 = df1.index

    #mapping from df0.values indices to original indices
    map0 = df0.index

    #find nearest point in lev1 of each
    #point in lev0

    tree = BallTree(df1.values, leaf_size=16)
    dist, adj = tree.query(df0.values,k=k)

    dmin = np.amin(dist)

    dist += 1.0

    print("linking","dmin",dmin)

    #now construct dok_matrix to describe graph with these edges
    #ok this works
    #here the df1.values
    adj2 = map1[adj]

    N = df.shape[0]
    
    dk = sp.dok_matrix( (N,N) )

    #for matrix need original df0 indices
    for i in range(k):
        dk[map0,adj2[:,i]] = dist[:,i]

    graph_sym(dk)
    return dk

def level_link2(X,lev0,lev1,k=1,nsegs=8):
    """
    df is data frame with original indexing
    and transformed markers containing
    both levels

    this function finds nearest neighbors of
    lev0 points among lev1 points

    this version does not symmetrize
    """
    N = X.shape[0]
    X0 = X[lev0]
    X1 = X[lev1]

    map0 = np.arange(N)
    map0 = map0[lev0]

    map1 = np.arange(N)
    map1 = map1[lev1]    

    #find nearest point in lev1 of each
    #point in lev0

    knn = knn_graph(X1,X0,nsegs=nsegs)
    knn.run(k)

    adj = knn.ind
    dist = knn.dist   
    
    #now construct dok_matrix to describe graph with these edges
    #ok this works
    #here the df1.values
    adj2 = map1[adj]
    dk = sp.dok_matrix( (N,N) )

    #for matrix need original df0 indices
    for i in range(k):
        dk[map0,adj2[:,i]] = dist[:,i]

    #remove this so graph we have directed edges
    #graph_sym(dk)
    return dk



def dk_append(dk0,dk1):
    #append dk1 to dk0
    #expect non overlapping

    irow,icol = list(zip(*dk1.keys()))
    dk0[irow,icol] = dk1[irow,icol]
    return dk0

def graph_sym(dk0):
    #make edges undirected
    irow,icol = list(zip(*dk0.keys()))
    dk0[icol,irow] = dk0[irow,icol]       

def graph_sym_lil(dk0):
    #make edges undirected
    irow,icol = dk0.nonzero()
    dk0[icol,irow] = dk0[irow,icol]       


def sym_matrix(g):
    rows,cols = g.nonzero()
    g0 = g.tolil()
    g0[cols,rows] = g0[rows,cols]
    g = g0.tocsr()

    return g

    




class cytoskel:
    def __init__(self,project_dir):

        ch_last = project_dir[-1]
        if ch_last != "/":
            project_dir = project_dir + "/"

        self.project_dir = project_dir

        if not os.path.exists(project_dir):
            os.mkdir(project_dir)                


    def create(self,df,
               traj_markers = [],
               avg_markers = [],
               mon_markers = [],
               l1_normalize = True,
               level_marker = None,
               pca_dim = 0
    ):
        
    
        project_dir = self.project_dir
        dir_name = self.project_dir

        if not os.path.exists(project_dir):
            os.mkdir(project_dir)

        self.mk_log()
        
        self.level_marker = level_marker
        self.l1_normalize = l1_normalize        

        self.params = {
            "l1_normalize":l1_normalize,
            "navg":-1,
            "ntree":-1,
            "level_marker":level_marker,
            "nn_neighbors":-1,
            "branchings":-1,
            }

        self.write_params()
        
        f0 = open(project_dir+"params.txt","w")
        f0.write(str(self.params))
        f0.close()

        self.df = df



        self.markers = list(self.df.columns)
        self.traj_markers = traj_markers
        self.avg_markers = avg_markers
        self.mon_markers = mon_markers

        self.pca_dim = pca_dim


        write_setup(self.project_dir+"run.setup",
                        self.markers,
                        self.traj_markers,
                        self.avg_markers,
                        self.mon_markers
                        )

        #for nn and mst construction
        self.df_traj = self.df.loc[:,self.traj_markers].copy()


        #write original data
        self.df.to_csv(dir_name+"mdata.csv",index=False)        


        if self.l1_normalize:
            x = self.df_traj.values
            x = normalize1(x)
            self.df_traj.loc[:,:] = x

        if pca_dim == 0:
            self.X = self.df_traj.values.copy()
        elif pca_dim > 0:
            x = self.df_traj.values
            pca = pca_coords(x,self.pca_dim)
            ux,urad = pca.get_coords(x)

            self.X = ux
        else:
            print("pca_dim error")
            exit()

    #create an alias for create
    setup_graph_data = create


    def open(self):
        t0 = time.time()
        project_dir = self.project_dir
        
        if not os.path.exists(project_dir):
            print("no project directory:",project_dir)
            return False

        vdir = project_dir + "pca_views/"
        if not os.path.exists(vdir):
            os.mkdir(vdir)

        self.pca_views = vdir

        is_params = os.path.exists(project_dir+"params.txt")
        is_setup = os.path.exists(project_dir+"run.setup")
        is_thead2 = os.path.exists(project_dir+"thead2.txt")
        is_ahead = os.path.exists(project_dir+"ahead.txt")        

        is_mst_npz = os.path.exists(project_dir+"mst.npz")

        is_nn = os.path.exists(project_dir+"nn.npz")

        if is_nn:
            self.csr_nn = sp.load_npz(self.project_dir+"nn.npz")

        if os.path.exists(project_dir+"csr_br.npz"):
            self.csr_br = sp.load_npz(self.project_dir+"csr_br.npz")

        if os.path.exists(project_dir+"csr_avg.npz"):
            self.csr_avg = sp.load_npz(self.project_dir+"csr_avg.npz")                    
            

        if is_setup:
            gi = get_info(project_dir+"run.setup")
            gi.get_marker_usage()

            self.avg_markers = gi.get_avg_cols()
            self.traj_markers = gi.get_traj_cols()
            self.mon_markers = gi.get_all_mon_cols()

        else:
            print("no run.setup file")
            return False
        
        #so far params not used for anything?
        if is_params:
            self.read_params()

        self.df = pd.read_csv(project_dir+"mdata.csv")
        self.markers = list(self.df.columns)


        if os.path.exists(project_dir+"avg_mdata.csv"):
            self.df_avg = pd.read_csv(project_dir+"avg_mdata.csv")
        else:
            print("need to run averaging")
            self.df_avg = None

        if os.path.exists(project_dir+"qmst.npz"):
            self.csr_qmst = sp.load_npz(self.project_dir+"qmst.npz")                                

        markers = list(self.df.columns)

        if is_mst_npz:
            self.csr_mst = sp.load_npz(project_dir+"mst.npz")

            ltmp = self.csr_mst.tolil()
            self.mst_adj = ltmp.rows
            self.mst_dist = ltmp.data
        elif os.path.exists(project_dir+"mst.adj"):
            self.mst_adj = read_sdict(project_dir+"mst.adj")
            self.csr_mst = read_sdict2(project_dir+"mst.adj")
            if os.path.isfile(project_dir+"nn.npz"):
                csr_nn = sp.load_npz(project_dir+"nn.npz")
                self.csr_mst = csr_nn.multiply(self.csr_mst)
            print("mst read")
        else:
            self.mst_adj = None
            

        if os.path.isfile(project_dir+"nn.npz"):
            self.csr_nn = sp.load_npz(project_dir+"nn.npz")                


        cg = self.load_cg()
        if cg:
            self.cg = cg
            self.br_adj = cg.br_adj
        elif os.path.exists(project_dir+"br.adj"):
            self.br_adj = read_sdict(project_dir+"br.adj")
            cg = cgraph()
            cg.get_reduced_graph(self.br_adj)
            self.cg = cg
        else:
            self.br_adj = None
            self.cg = None


        if os.path.exists(project_dir+"df_mds2.csv"):
            self.df_mds2 = pd.read_csv(project_dir+"df_mds2.csv",index_col=0)

        t1 = time.time()

        print("open time",t1 - t0)
            
        return True


    def mk_log(self):
        #run in create
        argv = sys.argv

        dir = self.project_dir

        f = open(dir+"log.txt","w")
        argv = str(argv)
        #record the script being run
        f.write("argv: "+argv+"\n")

        cwd = os.getcwd()
        #where the script was run
        f.write("cwd "+cwd+"\n")

        #where the output data directory was initially
        f.write("data dir: "+dir+"\n")

        f.close()

    def add_log(self,s):
        f = open(self.project_dir+"log.txt","a")        
        f.write(s + "\n")
        f.close()


    def set_num_knn_processes(self,n_processes = 8):
        sel.n_processes = n_processes


    def prune(self,plist):
        cg = cgraph()
        cg.get_reduced_graph(self.br_adj)
        cg.get_end_segment_cells(self.mst_adj,True)

        cg.cells_to_remove = []

        for vend in plist:
            cg.prune_end_segment(vend)

        self.cg = cg
        """
        cg.prune_end_segment(370743)        
        cg.prune_end_segment(255702)
        cg.prune_end_segment(214925)
        cg.prune_end_segment(286465)
        cg.prune_end_segment(135623)


        """

    def split(self,vsplit,plist=None):
        cg = cgraph()
        cg.get_reduced_graph(self.br_adj)
        self.cg = cg
        #cg.get_end_segment_cells(self.mst_adj,True)

        #find the segment that contains vsplit

        for vp in cg.segments:
            if vp[0] > vp[1]: continue
            seg = cg.segments[vp]
            if vsplit in seg:
                idx = seg.index(vsplit)
                print("split",vsplit,vp,idx)
                if vsplit in vp:
                    print("vsplit is seg end, no split")
                else:
                    v0 = vp[0]
                    v1 = vp[1]

            
        if v0 != None:
            print("splitting")
            self.split_segment(v0,v1,vsplit)
            cg.bpnts[2].append(vsplit)


        cg.get_end_segment_cells(self.mst_adj,True)            

        if plist:
            cg.cells_to_remove = []

            for vend in plist:
                cg.prune_end_segment(vend)
                #try redoing after each prune
                cg.get_end_segment_cells(self.mst_adj,True)                        

    def split_segment(self,v0,v1,vsplit):
        cg = self.cg
        seg = cg.segments[(v0,v1)]
        add_segs = {}
        remove_segs = {}

        cg.reduced_adj[v0].remove(v1)
        cg.reduced_adj[v1].remove(v0)
        #need only one order
        remove_segs[(v0,v1)] = seg

        ve = vsplit
        isplit = seg.index(vsplit)

        
        nseg = len(seg)

        vs = v0
        vlist = [vs]

        nusegs = {}

        seg0 = seg[0:isplit+1]
        add_segs[(v0,ve)] = seg0
        add_segs[(ve,v0)] = seg0[::-1]

        seg1 = seg[isplit:]
        add_segs[(ve,v1)] = seg1
        add_segs[(v1,ve)] = seg1[::-1]

        cg.reduced_adj[ve] = [v0,v1]
        cg.reduced_adj[v0].append(ve)
        cg.reduced_adj[v1].append(ve)

        for key in add_segs:
            cg.segments[key] = add_segs[key]

        for v0,v1 in remove_segs:
            del cg.segments[(v0,v1)]
            del cg.segments[(v1,v0)]
        
        

    def write_pruned(self,gated_dir = None):
        N = self.df.shape[0]
        dfkeep = pd.DataFrame( np.full(N,1.0), columns=['keep'])
        cg = self.cg

        dfkeep.iloc[cg.cells_to_remove,:] = 0.0
        keep = dfkeep.iloc[:,0] == 1.0
        map0 = dfkeep.loc[keep,:].index.values
        idx0 = np.arange(map0.shape[0])
        
        a = list(zip(map0,idx0))
        rmap0 = dict(a)

        br_adj = cg.reduced_to_br_adj(rmap0)

        self.br_adj = br_adj

        for vp in cg.segments:
            print("seg vp",vp)




        dir1 = self.project_dir

        if gated_dir == None:
            dir2 = dir1 + "/br_gated/"
        else:
            dir2 = gated_dir
            if dir2[-1] != "/":
                dir2 = dir2 + "/"
                dir2 = dir1 + dir2

        if not os.path.exists(dir2):
            os.mkdir(dir2)
            print("created directory",dir2)


        shutil.copyfile(dir1+"/run.setup",dir2+"run.setup")


        df2 = self.df.loc[keep,:]
        df2.to_csv(dir2+"mdata.csv",index=False)

        df2_avg = self.df_avg.loc[keep,:]
        df2_avg.to_csv(dir2+"avg_mdata.csv",index=False)

        #for comparison
        self.df_avg = df2_avg

        write_sub_adj(br_adj,dir2+"br.adj")        


    def subway(self,df_pcells):
        sub = subway(self,df_pcells,[])
        sub.subway()
        fig = plt.figure(figsize=(14,6))
        sub.draw_subway(fig)
        plt.show()


    def write_params(self):
        project_dir = self.project_dir
        f0 = open(project_dir+"params.txt","w")
        f0.write(str(self.params))
        f0.close()

    def read_params(self):
        f = open(self.project_dir+"params.txt","r")
        sf = f.read()
        f.close()
        self.params = ast.literal_eval(sf)

    def read_list(self,fname):
        f = open(self.project_dir+fname,"r")
        sf = f.read()
        llist = ast.literal_eval(sf)
        f.close()

        return llist

    def write_x(self,x,fname):
        f = open(self.project_dir+fname,"w")
        f.write(str(x))
        f.close()

    def nn_graph0(self,k=30,n_process=8):
        k = int(k) # in case of R
        self.params['nn_neighbors'] = k
        self.write_params()

        nsegs = n_process

        
        #knn = knn_graph(self.df_traj.values,self.df_traj.values,nsegs=nsegs)
        knn = knn_graph(self.X,self.X,nsegs=nsegs)
        knn.run(k)
        self.adj = knn.ind[:,1:]
        self.dist = knn.dist[:,1:]

        self.fadj = self.adj
        self.fdist = self.dist

        np.savez(self.project_dir+"nn0.npz",adj = self.adj, dist = self.dist)




    def mst_graph0(self):
        mst0 = mst(self.adj, self.dist)
        self.csr_mst = mst0.csr_mst
        sp.save_npz(self.project_dir+"mst.npz",self.csr_mst)

        ltmp = self.csr_mst.tolil()

        self.mst_adj = ltmp.rows
        self.mst_dist = ltmp.data

        #write the graph data here
        write_full_adj(self.mst_adj,self.project_dir+"mst.adj")


    def do_graphs(self,nn_neighbors=30,n_process=8):

        self.n_process = n_process

        print("starting graphs with",n_process,"processes")
        t0 = time.time()

        if not self.level_marker:
            self.params['nn_neighbors'] = nn_neighbors
            self.write_params()         
            self.do_nn_graph(nn_neighbors)
            self.do_mst_graph()
        else:
            self.link(self.level_marker)

        t1= time.time()

        self.write_params()

        print("graph time",t1 - t0)

        ltmp = self.csr_mst.tolil()

        self.mst_adj = ltmp.rows
        self.mst_dist = ltmp.data

        #write the graph data here
        write_full_adj(self.mst_adj,self.project_dir+"mst.adj")
        sp.save_npz(self.project_dir+"mst.npz",self.csr_mst)


    def do_nn_graph(self,k):
        k = int(k)
        nsegs = self.n_process
        
        #knn = knn_graph(self.df_traj.values,self.df_traj.values,nsegs=nsegs)
        knn = knn_graph(self.X,self.X,nsegs=nsegs)
        knn.run(k)
        self.adj = knn.ind[:,1:]
        self.dist = knn.dist[:,1:]

        np.savez(self.project_dir+"nn0.npz",adj = self.adj, dist = self.dist)        


    def do_density(self):
        self.density = 1.0/(self.dist[:,0] + 1e-2)
        df_density = pd.DataFrame(self.density,columns=['density'])
        #df_density.to_csv(self.project_dir+'density.csv')
        return df_density


    def do_mst_graph(self):
        mst0 = mst(self.adj, self.dist)
        self.csr_mst = mst0.csr_mst
        sp.save_npz(self.project_dir+"nn.npz",mst0.csr_nn)
        self.csr_nn = mst0.csr_nn


        


    def do_branches(self,start=-1,branchings=4):

        start = int(start)
        branchings = int(branchings)

        self.params['branchings'] = branchings
        self.write_params()


        paths = find_branches(self.mst_adj,self.mst_dist,branchings,start)

        self.paths = paths

        self.csr_br = branch_csr(paths,self.df.shape[0])

        self.br_adj = branch_adj2(paths)

        cg = cgraph()
        cg.get_reduced_graph(self.br_adj)
        self.cg = cg
        self.dump_cg()

        write_sub_adj(self.br_adj,self.project_dir+"br.adj")
        sp.save_npz(self.project_dir+"csr_br.npz",self.csr_br)        


    def get_average_fix(self,avg_markers,fixed=[],navg=5,ntree=4,n_radius = -1, sfile=None):
        #note: radius is ntree+1

        self.write_x(avg_markers,"ahead.txt")
        self.avg_markers = avg_markers

        write_setup(self.project_dir+"run.setup",
                        self.markers,
                        self.traj_markers,
                        self.avg_markers,
                        self.mon_markers
                        )
        
        navg = int(navg)
        ntree = int(ntree)

        if hasattr(self,'params'):
            self.params['navg'] = navg
            self.params['ntree'] = ntree
            self.write_params()        

        t0 = time.time()            

        csr_mst = self.csr_mst

        if n_radius == -1: n_radius = ntree + 1
        tot = nn_from_tree_2(csr_mst,n_radius)
        tot = tot.tolil()
        tot.setdiag(0)

        """
        #ok this seems not to be necessary here
        print("foo")
        tmp5 = tot.tocoo()
        tmp5.eliminate_zeros()
        dk = tmp5.todok()
        """


        dk = tot.todok()
        
        irow,icol = list(zip(*dk.keys()))
        dk[irow,icol] = 1.0
        dk = dk.tocsr()
        rsums = dk.sum(axis=1)
        rsums = np.array(rsums)

        rsums = rsums**(-1)
        N = tot.shape[0]

        tot2 = sp.lil_matrix((N,N))
        tot2.setdiag(rsums)
        tot2 = tot2.tocsr()
        tot2 = tot2 @ dk

        #add averaging matrix to the csk object
        self.csr_avg = tot2

        sp.save_npz(self.project_dir+"csr_avg.npz",self.csr_avg)

        #self.mst_nn_adj = tot.rows #this seems not to be used for anything

        df2 = self.df.copy()
        df_avg = df2[avg_markers]

        t1= time.time()

        print("mst_nn done",t1-t0)


        val0 = df_avg.values
        val2 = df_avg.copy().values

        navg0 = navg

        while navg0 > 0:
            val1 = tot2 @ val0
            val0[:,:] = val1[:,:]
            navg0 -= 1
        
        t2 = time.time()

        print("avg time",t2-t1)

        df_avg.values[:,:] = val0[:,:]
        df2[avg_markers] = df_avg[avg_markers]

        self.df_avg = df2

        if sfile == None:
            self.df_avg.to_csv(self.project_dir + "avg_mdata.csv",index=False)
        else:
            self.df_avg.to_csv(self.project_dir + sfile,index=False)

        t1 = time.time()     
        self.add_log("average "+ str(t1-t0))

    do_mst_averaging = get_average_fix #set an alias with a better name

    def link(self,gcol,k_intra=10,k_inter=1,
                 glist = [],
                 n_process=8):

        self.add_log("starting link")

        self.n_process = n_process
        t0 = time.time()

        if glist == []:
            glist = set(self.df[gcol])
            glist = list(glist)
            glist.sort()


        #create booleans for the groups in order
        lev = []
        for ilev,grp in enumerate(glist):
            alev = self.df[gcol] == grp
            lev.append(alev)

        #data frame for only the trajectory markers

        intra_dk = []
        dk_links = []
        intra_mst = []

        N = self.X.shape[0]
        fadj = np.full( (N,k_intra),-1,dtype=np.int)
        fdist = np.full( (N,k_intra),0.0,dtype=float)

        self.fadj = fadj
        self.fdist = fdist

        np.savez(self.project_dir+"nn0.npz",adj = self.fadj, dist = self.fdist)        

        #form the within group links
        for ilev in range(len(lev)):
            t00 = time.time()
            dk,lev_map0,lev_adj,lev_dist = level_nn(self.X,lev[ilev],k=k_intra,nsegs=self.n_process)
            t11 = time.time()

            #record the nearest neighbors, intra
            fadj[lev_map0] = lev_adj
            fdist[lev_map0] = lev_dist

            self.add_log("level "+str(ilev)+" time "+str(t11-t00))

            graph_sym(dk)

            intra_dk.append(dk)

            dk_csr = dk.tocsr()
            ncomp, labels = sp.csgraph.connected_components(dk_csr,directed=True)

            tmp = sp.csgraph.minimum_spanning_tree(dk_csr)
            tmp = sym_matrix(tmp)
            tmp = tmp.todok() #??
            intra_mst.append(tmp)
            
        #make the inter group links
        for ilev in range(1,len(lev)):
            #these will be directed edges
            dk_link = level_link2(self.X,lev[ilev-1],lev[ilev],k=k_inter,nsegs=self.n_process)
            self.add_log("linking "+str(ilev-1) + " " + str(ilev))
            dk_links.append(dk_link)

        #print("fdist min",np.amin(fdist))

        #now make the DSTs

        dsts = []

        for i in range(1,len(lev)):
            AB = dk_links[i-1]
            B = intra_mst[i]
            dmst = self.dst(AB,B)
            dsts.append(dmst)

        #append all
        dk0 = intra_mst[0]

        for dk in dsts:
            dk_append(dk0,dk)

        self.csr_mst = dk0.tocsr()

        t1 = time.time()

        self.add_log("link " + str(t1-t0))
        sp.save_npz(self.project_dir+"mst.npz",self.csr_mst)        

        """
        #construct nn graph matrix
        #intra_dk has all the internal group nn neighbors
        dk_tot = intra_dk[0]

        for i in range(1,len(intra_dk)):
            dk_append(dk_tot,intra_dk[i])

        for dklink in dk_links:
            dk_append(dk_tot,dklink)

        #csr_tot = dk_tot.tocsr()

        #sp.save_npz(self.project_dir+"nn.npz",csr_tot)
        """

    def dst(self,AB,B):
        #make mst for B
        bcsr = B.tocsr()
        bmst = sp.csgraph.minimum_spanning_tree(bcsr)
        bmst= bmst.tocoo()
        bmst = sym_coo(bmst)

        #edges most easily gotten from coo_matrix
        #row is array of edge starts (tails), col is list of edge ends (heads)
        AB = AB.tocoo()  
        bmst = bmst.tocoo()

        N = B.shape[0]    

        #here AB row and col appear twice to symmetrize
        trow = list(bmst.row) + list(AB.row) + list(AB.col)    
        tcol = list(bmst.col) + list(AB.col) + list(AB.row)
        tdata = list(bmst.data) + list(AB.data) + list(AB.data)

        #we make head and tail lists for the N to A edges
        ecol = list(AB.row)
        erow = [N]*len(ecol)
        eps = 1e-6
        edata = [eps]*len(ecol)

        #we add these edges to the others
        trow = trow + erow + ecol
        tcol = tcol + ecol + erow
        tdata = tdata + edata + edata

        tmat = sp.coo_matrix( (tdata,(trow,tcol)) )
        tmat = tmat.tocsr()

        t_mst = sp.csgraph.minimum_spanning_tree(tmat)
        t_mst = sym_matrix(t_mst)        

        dmst = t_mst.todok()

        dmst[N,:] = 0.0
        dmst[:,N] = 0.0

        return dmst


    def path_traj_dist(self,v0,v1):
        gsp = gspaths2(self.br_adj)
        path = gsp.get_path2(v0,v1)

        tcols = self.traj_markers
        df_path = self.df_avg.loc[path,:].copy()
        npath = df_path.shape[0]

        dist0 = np.zeros(npath)
        adj0 = np.arange(npath)

        df_path.loc[:,'traj_dist'] = dist0
        df_path.loc[:,'ptime'] = adj0

        df_tdata = self.df_avg.loc[path,tcols]
        tdata = df_tdata.values
        odata = self.df.loc[:,tcols].values

        knn = knn_graph(tdata,odata)
        knn.run(1)
        dist,adj = knn.dist,knn.ind

        idx0 = np.arange(len(path))
        a = list(zip(path,idx0))        
        rmap0 = dict(a)


        self.df.loc[:,'traj_dist'] = dist
        self.df.loc[:,'ptime'] = adj[:,0]
        #print("added traj_dist and ptime")

        return self.df,df_path


    def dump(self,obj,fname):
        f = open(self.project_dir+fname,"wb")
        pickle.dump(obj,f)
        f.close()


    def load(self,fname0):
        #check if file exists first

        fname = self.project_dir + fname0



        if not os.path.exists(fname):
            return None

        print("loading: ",fname)

        #traceback.print_stack()

        f = open(fname,"rb")
        obj = pickle.load(f)
        f.close()

        return obj
        

    def load_cg(self):
        #check if file exists first

        fname0 = "cgraph.pkl"

        return self.load(fname0)


    def dump_cg(self):
        fname0 = "cgraph.pkl"
        self.dump(self.cg,fname0)


    def dump_cg0(self):
        #dump a backup
        fname0 = "cgraph0.pkl"
        self.dump(self.cg,fname0)
        

    def check_same(self,pdir0):
        csk0 = cytoskel(pdir0)
        csk0.open()

        br0_adj = csk0.br_adj
        br_adj = self.br_adj

        same = True
        for v in br0_adj.keys():
            adjv0 = br0_adj[v]
            adjv = br_adj[v]
            same = adjv0 == adjv
            if not same: break

        Xself = self.df_avg.loc[:,self.traj_markers].values
        X0 = csk0.df_avg.loc[:,self.traj_markers].values        

        avg_same = np.allclose(Xself,X0,atol=1e-4)
        return same,avg_same


    def tumap(
            self,
            tscale = 4.0, #for mon mds?
            min_dist = .05,
            spread = 1.0,
            use_avg = False,
            mds_init = False
            ):


        tmap0 = tmap(
            self,
            tscale=tscale,
            min_dist = min_dist,
            spread = spread,
            mds_init = mds_init
            )

        print("tmap")



    def get_tumap(self):
        df = pd.read_csv(self.project_dir+"df_tumap.csv")

        is_data = df.loc[:,'cell_type'] == 1
        is_traj = df.loc[:,'cell_type'] == 2
        ep =  df.loc[:,'parent'] != -1

        is_traj = np.logical_and(is_traj,ep)

        e1 = df.loc[is_traj,'parent']
        e0 = list(e1.index)
        edges = np.array([e0,e1]).T

        ux = df.loc[:,["tiUMAP1","tiUMAP2"]].values

        segs = ux[edges].copy()
        Rsegs = ux[edges].reshape( (-1,4) )
        self.df_rsegs = pd.DataFrame(Rsegs,columns=["x","y","xend","yend"])
        self.df_tumap = df
        self.segs = segs
        
        

class tree_dist:
    def __init__(self,csk,pstart=None):
        #assume already opened
        #csk.open()
        self.csk = csk
        self.pstart = pstart

    def splice(self):
        #get edge lengths in br_adj for later pseudo time
        csk = self.csk
        self.df = csk.df
        self.df_avg = csk.df_avg
        tcols = self.csk.traj_markers
        tdata0 = self.df_avg.loc[:,tcols].values
        pstart = self.pstart

        edges,ecost = get_edge_costs(self.csk.br_adj,tdata0)

        
        #path pseudo cells
        pcells = list(self.csk.br_adj.keys())
        
        df = self.df.copy()
        df_path = self.df_avg.loc[pcells,:].copy()

        #data cells
        df.loc[:,'cell_type'] = 1

        #pseudo cells
        df_path.loc[:,'cell_type'] = 2

        ndata = df.shape[0]
        npcells = df_path.shape[0]


        #let there be M pcells
        #map0 is array of original pcell indices
        map0,rmap0 = mk_maps(self.df,pcells)

        #rmap0 will now be mapping from pcell
        # positions in df_avg to their new
        #appended positions in df_tot
        rmap0 += ndata
        self.rmap0 = rmap0
        self.N0 = ndata
        self.N1 = len(pcells)

        self.tree_edges = rmap0[edges]

        #get nearest pcell for each data cell
        df_tdata = df_path.loc[:,tcols]

        tdata = df_tdata.values
        odata = df.loc[:,tcols].values

        knn = knn_graph(tdata,odata)
        knn.run(1)
        dist, adj = knn.dist,knn.ind

        #adj2 are the original indices
        #of the parents of the data cells
        adj2 = map0[adj]



        gsp = gspaths2(self.csk.br_adj,ecost)

        if pstart == None:
            pstart = self.csk.cg.start
            gsp.get_parents0([pstart])
        else:
            gsp.get_parents0([pstart])
            
        ####

        #keys = list(gsp.cost.keys())
        sdist = list(gsp.cost.values())
        sdist = np.array(sdist)

        df_path.loc[:,'ptime'] = sdist        
        tr_time = sp.dok_matrix( (ndata,1) )

        tr_time[pcells,[0]*npcells] = sdist
        df.loc[:, 'ptime'] = tr_time[adj2[:,0]].todense()        


        ####        

        #gsp.parent0 is dictionary pcell to parent pcell
        #start has parent -1

        #we can do this because parent0 is a SortedDict
        #with same keys as sorted dict br_adj
        pcell_parents = list(gsp.parent0.values())
        df_path.loc[:,'parent'] = pcell_parents
        
        df.loc[:,'parent'] = adj2[:,0]
        df.loc[:,'traj_dist'] = dist

        df_path.loc[self.csk.cg.start,'parent'] = self.csk.cg.start
        df_path.loc[:,'traj_dist'] = 0.0

        #new_start = rmap0[self.start]
        new_start = rmap0[self.csk.cg.start]


        """
        #no branch_id
        #do branch id
        z0 = np.zeros(npcells,dtype=int)        
        br_map = sp.dok_matrix((ndata,1),dtype=int)
        br_map[pcells,z0] = list(br_map.values())

        df.loc[:, 'branch_id'] = br_map[adj2[:,0]].todense()
        df_path.loc[:,'branch_id'] = list(self.br_map.values())

        print(df_path.loc[:,'branch_id'])
        """

        #make total

        df_tot = pd.concat([df,df_path],axis=0)

        print("df",df.shape)        
        print("tot",df_tot.shape)

        print("cols",df_tot.columns)

        print("df_path",df_path.shape)

        #reset the index of df_tot
        ntot = df_tot.shape[0]
        nidx = range(ntot)
        df_tot.index = nidx

        print(df_tot.loc[new_start,'parent'])

        #df_tot.loc[:,'parent'] = rmap0[df_tot['parent'].values].todense()
        df_tot.loc[:,'parent'] = rmap0[df_tot['parent'].values]

        df_tot.loc[new_start,'parent'] = -1

        #is this conversion to R convention ?
        #df_tot.loc[:,'parent'] += 1

        self.df_tot = df_tot
        return df_tot,self.tree_edges



    def splice2(self):
        #get edge lengths in br_adj for later pseudo time
        csk = self.csk
        self.df = csk.df
        self.df_avg = csk.df_avg
        tcols = self.csk.traj_markers
        tdata0 = self.df_avg.loc[:,tcols].values

        edges,ecost = get_edge_costs(self.csk.br_adj,tdata0)

        
        #path pseudo cells
        pcells = list(self.csk.br_adj.keys())
        
        df = self.df.copy()
        df_path = self.df_avg.loc[pcells,:].copy()

        #data cells
        df.loc[:,'cell_type'] = 1

        #pseudo cells
        df_path.loc[:,'cell_type'] = 2

        ndata = df.shape[0]
        npcells = df_path.shape[0]


        #let there be M pcells
        #map0 is array of original pcell indices
        map0,rmap0 = mk_maps(self.df,pcells)

        #rmap0 will now be mapping from pcell
        # positions in df_avg to their new
        #appended positions in df_tot
        rmap0 += ndata
        self.rmap0 = rmap0
        self.N0 = ndata
        self.N1 = len(pcells)

        self.tree_edges = rmap0[edges]

        #get nearest pcell for each data cell
        df_tdata = df_path.loc[:,tcols]

        tdata = df_tdata.values
        odata = df.loc[:,tcols].values

        knn = knn_graph(tdata,odata)
        knn.run(1)
        dist, adj = knn.dist,knn.ind

        #adj2 are the original indices
        #of the parents of the data cells
        adj2 = map0[adj]



        gsp = gspaths2(self.csk.br_adj,ecost)
        gsp.get_parents0([self.csk.cg.start])

        ####

        #keys = list(gsp.cost.keys())
        sdist = list(gsp.cost.values())
        sdist = np.array(sdist)

        df_path.loc[:,'ptime'] = sdist        
        tr_time = sp.dok_matrix( (ndata,1) )

        tr_time[pcells,[0]*npcells] = sdist
        df.loc[:, 'ptime'] = tr_time[adj2[:,0]].todense()        


        ####        

        #gsp.parent0 is dictionary pcell to parent pcell
        #start has parent -1

        #we can do this because parent0 is a SortedDict
        #with same keys as sorted dict br_adj
        pcell_parents = list(gsp.parent0.values())
        df_path.loc[:,'parent'] = pcell_parents
        
        df.loc[:,'parent'] = adj2[:,0]
        df.loc[:,'traj_dist'] = dist

        df_path.loc[self.csk.cg.start,'parent'] = self.csk.cg.start
        df_path.loc[:,'traj_dist'] = 0.0

        #new_start = rmap0[self.start]
        new_start = rmap0[self.csk.cg.start]


        """
        #no branch_id
        #do branch id
        z0 = np.zeros(npcells,dtype=int)        
        br_map = sp.dok_matrix((ndata,1),dtype=int)
        br_map[pcells,z0] = list(br_map.values())

        df.loc[:, 'branch_id'] = br_map[adj2[:,0]].todense()
        df_path.loc[:,'branch_id'] = list(self.br_map.values())

        print(df_path.loc[:,'branch_id'])
        """

        #make total

        df_tot = pd.concat([df,df_path],axis=0)

        print("df",df.shape)        
        print("tot",df_tot.shape)

        print("cols",df_tot.columns)

        print("df_path",df_path.shape)

        #reset the index of df_tot
        ntot = df_tot.shape[0]
        nidx = range(ntot)
        df_tot.index = nidx

        print(df_tot.loc[new_start,'parent'])

        #df_tot.loc[:,'parent'] = rmap0[df_tot['parent'].values].todense()
        df_tot.loc[:,'parent'] = rmap0[df_tot['parent'].values]

        df_tot.loc[new_start,'parent'] = -1

        #is this conversion to R convention ?
        #df_tot.loc[:,'parent'] += 1

        self.df_tot = df_tot
        return df_tot,self.tree_edges
    


    def check(self):
        #what does this do???

        N = self.df.shape[0]

        dft = self.df_tot.loc[self.df_tot['cell_type'] == 2,:]

        idx0 = dft.index.values
        idx1 = dft['parent'].values
        
        a0 = np.where(idx1 == 0)[0]

        idx0 = np.delete(idx0,a0)
        idx1 = np.delete(idx1,a0)       
        idx1 -= 1

        colors = dft.loc[idx0,'branch_id']

        idx0 -= N
        idx1 -= N
        
        isegs = np.concatenate( [idx0[:,None],idx1[:,None]],axis=1)
        print(isegs.shape)
        

        df_tot2 = self.df_tot

        df_tot= df_tot2.loc[df_tot2['traj_dist'] < 2.5,:]

        tcols = self.csk.traj_markers

        X = dft.loc[:,tcols].values

        X2 = df_tot.loc[:,tcols].values

        pca = pca_coords(X)

        ux,urad = pca.get_coords(X - pca.mu)

        apca = 0
        bpca = 1        

        uxx = ux[:,[apca,bpca]]

        glines = uxx[isegs]

        print(glines.shape)

        

        lc = LineCollection(glines,cmap='viridis')
        lc.set_array(np.array(colors))
        """

        lc = LineCollection(glines,color='g')
        #lc.set_array(np.array(colors))        
        """
        lc.set_linewidth(2)        

        ux2,urad2 = pca.get_coords(X2 - pca.mu)

        #plt.scatter(ux[:,0],ux[:,1],s=2,c=dft.loc[:,'branch_id'])
        #plt.scatter(ux[:,0],ux[:,1],s=2,c=dft.loc[:,'CD34'])
        #plt.scatter(ux2[:,0],ux2[:,1],s=2,c=df_tot.loc[:,'CD34'])

        fig,ax = plt.subplots()


        
        pnts = ax.scatter(ux2[:,apca],ux2[:,bpca],s=.5,c=df_tot.loc[:,'branch_id'],cmap=cm.jet,alpha=.8)
        cb = fig.colorbar(pnts,ax=ax)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        cb.set_label("branch_id")

        ax.add_collection(lc)

        ax.set_ylim(-3,1)

        cb2 = fig.colorbar(lc)
        cb2.set_label("trajectory branch_id")

        
        #plt.scatter(ux2[:,0],df_tot.loc[:,'CD34'],s=2,c=df_tot.loc[:,'branch_id'])
        #plt.scatter(ux[:,0],dft.loc[:,'CD34'],s=2,c=dft.loc[:,'branch_id'])
        #plt.scatter(dft.loc[:,'ptime'],dft.loc[:,'CD34'],s=2,c=dft.loc[:,'branch_id'])        

        plt.show()

        

    def write_df_tot(self,fname):
        #have option for R convention
        self.df_tot.to_csv(fname,index=False)
        #df_tot.to_csv("endo_even10_tdist.csv",index=False)
        



    def get_paths4(self,vstart,ends):
        cg = cgraph()
        self.cg = cg
        csk = self.csk
        
        cg.get_reduced_graph(csk.br_adj)      
        cg.forward_graph(cg.reduced_adj,start=vstart)
        self.br_adj = csk.br_adj

        self.cg = cg

        self.start = vstart

        print("start",vstart)

        print(self.cg.back_adj)

        paths = {}
        back_adj = self.cg.back_adj

        all_segs = set()

        #seg_paths[vp] is a set of end vert
        #specifying paths which run through segment vp
        seg_paths = {}

        for v in ends:
            segs = []
            v1 = v
            vadj = back_adj[v]
            while len(vadj) > 0:
                v0 =  vadj[0]
                #segs.append(self.segments_sym[(v0,v1)])
                segs.append((v0,v1))
                all_segs.add((v0,v1))                
                if (v0,v1) not in seg_paths:
                    seg_paths[(v0,v1)] = set()
                seg_paths[(v0,v1)].add(v)
                v1 = v0
                vadj = back_adj[v1]

            segs = segs[::-1]
            paths[v] = segs
            """
            paths is dictionary indexed by end vertices
            and each entry is list of correctly
            oriented segments from start to end
            that is, for key (v0,v1) v0 is the earlier vertex
            """

        #now seg_paths[vp] is a ordered tuple of
        #end vertices specifying which paths run thru segment
        #vp - each branch has a unique such tuple
        for key in seg_paths:
            a = seg_paths[key]
            a = list(a)
            a.sort()
            a = tuple(a)
            seg_paths[key] = a

        #redundant to above
        #seg_paths in indexed by segments (v0,v1)
        #and the item is a sorted tuple of the ends
        #the segment leads to

        #find unique branch tuples

        branch_labels0 = set()
        for key in seg_paths:
            a = seg_paths[key]
            branch_labels0.add(a)


        branch_labels0 = list(branch_labels0)

        print(branch_labels0)

        #sort the tuples by length, longest first
        #so that earlier "branches" will have lowest labels
        blen = [len(x) for x in branch_labels0]
        blen = np.array(blen)
        idx = np.argsort(blen)
        #branch_labels0 = np.array(branch_labels0)
        branch_labels0 = np.array(branch_labels0,dtype=np.object)
        branch_labels0 = branch_labels0[idx]
        branch_labels0 = branch_labels0[::-1]

        nlab = len(branch_labels0)

        #assign an integer label to each branch tuple, starting from 1
        #for the stem(s)
        #hmm, test this with V start
        branch_labels = {}
        for i in range(1,nlab+1):
            lab = branch_labels0[i-1]
            lab = tuple(lab)
            branch_labels[lab] = i

        print("branch labels",branch_labels)


        seg_branches = {}

        #assign each segment its branch id
        for vp in seg_paths:
            a = seg_paths[vp]
            seg_branches[vp] = branch_labels[a]

        print("seg_branches",seg_branches)

        #assign a branch label to each cell in
        #br_adj
        br_cells = list(self.br_adj.keys())
        br_cells.sort()
        nbr = len(br_cells)

        br_map = SortedDict()

        for vp in all_segs:
            seg = cg.segments[vp]
            for cell in seg[1:]:
                br_map[cell] = seg_branches[vp]

        br_map[vstart] = 1


        self.all_segs = all_segs


        self.br_map = br_map

        return paths,seg_branches



