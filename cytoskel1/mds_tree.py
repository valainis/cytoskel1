import sys
import os
import time

from itertools import product,permutations,chain
from sortedcontainers import SortedDict
import copy

import numpy as np
import numpy.linalg as la
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection,PatchCollection
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.collections as mc

from sklearn.metrics import euclidean_distances
from sklearn import manifold
from sklearn.manifold import smacof

from .tmap import ux_init2
from .gu import critical0
from .csk4 import mk_submatrix


def csr_edges(csr):
    coo = csr.tocoo()
    e0 = coo.row
    e1 = coo.col

    edges = np.array([e0,e1],dtype=int).T

    return edges,e0,e1


def mk_rmap0(N,pcells):
    #N is the size of the full cell set

    npcells = len(pcells)
    idx0 = np.arange(npcells)
    rmap0 = np.full((N,),-1,dtype=np.int)
    rmap0[pcells] = idx0

    return rmap0


def ax_plot0(fig,ax,ux2,df_avg,segs,e0,e1,map0,splot=True,name=None):
    #df_avg is now a Series
    #splot True means normal, False means blank out everything

    cmap0 = mpl.cm.jet
    #cmap0 = mpl.cm.inferno    
    #cmap0 = mpl.cm.tab20

    if not splot:

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return

    m = df_avg.name
    #color = df_avg.loc[map0,m].values
    color = df_avg.loc[map0].values

    vmax = np.amax(color)
    vmin = np.amin(color)
    dv = vmax - vmin
    ecolor0 = color[e0]
    ecolor1 = color[e1]
    ecolor = .5*(ecolor0 + ecolor1)

    ecolor = (ecolor-vmin)/dv

    #pnts = ax.scatter(ux2[:,0],ux2[:,1],s=15,c=color,cmap=mpl.cm.jet,vmin=0.0)
    #pnts = ax.scatter(ux2[:,0],ux2[:,1],s=15,c=color,cmap=mpl.cm.inferno,vmin=0.0)
    pnts = ax.scatter(ux2[:,0],ux2[:,1],s=15,c=color,cmap=cmap0,vmin=vmin,vmax=vmax)

    #rgba = mpl.cm.jet(ecolor)
    rgba = cmap0(ecolor)
    #rgba = mpl.cm.inferno(ecolor)

    seg_col = mc.LineCollection(segs,color=rgba)
    #seg_col = mc.LineCollection(segs)
    if name:
        fig.colorbar(pnts,ax=ax,label=name)
    else:
        fig.colorbar(pnts,ax=ax)
    ax.add_collection(seg_col)
    ax.set_xlabel(m)

    #ax.axis('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])        
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

def ax_dplot0(fig,ax,ux2,df_col,segs,e0,e1,map0,dcolor=None):
    #df_col is now a Series

    cmap0 = mpl.cm.inferno    
    m = df_col.name    

    xcol = df_col.values

    scol = list(set(xcol))
    scol.sort()

    print(scol)

    #dcolor = ['r','tab:orange','k','g','b','m','y']

    dcolor = ['r','y','g','b','m','k','c']
    

    #color = df_col.loc[map0,m].values
    color0 = df_col.loc[map0].values

    #ux2 marker cluster designnation

    """
    color = []

    for cx in color0:
        color.append(dcolor[cx])
    """
    marks = ['o','v','^','<','>','s']

    for s in scol:
        sel = color0 == s
        nsel = np.sum(sel)
        print("cluster",s,nsel)
        if nsel > 0:
            ux_sel = ux2[sel]
            pnts = ax.scatter(ux_sel[:,0],ux_sel[:,1],s=30,c=dcolor[s % 7],label="cluster " + str(s),marker=marks[s % 6])


    seg_col = mc.LineCollection(segs,color='k',alpha=.3)

    #fig.colorbar(pnts,ax=ax)
    ax.add_collection(seg_col)
    ax.set_xlabel(m)

    #ax.legend(loc='upper right',bbox_to_anchor=(1.1, 1.05))
    ax.legend(loc='upper right',bbox_to_anchor=(1.1, 1.0))              

    #ax.axis('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])        
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)


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


class mds_info:
    def __init__(self,df_avg):
        pass


class mdsxx:
    def __init__(self,df_avg,df_seg0,icells,edges,init=None,seed=137,xfac=1.0):
        map0,rmap0 = mk_maps(df_avg,icells)
        edges_0 = rmap0[edges]

        self.e0 = edges_0[:,0]
        self.e1 = edges_0[:,1]

        X = df_seg0.loc[icells,:].values
        dist = euclidean_distances(X)

        if init is None:
            np.random.seed(seed)

        #n_init should not be 1 for random seed case
        ux2,stress,n_iter = smacof(dist,init=init,n_init=1,return_n_iter=True)
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


    def plot0(self,nrow,ncol,clist):

        df_avg = self.df_avg
        ux2 = self.ux2
        segs = self.segs
        e0 = self.e0
        e1 = self.e1

        map0 = self.map0
        
        pfac = 2.8

        ww = pfac*ncol
        hh = pfac*nrow

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

    def plot1(self,df_col,save=False):

        ux2 = self.ux2
        segs = self.segs
        e0 = self.e0
        e1 = self.e1

        map0 = self.map0

        fac = 1.0
        w = fac*8
        h = fac*7

        fig,ax = plt.subplots(figsize=(13,9.5))


        ax_plot0(fig,ax,ux2,df_col,segs,e0,e1,map0)

        plt.tight_layout()
        if save:
            plt.savefig("save_"+df_col.name+".pdf",format="pdf")
        else:
            plt.show()


class mds2_plot:
    def __init__(self,csk,xfac=1.0,yfac=1.0,fig_file=None):
        if not hasattr(csk,'df_mds2'):
            print('cytoskel object has no attribute df_mds2')
            print('run mds_tree.mk_tree')
            exit()

        self.fig_file = fig_file
        self.csk = csk

        if hasattr(csk,'csr_br'):
            self.csr_br = csk.csr_br

        self.ux2 = csk.df_mds2.values
        self.ux2[:,0] *= xfac
        self.ux2[:,1] *= yfac        
        self.br_adj = csk.br_adj
        edges = get_edges(self.br_adj)        
        
        self.tree_cells = np.array( list( self.br_adj.keys() ), dtype=int)
        

        map0,rmap0 = mk_maps(csk.df_avg,self.tree_cells)
        self.map0 = map0 #same as tree_cells
        self.rmap0 = rmap0
        edges_0 = rmap0[edges]

        self.e0 = edges_0[:,0]
        self.e1 = edges_0[:,1]

        segs = self.ux2[edges_0]
        segs0 = np.mean(segs,axis=1,keepdims=True)
        dsegs = segs - segs0
        segs = .95*dsegs + segs0
        self.segs = segs


    def subgraph(self,keep):
        """
        #construct what is needed for plotting a subgraph
        #of the branch graph
        #results used by kplot1
        """

        self.keep = keep
        csr_br2 = mk_submatrix(keep,self.csk.csr_br)

        #the ux2 indices of the keep cells
        keep_0 = self.rmap0[keep]

        kpnts = self.ux2[keep_0]
        self.kpnts = kpnts

        kedges,ke0,ke1 = csr_edges(csr_br2)        



        krmap0 = mk_rmap0(self.rmap0.shape[0],keep)
        kedges_0 = krmap0[kedges]
        self.kedges_0 = kedges_0

        ksegs = kpnts[kedges_0]
        segs0 = np.mean(ksegs,axis=1,keepdims=True)
        dsegs = ksegs - segs0
        ksegs = .95*dsegs + segs0

        self.ksegs = ksegs
        

        
    def plot1(self,df_col,save=False,crit_pnts=None,xfig=13,yfig=9.5):

        ux2 = self.ux2
        segs = self.segs
        e0 = self.e0
        e1 = self.e1

        map0 = self.map0

        fig,ax = plt.subplots(figsize=(xfig,yfig))

        ax_plot0(fig,ax,ux2,df_col,segs,e0,e1,map0,name=df_col.name)

        if crit_pnts:
            xcrit = ux2[self.rmap0[crit_pnts]]
            print(xcrit)

            ax.scatter(xcrit[:,0],xcrit[:,1],c='m',s=80)

        plt.tight_layout()
        if save:
            plt.savefig("save_"+df_col.name+".pdf",format="pdf")
        else:
            plt.show()


    def kplot1(self,df_col,save=""):
        """
        plot df_col , one marker
        uses results of subgraph to plot the subgraph
        but using the full mds2 coordinates
        """

        ux2 = self.kpnts
        segs = self.ksegs

        e0 = self.kedges_0[:,0]
        e1 = self.kedges_0[:,0]

        map0 = self.keep

        fac = 1.0
        w = fac*8
        h = fac*7

        fig,ax = plt.subplots(figsize=(13,9.5))


        ax_plot0(fig,ax,ux2,df_col,segs,e0,e1,map0)

        plt.tight_layout()
        if save:
            plt.savefig(save+df_col.name+".pdf",format="pdf")
        else:
            plt.show()


    def plot0(self,nrow,ncol,clist,df_avg,pfac=4.0):
        """
        plot markers in clist using
        values in df_avg
        """

        print("mds2_plot plot0")

        ux2 = self.ux2
        br_adj = self.br_adj
        edges = get_edges(br_adj)
        N = df_avg.shape[0]

        segs = self.segs

        e0 = self.e0
        e1 = self.e1   

        #pfac = 2.8
        #pfac = 4

        ww = pfac*ncol
        hh = pfac*nrow

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

            color = df_avg.loc[self.map0,clist[i]].values
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
        if self.fig_file:
            plt.savefig(self.fig_file,format='pdf',bbox_inches='tight')
        plt.show()

    def cat_plot1(self,df_col,save=""):
        """
        #df_col has a set of integer values
        mds plot of categorical values
        """

        ux2 = self.ux2
        segs = self.segs
        e0 = self.e0
        e1 = self.e1


        map0 = self.map0

        fac = 1.0
        w = fac*8
        h = fac*7


        fig,ax = plt.subplots(figsize=(13,9.5))


        ax_dplot0(fig,ax,ux2,df_col,segs,e0,e1,map0)

        plt.tight_layout()
        if save:
            plt.savefig("save_"+df_col.name+".pdf",format="pdf")
        else:
            plt.show()
            
            



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


def crit_segs_dx(tj1):
    N = tj1.csk.df.shape[0]    
    crit,crit_segs = tj1.do_critical(tj1.csk.cg.segments)
    #crit,crit_segs = tj1.do_critical(crit_segs,do_all=True)
    crit = crit - N

    return crit,crit_segs


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

        ilist = harch(X,1)
        imid = ilist[1]

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


    def mk_tree(self,seed=137,level=1,xfac=1.0,traj_markers=None):

        if not traj_markers:
            traj_markers = self.traj_markers
        df_seg0 = self.df_avg.loc[:,traj_markers]

        crit0 = critical0(self.csk,level)
        crit_segs1 = crit0.crit_segs1
        self.crit_segs1 = crit_segs1
        self.crit0 = crit0
       

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
        mds = mdsxx(self.df_avg,df_seg0,self.tree_cells,tedges,init=ux4_0,xfac=xfac)

        self.mds= mds

        #save the mds coordinates
        self.write_mds2_coords()


    def write_mds2_coords(self):
        mds = self.mds
        df_mds2 = pd.DataFrame(mds.ux2,columns=["mds1","mds2"],index=mds.map0)
        df_mds2.to_csv(self.csk.project_dir + "df_mds2.csv")
        self.csk.df_mds2 = df_mds2

        


    def plot(self,clist,nrow,ncol):
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
        mds = mdsxx(self.df_avg,df_seg0,self.tree_cells,tedges,init=ux4_0)
        mds.plot0(nrow,ncol,clist)
