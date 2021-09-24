import sys
from collections import OrderedDict
from sortedcontainers import SortedDict,SortedList
import scipy.sparse as sp

import numpy as np
import numpy.linalg as la

import queue as Q


class directed_adj:

    def __init__(self,adj,start):

        back_adj = SortedDict()
        for_adj = SortedDict()
        back_edges = []
        for_edges = []

        far_ends = []

        lifo = Q.LifoQueue()
        lifo.put( (start,-1))


        while not lifo.empty():
            edge = lifo.get()
            v0 = edge[0]
            vprev = edge[1]

            if vprev == -1:
                back_adj[v0] = []
            else:
                back_adj[v0] = [vprev]

            for_adj[v0] = []

            for v1 in adj[v0]:
                if v1 == vprev: continue
                for_adj[v0].append(v1)
                for_edges.append((v0,v1))
                back_edges.append((v1,v0))
                lifo.put((v1,v0))

        for v in for_adj:
            vadj = for_adj[v]
            if len(vadj) == 0:
                far_ends.append(v)

        self.adj = adj
        self.start = start
        self.back_adj = back_adj
        self.for_adj = for_adj

        self.far_ends = far_ends
        self.for_edges = for_edges
        self.back_edges = back_edges




    def dots(self,df):
        pcells = list(self.adj.keys())
        map0,rmap0 = mk_maps(df,pcells)

        n_cells = len(pcells)

        p = df.shape[1]

        back_edges = np.array(rmap0[self.back_edges],dtype=np.int)

        X = df.loc[pcells,:].values

        A = sp.dok_matrix( (n_cells,n_cells))

        idx0 = np.arange(n_cells)

        A[back_edges[:,0],back_edges[:,1]] = -1.0
        A[idx0,idx0] = 1.0

        A = A.tocsr()

        T = A @ X

        rstart = rmap0[self.start]
        Tn = la.norm(T,axis=1)

        T1 = T / Tn[:,None]

        #get the average tangent at start

        Tavg = np.zeros((p,))
        for v in self.for_adj[self.start]:
            v = rmap0[v]
            Tavg += T[v]

        Tavg /= len(self.for_adj)

        Tavg_n = la.norm(Tavg)

        Tavg_1 = Tavg / Tavg_n

        T1[rstart] = Tavg_1

        Tn[rstart] = Tavg_n

        Tdev = A @ T1

        Tdev_n = la.norm(Tdev,axis=1)

        return Tn,Tdev_n









def get_edge_costs(br_adj,tdata):
    edges = []
    ecost = SortedDict()

    
    for v0 in br_adj:
        links = br_adj[v0]
        p0 = tdata[v0]
        ecost[v0] = []
        for v1 in links:
            p1 = tdata[v1]
            edges.append((v0,v1))
            dx = p1 - p0
            ds = la.norm(dx)
            ecost[v0].append(ds)
    return edges,ecost




def adj2csr(adj):
    N = len(adj)

    lil_adj = sp.lil_matrix((N,N))

    rows = []
    data = []

    for row in adj:
        rdata = [1.0]*len(row)
        data.append(rdata)

    lil_adj.rows = np.array(adj,dtype=object)
    lil_adj.data = np.array(data,dtype=object)

    csr_adj = lil_adj.tocsr()

    print("adj nnz",csr_adj.nnz)

    return csr_adj



def read_sdict(sdict_name):
    """
    read integer dict from file object
    return as SortedDict
    """

    fadj = open(sdict_name,"r")

    s = fadj.read()
    s = s.split("\n")
    s = [x.strip() for x in s]
    s = [x for x in s if x != ""]
    s = [x.split() for x in s]


    sdict = SortedDict()

    for row in s:
        #icell = int(row[0])
        row = [int(x) for x in row]
        sdict[row[0]] = row[1:]

    fadj.close()

    return sdict


def read_sdict2(sdict_name):
    """
    read integer dict from file object
    return as SortedDict
    """

    fadj = open(sdict_name,"r")

    s = fadj.read()
    s = s.split("\n")
    s = [x.strip() for x in s]
    s = [x for x in s if x != ""]
    s = [x.split() for x in s]

    N = len(s)

    lil_adj = sp.lil_matrix((N,N))


    rows = []
    data = []

    for row in s:
        #icell = int(row[0])
        row = [int(x) for x in row]
        rdata = [1.0]*len(row[1:])
        rows.append(row[1:])
        data.append(rdata)

        

    lil_adj.rows = np.array(rows,dtype=object)
    lil_adj.data = np.array(data,dtype=object)

    csr_adj = lil_adj.tocsr()

    fadj.close()

    return csr_adj


#get the info from run.setup etc
class get_info:
    def __init__(self,sname):
        fsu = open(sname,"r")
        s = fsu.read()
        s = s.split("\n")
        #should clean empty and commented lines
        self.s = s
        fsu.close()

        #no reason not to do this here
        self.get_marker_usage()
    
    def get_marker_usage(self):
        s = self.s
        mstart = 0
        for i,s1 in enumerate(s):
            if "marker" in s1:
                mstart = i+1

        s = s[mstart:]
        marker_odict = OrderedDict()

        for i,s1 in enumerate(s):
            s2 = s1.split(',')
            s2 = [ys.strip() for ys in s2]
            s2 = [ys for ys in s2 if ys != ""]
            if len(s2) == 2:
                marker_odict[ s2[0] ] = float(s2[1])

        self.marker_odict = marker_odict

    def get_traj_cols(self):
        cols = []
        for m in self.marker_odict:
            use = self.marker_odict[m]
            if use == 1.0:
                cols.append(m)

        return cols

    def get_avg_cols(self):
        cols = []
        for m in self.marker_odict:
            use = self.marker_odict[m]
            if use == 1.0 or use == 2.0:
                cols.append(m)

        return cols

    def get_mon_cols(self):
        #columns to only be monitored
        cols = []
        for m in self.marker_odict:
            use = self.marker_odict[m]
            if use == 3.0:
                cols.append(m)

        return cols

    def get_all_mon_cols(self):
        cols = []
        for m in self.marker_odict:
            use = self.marker_odict[m]
            if use == 1.0 or use == 2.0 or use == 3.0:
                cols.append(m)

        return cols
    
    

    def get_adj_dict(self,adj_name):
        fadj = open(adj_name,"r")
        s = fadj.read()
        s = s.split("\n")
        s = [x.strip() for x in s]
        s = [x for x in s if x != ""]
        s = [x.split() for x in s]


        br_adj = SortedDict()

        for row in s:
            #icell = int(row[0])
            row = [int(x) for x in row]
            br_adj[row[0]] = row[1:]

        return br_adj

    def get_segments(self,seg_file):
        fseg = open(seg_file,"r")
        s = fseg.read()
        s = s.split("\n")
        s = [x.strip() for x in s]
        s = [x for x in s if x != ""]

        s = [x.split() for x in s]

        segments = {}
        vset = set()

        for row in s:
            row = [int(x) for x in row]
            v0 = row[0]
            v1 = row[-1]
            vp = (v0,v1)

            segments[vp] = row
            vset.add(v0)
            vset.add(v1)
        return segments,vset

#assumes each row number is cell index
def write_full_adj(adj,fname):
    afile = open(fname,"w")
    adj_str = []
    for i,row in enumerate(adj):
        srow = [str(i)]
        row = [str(x) for x in row]
        srow.extend(row)
        adj_str.append(" ".join(srow))
    adj_str = "\n".join(adj_str)
    afile.write(adj_str+"\n")
    afile.close()

def read_header(fname):
    f = open(fname,"r")

    markers = []

    for line in f.readlines():
        line = line.strip()
        if line != "":
            markers.append(line)

    return markers

#adj is sortedDict of lists
def write_sub_adj(adj,fname):
    afile = open(fname,"w")
    adj_str = []
    for i in adj:
        srow = [str(i)]
        row = [str(x) for x in adj[i]]
        srow.extend(row)
        adj_str.append(" ".join(srow))
    adj_str = "\n".join(adj_str)
    afile.write(adj_str)
    afile.close()


def mk_maps(df,pcells):

    map0 = np.array(df.loc[pcells,:].index)
    npcells = len(pcells)

    idx0 = np.arange(npcells)

    N = df.shape[0]

    rmap0 = np.full((N,),-1,dtype=np.int)
    rmap0[map0] = idx0

    #alternative to above , not used
    #rmap2 = np.full((N,),-1,dtype=np.int)
    #rmap2[pcells] = idx0

    return map0,rmap0


def mk_rmap0(N,pcells):

    map0 = np.array(pcells,dtype=int)
    npcells = len(pcells)

    idx0 = np.arange(npcells)

    rmap0 = np.full((N,),-1,dtype=np.int)
    rmap0[map0] = idx0

    return map0,rmap0


def write_setup(sname,
                    markers,
                    traj_m,
                    avg_m = [],
                    mon_m = []):
    f = open(sname,"w")

    s0 = "%-12s" % "marker" + " , " + "use" + "\n"
    f.write(s0)

    #print("extra",setup.extra)

    for i,m0 in enumerate(markers):
        useit = 0

        if m0 in traj_m:
            useit = 1
        elif m0 in avg_m:
            useit = 2
        elif m0 in mon_m:
            usit = 3

        s1 = "%-12s" % m0 + " , " + "%3d" % useit
        f.write(s1)

        f.write("\n")
    f.write("\n")
    f.close()


    


class pca_coords:
    def __init__(self,vecs,cdim=-1):
        """
        the pca eigenvectors and eigenvalues
        cdim is how many pca coords to use
        """
        if cdim == -1:
            cdim = vecs[0].shape[0]
        self.cdim = cdim

        mu = np.mean(vecs,axis=0)
        vecs0 = vecs - mu
        kov = np.cov(vecs0.T,bias=True)

        self.mu = mu
        self.kov = kov

        self.vecs0 = vecs0

        evals,evecs = la.eigh(kov)
        evals = evals[::-1]
        evecs = evecs[:,::-1]

        self.evecs = evecs
        self.evals = evals        

        uu = []

        #dont need this loop just do
        #self.uu = evecs[:,:cdim]

        for i in range(cdim):
            u0 = evecs[:,i]
            uu.append(u0)

        uu = np.array(uu).T
        self.uu = uu

    def pca_df(self,tmark):
        nv = self.evecs.shape[1]
        cols = [str(i) for i in range(nv)]
        print(cols)

        dfpca = pd.DataFrame(self.evecs,columns=cols,index=tmark)

        dfpca.to_csv("pca.csv")

    def get_coords(self,vecs0):
        ucoord = vecs0 @ self.uu
        unorms = la.norm(ucoord,axis=1)
        urad = np.amax(unorms)

        return ucoord, urad

    def fix_evecs(self):
        n = self.evecs.shape[1]
        for i in range(n):
            evec = self.evecs[:,i]
            sgn = evec @ self.mu
            if sgn < 0.0:
                self.evecs[:,i] *= -1.0
            sgn = evec @ self.mu                
            print("sgn",i,sgn)

    def project(self,vecs0,nlim):
        P = self.uu[:,:nlim]
        mu0 = np.mean(vecs0,axis=0)
        vecs00 = vecs0 - mu0
        ucoord = vecs00 @ P
        vecs00 = ucoord @ P.T
        vecs00 += mu0
        return vecs00

    def mdist(self,vecs,nlim=None):
        if nlim == None:
            nlim = self.cdim
            
        vecs0 = vecs - self.mu
        ucoord,urad = self.get_coords(vecs0)
        dist = ucoord[:,:nlim]**2/self.evals[:nlim]
        dist = np.sum(dist,axis=1)**.5
        return dist

    def mahala_g(self,evals,evecs,nlim=None):
        """
        setup mahalanobis metric
        """
        icomp = range(nlim)

        n = evals.shape[0]
        gm = np.zeros((n,n))

        for i in icomp:
            gm += np.outer(evecs[:,i],evecs[:,i]) / evals[i]

        return gm

    def show_evals2(self):
        n = self.evals.shape[0]
        x = range(n)
        fig,ax = plt.subplots()
        ax.scatter(x,self.evals)
        plt.show()

    def show_evals(self):
        n = self.evals.shape[0]
        x = range(n)
        fig,ax = plt.subplots()
        ax.scatter(x,self.evals**.5)
        plt.show()
    
