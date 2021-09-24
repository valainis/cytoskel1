from .tumap import *
from .csk4 import *
import cytoskel1 as csk1
from .ofun import *

from sklearn.metrics import euclidean_distances
from sklearn import manifold


def ux_init0(csk,tscale):
    td = csk1.tree_dist(csk)
    td.splice()
    cg = csk.cg

    csk.cg.forward_graph(csk.cg.reduced_adj)
    start = csk.cg.start    

    N0 = td.N0
    N1 = td.N1

    vpos,vsegs,vedges = csk1.set_pos1(csk.cg)


    ends = cg.far_ends
    yends = []

    for v in ends:
        yends.append(vpos[v][1])

    ends = np.array(ends)
    yends = np.array(yends)

    idx = np.argsort(yends)

    ends = ends[idx]
    yends = yends[idx]

    #arrange them around circle from -np.pi to +np.py
    theta = np.linspace(-np.pi,np.pi,len(ends))/2.0

    r_edges = [ [cg.start,ends[0]] ]

    for i,v in enumerate(ends[:-1]):
        r_edges.append( [v, ends[i+1] ])

    r_edges.append( [ends[-1],cg.start])
    r_edges = np.array(r_edges)

    start = td.rmap0[start] - td.N0

    print("start 2", start)
    r_edges = td.rmap0[r_edges] - td.N0
    ends = td.rmap0[ends] - td.N0

    idx = np.arange(N1)
    unfix = idx != start
    unfix = idx[unfix]

    fix = idx == start

    X0 = np.zeros((N1,2))
    xpos = np.array([np.cos(theta),np.sin(theta)]).T

    X0[ends,:] = xpos
    X0[start,:] = np.array([-1.0,0.0])

    edges = td.tree_edges - N0
    edsel = edges[:,0] < edges[:,1]
    edges = edges[edsel]


    nf2 = nfun2(X0,unfix,edges,r_edges,n_iter=10,alpha=1.0)
    nopt0 = nopt()
    x00 = nopt0.minimize(nf2,X0.flatten())
    X0 = x00.reshape( X0.shape )

    r_edges = csk1.all_edges(X0.shape[0])
    f0 = fun0(X0,[start],edges,r_edges,alpha=1.0e-5)
    x0 = X0.flatten()

    options0 = {'disp':True,'gtol':1e-2}            
    res = sopt.minimize(f0,x0,tol=1e-3,method='CG',jac=True,options=options0)

    Xtraj = tscale*res.x.reshape(X0.shape)

    return Xtraj,td,edges
    


def ux_init2(csk,tscale,seed=3738):
    td = csk1.tree_dist(csk)
    df_tot,tree_edges = td.splice()    

    cg = csk.cg

    csk.cg.forward_graph(csk.cg.reduced_adj)

    N = csk.df.shape[0]

    X = df_tot.loc[:,csk.traj_markers].values
    X =X[N:]

    print("X",X.shape)


    Xmu = np.mean(X,axis=0)
    dX = X - Xmu

    tedges = tree_edges - N
    dist = euclidean_distances(dX)

    #np.random.seed(3738)
    np.random.seed(seed)
    t0 = time.time()
    mds = manifold.MDS(n_components=2,dissimilarity="precomputed")
    ux2 = mds.fit(dist).embedding_
    t1 = time.time()

    print("niter",mds.n_iter_)

    ####
    tedges = td.tree_edges - N
    edsel = tedges[:,0] < tedges[:,1]
    tedges = tedges[edsel]


    print("time",t1-t0)

    segs = ux2[tedges]

    segs0 = np.mean(segs,axis=1,keepdims=True)

    dsegs = segs - segs0

    segs = .9*dsegs + segs0

    e0 = tedges[:,0]
    e1 = tedges[:,1]



    Xtraj = tscale*ux2

    return Xtraj,td,tedges



class tmap:
    def __init__(
            self,
            csk,
            n_knn = 10,
            tscale = 2.0,
            min_dist = .05,
            spread = 1.0,
            read_knn = False,
            use_avg = False,
            have_fixed = True,
            mds_init = False
            ):
        #csk is self
        cg = csk.cg

        N0 = csk.df.shape[0]

        #Xtraj = ux_init0(csk,tscale)
        if mds_init:
            Xtraj,td,edges = ux_init2(csk,tscale)        
        else:
            Xtraj,td,edges = ux_init0(csk,tscale)

        print(Xtraj.shape)

        """
        plt.scatter(Xtraj[:,0],Xtraj[:,1])
        plt.show()

        exit()
        """

        X = td.df_tot.loc[:,csk.traj_markers].values

        t0 = time.time()
        tm4 = tumap(
            csk,
            td,
            min_dist,
            spread
            )
        tm4.init_umap00(n_knn,X,read_knn)
        tm4.init_pos0(Xtraj)

        tm4.run00(200,have_fixed)
        #tm4.run00(0,have_fixed)
        t1 = time.time()

        print("time 0",t1-t0)

        uhead = ["tiUMAP1","tiUMAP2"]
        dfu = pd.DataFrame(tm4.thead,columns=uhead)
        df2 = pd.concat([td.df_tot,dfu],axis=1)
        df2.to_csv(csk.project_dir + "df_tumap.csv",index = False)

        Xtraj1 = tm4.thead[N0:,:]
        segs = Xtraj1[edges]


        Rsegs = segs.reshape( (-1,4) )

        csk.df_rsegs = pd.DataFrame(Rsegs,columns=["x","y","xend","yend"])

        csk.df_tumap = df2
        csk.segs = segs


