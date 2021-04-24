#import cytoskel1 as csk1

import scipy.optimize as sopt
#from copt import *

import time
import numpy as np
import numpy.linalg as la
import pandas as pd

import scipy.sparse as sp
from itertools import product as cprod

import queue as Q
from sortedcontainers import SortedDict,SortedList


import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib import cm


class nfun2:
    def __init__(
            self,
            X0,
            unfix,
            edges,
            r_edges,
            n_iter = 100,
            alpha = 1.0
            ):
        
        # X0[-1] is start, taken as fixed
        self.X0 = X0
        self.unfix = unfix
        self.p = X0.shape[1]
        self.N = X0.shape[0]
        self.edges = edges
        self.r_edges = r_edges

        self.alpha = alpha
        self.n_iter = n_iter

        self.X = self.X0.copy()

        self.pits = []
        self.Xpits = []

    def step1(self):
        N = self.N
        p = self.p
        X = self.X

        pits = self.pits
        Xpits = self.Xpits

        alpha = self.alpha

        eps0 = 1e-7

        edges = self.edges
        r_edges = self.r_edges


        #internal edge contribution
        e0 = edges[:,0]
        e1 = edges[:,1]

        phi = 0.0
        k0 = 2.0

        
        G = np.zeros( (N,N,p) )
        dX = X[e0] - X[e1]

        phi += .5*k0*np.sum(dX*dX)

        G[e0,e1,:] = dX
        G[e1,e0,:] = -dX

        #the gradient
        g0 = k0*np.sum(G,axis=1)

        J = np.zeros( (N,N,p,p) )

        Ip = k0*np.eye(p)

        for e in edges:
            J[e[0],e[0],:,:] += Ip
            J[e[1],e[1],:,:] += Ip
            J[e[0],e[1],:,:] -= Ip
            J[e[1],e[0],:,:] -= Ip

        #finished with internal edge contribution
        #now do the rim repulsions

        re0 = r_edges[:,0]
        re1 = r_edges[:,1]
        GR = np.zeros( (N,N,p) )
        dX = X[re0] - X[re1]

        dx2 = np.sum(dX*dX,axis=1).reshape((-1,1))
        dx = np.sqrt(dx2)
        dx3 = (dx2*dx).reshape( (-1,1,1) )

        #alpha = 1.0

        phi += alpha * np.sum(1/dx)

        self.phi = phi

        dXn = dX/(dx + eps0)

        Ip = np.eye(p).reshape( (1,p,p) )       
        dXn2 = (Ip - 3*np.einsum("ij,ik->ijk",dXn,dXn))/dx3

        dXn2 *= alpha

        dx3 = dx3.reshape( (-1,1) )

        rg0 = -dX/(dx3+eps0)

        rg0 *= alpha

        #should be ok if edges are all distinct
        GR[re0,re1,:] = rg0
        GR[re1,re0,:] = -rg0

        rg0 = np.sum(GR,axis=1)

        g0 = g0 + rg0
        self.g0 = g0

        #J = np.zeros( (N,N,p,p) )
        for i,e in enumerate(r_edges):
            J[e[0],e[0],:,:] -= dXn2[i]
            J[e[1],e[1],:,:] -= dXn2[i]
            J[e[0],e[1],:,:] += dXn2[i]
            J[e[1],e[0],:,:] += dXn2[i]


        kpits = 1.0
        Ip = k0*np.eye(p)
        for i,icell in enumerate(pits):
            dX = X[icell] - Xpits[i]
            phi += .5 * kpits * np.sum(dX*dX)
            g0[icell] += kpits*dX

            J[icell,icell,:,:] += kpits * Ip

        unfix = self.unfix

        N_un = unfix.shape[0]
        Np_un = N_un * p
        self.Np_un = Np_un

        J = J[unfix[:,None],unfix]

        J2 = np.swapaxes(J,1,2)
        J2 = J2.reshape( (Np_un,Np_un) )
        self.J2 = J2

    def form_rhs(self):
        #g0[-1] will be neglected
        N = self.N
        p = self.p
        X = self.X
        edges = self.edges
        
        e0 = edges[:,0]
        e1 = edges[:,1]


        self.JJ = self.J2

        #form the RHS
        rhs = np.zeros(self.Np_un)
        g0 = self.g0[self.unfix]
        g0 = g0.flatten()

        self.g0 = g0

    def __call__(self,x):
        p = self.p
        N = self.N
        X = x.reshape( (-1,p) )

        self.X = X

        self.step1()
        self.form_rhs()

        return self.phi,self.g0,self.JJ
        


class nopt:
    def minimize(self,fun,x0):
        p = fun.p
        self.fun = fun
        self.unfix = fun.unfix

        phi,g0,JJ = fun(x0)

        phi0 = phi

        X = x0.reshape( (-1,p) )

        x0prev = x0.copy()

        for i in range(fun.n_iter):
            dX0 = la.solve(JJ,-g0)
            dX = dX0.reshape((-1,p))

            X[self.unfix,:] += dX

            x0 = X.flatten()
            phi,g0,JJ = fun(x0)

            
            if phi >= phi0:
                print("done")

                U,S,Vh = la.svd(JJ)
                print("cond",S[0]/S[-1])

                try:
                    L = la.cholesky(JJ)
                except:
                    print("not pos def")


                print("done",i)
                return x0prev
                break
            else:
                print("phi0", phi0)
                phi0 = phi
                x0prev = x0

        return x0


class fun0:
    def __init__(
            self,
            X0,
            fix,
            edges,
            r_edges,
            alpha = 1.0,
            k0 = 2.0
            ):
        
        # X0[-1] is start, taken as fixed
        self.X0 = X0
        self.fix = fix
        self.p = X0.shape[1]
        self.N = X0.shape[0]
        self.edges = edges
        self.r_edges = r_edges

        self.alpha = alpha
        self.k0 = k0

        self.X = self.X0.copy()

    def __call__(self,x):
        p = self.p
        N = self.N
        k0 = self.k0
        alpha = self.alpha

        fix = self.fix

        eps0 = 1e-7        
        
        X = np.reshape(x,(-1,p))

        edges = self.edges
        r_edges = self.r_edges


        #internal edge contribution
        e0 = edges[:,0]
        e1 = edges[:,1]

    
        phi = 0.0
        k0 = 2.0

        
        G = np.zeros( (N,N,p) )
        dX = X[e0] - X[e1]

        phi += .5*k0*np.sum(dX*dX)

        G[e0,e1,:] = dX
        G[e1,e0,:] = -dX

        #the gradient
        g0 = k0*np.sum(G,axis=1)

        re0 = r_edges[:,0]
        re1 = r_edges[:,1]
        GR = np.zeros( (N,N,p) )
        dX = X[re0] - X[re1]

        dx2 = np.sum(dX*dX,axis=1).reshape((-1,1))
        dx = np.sqrt(dx2)
        dx3 = (dx2*dx).reshape( (-1,1,1) )

        #alpha = 1.0

        phi += alpha * np.sum(1/dx)

        dx3 = dx3.reshape( (-1,1) )

        rg0 = -dX/(dx3+eps0)

        rg0 *= alpha

        #should be ok if edges are all distinct
        GR[re0,re1,:] = rg0
        GR[re1,re0,:] = -rg0

        rg0 = np.sum(GR,axis=1)

        g0 = g0 + rg0

        g0[fix] = 0.0

        self.g0 = g0.copy()

        g0 = g0.flatten()

        return phi, g0
