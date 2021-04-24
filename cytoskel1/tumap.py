import cytoskel1 as csk1

import scipy.optimize as sopt

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

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import numba

import os

from scipy.optimize import curve_fit

#locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result



@numba.njit(parallel=True, fastmath=True)
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In esscence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1])
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0


            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            if result[i] < MIN_K_DIST_SCALE * np.mean(ith_distances):
                result[i] = MIN_K_DIST_SCALE * np.mean(ith_distances)
        else:
            if result[i] < MIN_K_DIST_SCALE * np.mean(distances):
                result[i] = MIN_K_DIST_SCALE * np.mean(distances)

    return result, rho


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]



@numba.njit("i4(i8[:])")
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xffffffff) ^ (
        (((state[0] << 13) & 0xffffffff) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xffffffff) ^ (
        (((state[1] << 2) & 0xffffffff) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xffffffff) ^ (
        (((state[2] << 3) & 0xffffffff) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]


@numba.njit()
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


@numba.njit("f4(f4[:],f4[:])", fastmath=True)
def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2

    return result


#@numba.njit(fastmath=True, parallel=True)
@numba.njit(fastmath=True)
def optimize_layout(
    head_embedding,
    tail_embedding,
    head,
    tail,
    not_fixed,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    verbose=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).

    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.

    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.

    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.

    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.

    n_epochs: int
        The number of training epochs to use in optimization.

    n_vertices: int
        The number of vertices (0-simplices) in the dataset.

    epochs_per_samples: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.

    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.

    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    imin = np.argmin(epochs_per_sample)


    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    print("a b vert epochs/sample ",a,b,n_vertices,epochs_per_sample.shape[0])
    print("head",head.shape)

    nmax = 0

    ng0 = np.zeros(100)

    count = 0.0
    neg_count = 0.0
    for n in range(n_epochs):
        if n%10 == 0: print("n",n)

        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= n:            
                j = head[i]
                k = tail[i]
                count += 1

                current = head_embedding[j]
                other = tail_embedding[k]

                curr_not_fixed = not_fixed[j]
                other_not_fixed = not_fixed[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    if curr_not_fixed: current[d] += grad_d * alpha
                    if move_other:
                        if other_not_fixed: other[d] += -grad_d * alpha

                #since epochs_per_sample is wmax/w , w = weights
                #this causes i to be done roughly every wmax/w values of n
                epoch_of_next_sample[i] += epochs_per_sample[i]

                n_neg_samples = int(
                    (n - epoch_of_next_negative_sample[i])
                    / epochs_per_negative_sample[i]
                )

                ng0[n_neg_samples] = ng0[n_neg_samples] + 1
                if n_neg_samples > nmax:
                    nmax = n_neg_samples

                for p in range(n_neg_samples):
                    neg_count += 1.0
                    k = tau_rand_int(rng_state) % n_vertices

                    other = tail_embedding[k]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (
                            a * pow(dist_squared, b) + 1
                        )
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                        else:
                            grad_d = 4.0
                        if curr_not_fixed: current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")
    print("tot count",count,"neg",neg_count)
    return head_embedding



class tumap:
    def __init__(
            self,
            csk,
            td,
            min_dist = .05,
            spread = 1.0
            ):
        cg = csk.cg
        self.csk = csk
        self.cg = cg
        self.td = td

        self.min_dist = min_dist
        self.spread = spread

        self.N0 = self.csk.df.shape[0]


    def init_umap00(self,nn,X0,read_knn=False):
        rough = False

        nn = nn+1

        print("ummp00")

        if read_knn and os.path.isfile(self.csk.project_dir+"tmap_knn.npz"):
            print("reading knn")
            knn1 = np.load(self.csk.project_dir+"tmap_knn.npz")
            adj = knn1['adj'][:,1:]
            dist0 = knn1['dist']            
            dist = knn1['dist'][:,1:]
        else:
            knn0 = csk1.knn_graph(X0,X0)
            knn0.run(nn)

            print("saving knn")
            np.savez(self.csk.project_dir+"tmap_knn.npz",adj = knn0.ind, dist = knn0.dist)

            adj = knn0.ind[:,1:]
            dist = knn0.dist[:,1:]
            dist0 = knn0.dist

        if not rough:
        #note rho shape here is (n,) not (n,1)
            print("smoothing")
            #local_connectivity = 2.0 fails somehow
            #rho is just knn0.dist[:,1]
            sigma,rho = smooth_knn_dist(dist0, nn,local_connectivity=1.0)

            #print("all",np.allclose(knn0.dist[:,1],rho))

            #this is for smooth case
            ndist = (dist - rho[:,None])/sigma[:,None]
        else:
            rho = dist[:,0].reshape((-1,1))
            sigma = dist[:,nn-2].reshape(-1,1)            
            #this is for rough case
            ndist = (dist - rho)/sigma

        w = np.exp(-ndist)

        idx0 = np.arange(adj.shape[0])

        #this works for the non ragged case
        wmat = sp.dok_matrix( (adj.shape[0],adj.shape[0]) )

        for i in range(adj.shape[1]):
            wmat[idx0,adj[:,i]] = w[:,i]

        wmat = wmat.tocsr()
        twmat = wmat.transpose()
        ww = wmat.multiply(twmat)

        #this ends up as csr matrix
        wsym = wmat + twmat - ww

        self.wsym = wsym

        #check number of components of graph
        n_components, labels = sp.csgraph.connected_components(wsym)
        print("components",n_components)

        wcoo = wsym.tocoo()
        self.wcoo = wcoo
    
    def init_pos0(self,XX0):
        #min_dist = .05
        min_dist = self.min_dist
        #spread = 1.0
        spread = self.spread
        a,b = find_ab_params(spread, min_dist)

        print("ab",a,b)
        self.a = a
        self.b = b
        self.X0 = np.zeros( (self.td.df_tot.shape[0],2),dtype=np.float32 )

        #postion init
        #not sure why this way is necessary
        idx0 = self.td.df_tot.loc[:,'parent'].values
        idx0 = idx0[:self.N0]

        np.random.seed(137)
        delX = np.random.multivariate_normal([0.0,0.0], .01*np.eye(2),size=self.N0)


        self.X0[:self.N0,:] = XX0[idx0 - self.N0,:]
        self.X0[:self.N0,:] += delX   
        self.X0[self.N0:,:] = XX0


    def run00(self,n_epochs=200,have_fixed = True):
        wcoo = self.wcoo

        """
        wcoo.data[wcoo.data < (wcoo.data.max() / float(n_epochs))] = 0.0
        wcoo.eliminate_zeros()
        """

        head = wcoo.row
        tail = wcoo.col

        epochs_per_sample = make_epochs_per_sample(wcoo.data, n_epochs)
        epoch_of_next_sample = np.array(epochs_per_sample)

        a,b = self.a,self.b

        thead = self.X0
        #self.not_fixed = np.ones(self.X0.shape[0],dtype=int)

        if have_fixed:
            self.not_fixed = np.zeros(self.X0.shape[0],dtype=int)
            self.not_fixed[:self.N0] = 1
        else:
            self.not_fixed = np.ones(self.X0.shape[0],dtype=int)
        n_vertices = thead.shape[0]


        t3 = time.time()    
        random_state = np.random.RandomState(137)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        optimize_layout(
            thead,
            thead,
            head,
            tail,
            self.not_fixed,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma=1.0,
            initial_alpha=1.0,
            negative_sample_rate=5.0)

        t4 = time.time()

        print("embedding time",t4-t3)

        N0 = self.N0
        path_pnts = thead[N0:]

        self.thead = thead
