import queue as Q

import pandas as pd
from pandas import DataFrame,Series
from sortedcontainers import SortedDict,SortedList

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp

from sklearn import linear_model

from itertools import product as cart

from collections import OrderedDict

import numba


class gspaths2:

    class pq_vert:
        def __init__(self,v_,cost_):
            self.v = v_
            self.cost = cost_

        def __lt__(self,other):
            return self.cost < other.cost

        def __eq__(self,other):
            return self.cost == other.cost        


    
    def __init__(self,adj_,dist_ = None):
        """
        adj_ should be sorted dict
        """
        
        self.adj0 = adj_

        if dist_ == None:
            dist_ = {}
            for v in adj_:
                row = adj_[v]
                nrow = len(row)
                dist_[v] = [1.0]*nrow
                            
        self.dist0 = dist_

    def get_parents0(self,v0set):
        adj = self.adj0
        dist = self.dist0
        
        pq = Q.PriorityQueue()    
        nvert = len(adj)

        vert = adj.keys()
        pvert = [-1]*nvert
        #parent = dict(zip(vert,pvert))
        parent = SortedDict(zip(vert,pvert))

        cost = [-1.0]*nvert
        #i think this should be sorted dict
        cost = SortedDict(zip(vert,cost))

        for v0 in v0set:
            parent[v0] = -1
            pq.put((self.pq_vert(v0,0.0)))
            cost[v0] = 0.0

        while not pq.empty():
            pqv = pq.get()
            v = pqv.v

            for i,u in enumerate(adj[pqv.v]):
                ucost = dist[v][i] + pqv.cost

                if cost[u] == -1.0 or ucost < cost[u]:
                    cost[u] = ucost
                    pq.put(self.pq_vert(u,ucost))
                    parent[u] = v

        #self.parent0 is sorted dict giving parent of
        #each vertex, keys are same as self.adj0 keys
        #cost is list in same order of distance from
        #v0set
        self.parent0 = parent
        self.cost = cost

    def get_path(self,vend):
        v1 = vend
        v0 = self.parent0[v1]

        path = [v1]

        while v0 != -1:
            path.append(v0)
            v1 = v0
            v0 = self.parent0[v1]

        return path[::-1]

    def get_path2(self,vstart,vend):

        #set up all paths from vstart
        self.get_parents0([vstart])

        #now get the path from vstart to vend
        path = self.get_path(vend)

        return path


    def get_paths(self,vstart,vends):

        #set up all paths from vstart
        self.get_parents0([vstart])

        paths = []

        for v1 in vends:
            path = self.get_path(v1)
            paths.append(path)

        return paths
    

def get_path_positions(tdata, path,eqdist):

    npath = len(path)

    #path is list of cells
    pdata = tdata[path]
    tvecs = pdata[1:] - pdata[:-1]

    x = la.norm(tvecs,axis=1,keepdims=True)
    #make unit tangent vectors
    tvecs /= x
    dots = np.sum(tvecs[:-1]*tvecs[1:],axis=1)

    positions = np.zeros(len(path))

    positions[1:] = np.cumsum(x)

    if eqdist:
        positions = .3 * np.arange(npath)

    return positions,dots




def get_segments_positions(tdata, segments,eqdist=False):
    #segments is a dictionary

    segments_positions = {}
    segments_dots = {}
    for key in segments:
        segments_positions[key],segments_dots[key] = get_path_positions(tdata,segments[key],eqdist)

    return segments_positions, segments_dots

"""
cg.seg_positions, seg_dots = get_segments_positions(tdata0, cg.segments,eqdist=True)
"""


class cgraph2:

    def mk_bpnts(br_adj):
        bpnts = SortedDict()

        #initialize some entries
        for i in range(5):
            bpnts[i] = []        

        #for br_adj we ignore deg 2 vertices
        for key in br_adj:
            row = br_adj[key]

            nedge = len(row)
            if nedge != 2:
                if bpnts.get(nedge):
                    bpnts[nedge].append(key)
                else:
                    bpnts[nedge] = [key]

        return bpnts



    def get_far_ends(adj,start):

        parent = {}
        degree = {}
        
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

        return far_ends,parent,degree
    


    
    def __init__(
            self,
            reduced_adj = None,
            segments = None,
            br_adj = None,
            start = None
            ):
        self.reduced_adj = reduced_adj
        self.segments = segments
        self.br_adj = br_adj
        self.start = start


    def from_br_adj(
            self,
            br_adj,
            start = None
            ):

        self.bpnts = cgraph2.mk_bpnts(br_adj)

        if start == None:
            start = self.bpnts[1][0]

        self.start = start
        self.br_adj = br_adj


        far_ends,parent,degree = cgraph2.get_far_ends(br_adj,start)

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

        self.reduced_adj = reduced_adj
        self.segments = segments
    
    def split(self,vsplit,plist=None):
        print("splitting at ",vsplit)

        v0 = None
        for vp in self.segments:
            if vp[0] > vp[1]: continue
            seg = self.segments[vp]
            if vsplit in seg:
                idx = seg.index(vsplit)
                print("split",vsplit,vp,idx)
                if vsplit in vp:
                    print("vsplit is seg end, no split")
                else:
                    v0 = vp[0]
                    v1 = vp[1]

            
        if v0 != None:
            self.split_segment(v0,v1,vsplit)
        else:
            print("cannot split, no segment found")



    def split_segment(self,v0,v1,vsplit):
        cg = self
        cg.start = vsplit

        #self.br_adj is unchanged

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
            
    def forward_graph(self,adj,start=None):
        if start == None:
            #start = bd.bpnts[1][0]
            start = self.start

        self.start = start
        for_adj = SortedDict()
        back_adj = SortedDict()        

        #walk the reduced_graph forward from start
        #adj = self.reduced_adj_sym

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
                lifo.put((v1,v0))

        far_ends = []

        for v in for_adj:
            if for_adj[v] == []:
                far_ends.append(v)


        self.for_adj = for_adj
        self.back_adj = back_adj
        self.far_ends = far_ends
                

        print("forward done")
                     


class cgraph:

    def get_segments_positions(self,tdata, segments,eqdist=False):
        #segments is a dictionary

        segments_positions = {}
        segments_dots = {}
        for key in segments:
            segments_positions[key],segments_dots[key] = get_path_positions(tdata,segments[key],eqdist)

        self.seg_positions = segments_positions

        return segments_dots
    



    def get_far_ends(self,adj,start,parent,degree):
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

        return far_ends

    


    def mk_bpnts(self,br_adj):
        bpnts = SortedDict()


        #initialize some entries
        for i in range(5):
            bpnts[i] = []        

        #find branch graph vertices which are
        # not 2-vertices, bpnts key is number of neighbors
        #this will be either 1 or >= 3
        for key in br_adj:
            row = br_adj[key]

            nedge = len(row)
            if nedge != 2:
                if bpnts.get(nedge):
                    bpnts[nedge].append(key)
                else:
                    bpnts[nedge] = [key]

        return bpnts


    def vpos_set_x(self,start,gap):
        #assign x coords of branching points
        #breadth first? walk thru forward graph

        radj = self.for_adj

        lifo = Q.LifoQueue() 

        lifo.put(start)
        vpos = SortedDict()
        vpos[start] = np.array([0.0,0.0])

        #does dy here make sense
        dy = 1.0
        fac = .75

        while not lifo.empty():
            v0 = lifo.get()
            pos0 = np.array(vpos[v0]) #dont need to np.array here?
            nvert = len(radj[v0])

            if nvert == 0:
                ylevels = []
            else:
                ylevels = np.linspace(-dy,dy,nvert)

            #shift vertex postions to get gap
            for i,v1 in enumerate(radj[v0]):
                dxl = self.seg_positions[(v0,v1)][-1]+gap
                vpos[v1] = pos0+(dxl,ylevels[i])
                lifo.put(v1)

                
            #shrink the range for vertices deeper in graph
            #result is that we get ordering of far_ends by y
            dy = fac*dy
        return vpos


    def vpos_set_x_0(self,start,gap):
        #assign x coords of branching points
        #breadth first? walk thru forward graph

        radj = self.for_adj

        lifo = Q.LifoQueue() 

        lifo.put(start)
        vpos = SortedDict()
        vpos[start] = np.array([0.0,0.0])

        #does dy here make sense
        dy = 1.0
        fac = .75

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
                dxl = self.seg_positions[(v0,v1)][-1]+gap
                vpos[v1] = pos0+(dxl,ylevels[i])
                lifo.put(v1)

                
            #shrink the range for vertices deeper in graph
            #result is that we get ordering of far_ends by y
            dy = fac*dy
        return vpos

    def vpos_set_y(self,vpos,ntrack,start):
        # cnames are the marker names whose tracks we want to show
        # only the number is used here
        radj = self.for_adj
        vert = []
        vert_level = []

        for v in self.far_ends:
            vert_level.append(vpos[v][1])
        vert_level = np.array(vert_level)



        #sort far_ends by level
        idx = np.argsort(vert_level)
        far_ends = np.array(self.far_ends,dtype=int)
        far_ends = far_ends[idx]

        vlevel = {}

        #first set all vertices level to -1

        for v in vpos:
            vlevel[v] = -1

            
        ydel = .1


        twidth = (4.0*ntrack/3 + 2)*ydel

        #vertical placement
        for i,v in enumerate(far_ends):

            vlevel[v] = twidth*i
            vpos[v][1] = twidth*i
            while self.back_adj[v] != [] and vlevel[self.back_adj[v][0]] == -1:
                v = self.back_adj[v][0]
                vlevel[v] = twidth*i
                vpos[v][1] = twidth*i


        return ydel


        
    
    def vpos_eps_shift(self,vpos,start):
        lifo = Q.LifoQueue()         
        lifo.put(start)

        radj = self.for_adj

        vals = np.array(vpos.values())
        vmax = np.amax(vals,axis=0)

        #eps = .01 * vmax[0]
        eps = .012 * vmax[0]

        while not lifo.empty():
            v0 = lifo.get()
            nvert = len(radj[v0])
            pos0 = vpos[v0]

            #shift vertex postions to get gap
            for i,v1 in enumerate(radj[v0]):
                dxl = self.seg_positions[(v0,v1)][-1]+ 2*eps
                vpos[v1][0] =  pos0[0] + dxl
                lifo.put(v1)

        # construct vertical lines
        # lifo for depth first search
        lifo = Q.LifoQueue()
        lifo.put(start)
        vsegs = []

        while not lifo.empty():
            v = lifo.get()
            #x = vpos[v][0]-.15 # previously had slight offset for vertical line

            #x of vertical line from current vertex
            x = vpos[v][0]

            #y ends of vertical line from children of vertex, if any
            #if no children then no vertical line
            if len(radj[v]) > 1:

                vlo = radj[v][0]
                vhi = radj[v][-1]

                plo = [x,vpos[vlo][1]]
                phi = [x,vpos[vhi][1]]
                vsegs.append([plo,phi])

            for v1 in radj[v]:
                lifo.put(v1)

        #lift the bottom of segs to make room for text
        for seg in vsegs:
            plo = seg[0]
            seg_lift = .10 #was .08
            #seg_lift = .08
            plo[1] += seg_lift
                

        return eps,vsegs
    
    

    def get_reduced_graph(self,br_adj,mst_adj=None,start=None):
        adj = br_adj

        self.br_adj = br_adj

        if mst_adj != None:
            self.mst_adj = mst_adj

        #need an end to start from
        if -1 in adj.keys():
            start = adj[-1][0]
            adj.pop(-1)
        elif start != None:
            pass
        

        bpnts = self.mk_bpnts(adj)


        if start == None:
            #pick an arbitrary end for start
            print("using bpnts")
            start = bpnts[1][0]

        #print("bpnts",bpnts)


        
        #stack = [start]
        #we use a dict for the parent points
        #so every node in the tree has a parent back towards start
        parent = {}
        degree = {}


        far_ends = self.get_far_ends(adj,start,parent,degree)


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


        self.bpnts = bpnts                
        self.start = start
        self.reduced_adj = reduced_adj
        #self.segments_sym = segments
        self.segments = segments
        self.far_ends = far_ends
        self.fw_edges = list(fw_edges)

    def br_edges(self):
        edges = []

        for v0 in self.br_adj:
            v0adj = self.br_adj[v0]
            for v1 in v0adj:
                if v1 > v0:
                    edges.append([v0,v1])

        edges = np.array(edges,dtype=np.int)

        return edges

    def forward_graph(self,adj,start=None):
        if start == None:
            if hasattr(self,'start'):
                start  = self.start
            else:
                start = self.bpnts[1][0]

        print("forward graph",start)

        self.start = start

        for_adj = SortedDict()

        #walk the sym graph forward from start
        #adj = self.reduced_adj_sym

        lifo = Q.LifoQueue()
        lifo.put( (start,-1))

        back_adj = SortedDict()


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
                lifo.put((v1,v0))

                #do stuff for v0 to v1

        far_ends = []

        for v in for_adj:
            if for_adj[v] == []:
                far_ends.append(v)


        self.for_adj = for_adj
        self.back_adj = back_adj
        self.far_ends = far_ends


        for_segments = {}
        seg_pos = {}        

        #collect the forward segments
        for v in for_adj:
            vadj = for_adj[v]
            for v1 in vadj:
                fseg = self.segments[(v,v1)]
                seg_pos[(v,v1)] = .3 * np.arange(len(fseg))
                for_segments[(v,v1)] = fseg

        self.for_segments = for_segments
        self.seg_positions = seg_pos
        

    def just_mst_side_cells(self,v0,v1):
        #gather mst vertices towards and beyond v1
        #do not include the start cell v0

        cells = []

        lifo = Q.LifoQueue()

        #put pairs on to keep track of already visited vertex
        #and avoid backtracking
        lifo.put((v0,v1))

        while not lifo.empty():
            vp = lifo.get()
            cells.append(vp[1])

            for v2 in self.mst_adj[vp[1]]:
                if v2 != vp[0]:
                    lifo.put((vp[1],v2))

        return cells
        


    #tentative
    def get_end_segment_cells(self,mst_adj,all=False):
        #if all is False leave a stub
        #ends = self.bpnts[1]
        self.mst_adj = mst_adj

        #get ends anew each time
        ends = []

        for v in self.reduced_adj:
            adjv = self.reduced_adj[v]
            if len(adjv) == 1:
                ends.append(v)
        

        self.end_segment_cells = {}

        for key in self.segments:
            node0 = key[0]
            node1 = key[1]
            if node1 not in ends:
                continue
            
            seg = self.segments[key]
            nseg = len(seg)

            if all:
                seg_cells = set(seg[1:])
            else:
                seg_cells = set(seg[2:])                

            for i in range(1,nseg-1):
                v0 = seg[i]
                vm = seg[i-1]
                vp = seg[i+1]
                
                for v1 in self.mst_adj[v0]:
                    if v1 != vm and v1 != vp:
                        cells = self.just_mst_side_cells(v0,v1)
                        seg_cells.update(cells)

            self.end_segment_cells[node1] = seg_cells



    def prune_end_segment(self,vend):
        rg = self

        #check if vend is an end

        vadj = self.reduced_adj[vend]

        if len(vadj) != 1:
            print("not and end",vend)
            return

        #find the segment

        seg = None
        for vp in rg.segments:
            if vend == vp[0]:
                seg = rg.segments[vp]
                break
            else:
                continue

        if seg != None:
            print("removing",vp)
        else:
            print("nothing to remove",vend)
            return

        vin = vp[1]
        vp2 = (vin,vend)

        self.reduced_adj[vin].remove(vend)
        self.reduced_adj[vend].remove(vin)
        self.reduced_adj.pop(vend)


        del self.segments[vp]
        del self.segments[vp2]        

        rg.cells_to_remove.extend(self.end_segment_cells[vend])

    def reduced_to_br_adj(self,rmap0):
        edges = set()
        vert = set()

        for vp in self.segments:
            seg = self.segments[vp]

            vert.update(seg)

            edges0 = list(zip(seg[:-1],seg[1:]))
            edges.update(edges0)

        br_adj = SortedDict()

        for v in vert:
            vv = rmap0[v]
            br_adj[vv] = []

        for e in edges:
            ee0 = rmap0[e[0]]
            ee1 = rmap0[e[1]]
            br_adj[ee0].append(ee1)

        return br_adj


    def split(self,vsplit,plist=None):
        cg = self

        print("cg0 id",id(cg))


        print("splitting at ",vsplit)
        
        #cg.get_reduced_graph(self.br_adj)
        self.cg = cg
        #cg.get_end_segment_cells(self.mst_adj,True)

        #find the segment that contains vsplit

        v0 = None

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


        #cg.get_end_segment_cells(self.mst_adj,True)            

        if plist:
            cg.cells_to_remove = []

            for vend in plist:
                cg.prune_end_segment(vend)
                #try redoing after each prune
                cg.get_end_segment_cells(self.mst_adj,True)                        

    def split_segment(self,v0,v1,vsplit):
        cg = self
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
