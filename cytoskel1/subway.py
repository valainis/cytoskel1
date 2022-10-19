import numpy as np
import pandas as pd
import numpy.linalg as la

import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.collections import LineCollection,PatchCollection
from matplotlib import cm
from matplotlib.figure import Figure

import matplotlib.patches as mpatches

from .util import *
from .cgraph import *


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

class subway:
    def __init__(self,csk,df_pcells,selected_branches):
        self.csk = csk
        self.df_pcells = df_pcells
        self.selected_branches = selected_branches
        

    def subway(self):
        cg = self.csk.cg
        self.cg = cg
        csk = self.csk

        cnames = list(self.df_pcells.columns)
        self.cnames = cnames

        tcols = csk.traj_markers
        tdata0 = csk.df_avg.loc[:,tcols].values

        seg_dots = cg.get_segments_positions(tdata0, cg.segments,eqdist=True)


        gap = 0.0
        vpos = cg.vpos_set_x(cg.start,gap)
        self.gap = gap


        cg.ntrack = len(cnames)
        cg.ydel = cg.vpos_set_y(vpos,cg.ntrack,cg.start)

        cg.eps0,cg.vsegs = cg.vpos_eps_shift(vpos,cg.start)

        pnts = vpos.values()

        pnts = np.array(pnts)

        self.cg = cg
        self.vpos = vpos
        #self.tdata0 = tdata0

        self.pcells = list(csk.br_adj.keys())

        
    def draw_subway(self,fig,axplot=None):
        csk = self.csk
        vpos = self.vpos
        cg = self.cg
        start = cg.start
        seg_pos = cg.seg_positions
        gap = self.gap

        cdata = self.df_pcells
        cnames = list(cdata.columns)
        seg_cvecs = {}
        
        for vp in cg.for_segments:        
            seg_cvecs[vp] = cdata.loc[cg.for_segments[vp],:].values

        xydots = []
        adots = [] #will be all the dots

        #the cells of the dots
        xycells = []

        scol = []
        seglines = {}        

        for vp in cg.for_segments:
            pos0 = vpos[vp[0]]
            spos = seg_pos[vp]
            scell = cg.for_segments[vp]

            
            scell_1 = scell[1:]
            spos_1 = spos[1:]
            

            x0 = pos0[0]
            x0 = x0 + cg.eps0

            #xycells.extend(scell)
            xycells.extend(scell_1)
            for i,s in enumerate(spos_1):
                xydots.append([x0+s,vpos[vp[1]][1]])
                scol.append(seg_cvecs[vp][i+1])

            #a line segment for vp[0] to vp[1]
            seglines[vp] = np.array([[pos0[0],vpos[vp[1]][1]], [pos0[0]+spos[-1],vpos[vp[1]][1]]])


        xydots.append([-cg.eps0,0])
        xycells.append(start)
        scol.append(cdata.loc[start,:].values)
        #scol.append(csk.df_avg.loc[start,cnames])

        xydots = np.array(xydots)
            

        scol = np.array(scol).T

        xydots = np.array(xydots)

        off = -(cg.ntrack-1)*cg.ydel/2.0
        self.off = off
        xydots += (0,off)
        xyt = xydots.T

        ntracks = cg.ntrack

        amin = np.amin(xyt,axis=1)
        amax = np.amax(xyt,axis=1)

        ax = {}
        pts = {}

        xleft = .1
        dyax = .05
        cbh = .02

        #yplot = .1+len(cnames)*dyax
        yplot = .1+cg.ntrack*dyax

        yplot = .05
        hplot = .95 - yplot

        #fig = plt.figure(figsize=(14,6))



        #ydel = cg.ydel
        ydel = 1.2*cg.ydel
        self.ydel = ydel
        dely = 0.0
        tspace = ntracks*ydel + 1.5*ydel*(ntracks//3) #for sizeing ax limits

        max_tot = np.amax(cdata,axis=0)
        #min_tot = np.zeros(max_tot.shape)
        min_tot = np.amin(cdata,axis=0)
        cmap = cm.get_cmap('jet')                

        """
        axplot = fig.add_axes([xleft,yplot,.8,hplot])
        self.axplot = axplot
        

        axplot.set_yticks([])
        axplot.set_xticks([])
        """

        if axplot:
            axplot.clear()
            ax_pos = axplot.get_position()
            print("ax_pos 0",ax_pos)
            axplot.set_position([ax_pos.x0,.05,.9,.95])
            ax_pos = axplot.get_position()
            print("ax_pos 1",ax_pos)            
        else:
            axplot = fig.add_axes([xleft,yplot,.8,hplot])
            self.axplot = axplot
            print("ax dim",xleft,yplot,.8,hplot)

        axplot.set_yticks([])
        axplot.set_xticks([])
        

        for i in range(ntracks):
            xydots2 = xydots + (0,dely)
            adots.append(xydots2)
            #scol2 = scol[i]
            scol2 = (scol[i] - min_tot[i])/(max_tot[i] - min_tot[i])            
            rgba = cmap(scol2) 
            #pts[i] = axplot.scatter(xyt[0],xyt[1]+dely,s=5,c=rgba)
            pts[i] = axplot.scatter(xyt[0],xyt[1]+dely,s=5,c=rgba)            
            j = i % 3
            if j == 2:
                dyyy = 1.5*ydel
            else:
                dyyy = ydel
            dely += dyyy


        vcol = mc.LineCollection(cg.vsegs,linestyle='dashed')
        axplot.add_collection(vcol)                   


        vtext = SortedDict()
        for v in vpos:
            #following line puts in vertex numbers
            #txt = axplot.text(vpos[v][0]-.7*gap,vpos[v][1]-.1,str(v),rotation=90,fontsize=4,verticalalignment='center')
            #vfont = 6
            vfont = 9
            txt = axplot.text(vpos[v][0]-.7*gap,vpos[v][1],str(v),
                              rotation=90,fontsize=vfont,verticalalignment='center',horizontalalignment='center')
            vtext[v] = np.array((vpos[v][0]-.7*gap,vpos[v][1]-.1))


        yyy = 0.0
        #xxx = -.1
        xxx = -.06
        #dyyy0 = .054
        dyyy0 = .072
        dyyy1 = .033
            
        #max on averaged trajectories
        max_scol = np.amax(scol,axis=1)

        tfontsize = 12 # was 8

        for i,name in enumerate(cnames):
            #s = "%4.1f %4.1f %4.1f" % (max_scol[i],min_tot[i],max_tot[i])
            s = "%4.1f %4.1f" % (min_tot[i],max_tot[i])
            axplot.text(xxx,yyy,name,transform=axplot.transAxes,fontsize=tfontsize)
            axplot.text(xxx,yyy-dyyy1,s,transform=axplot.transAxes,fontsize=tfontsize)

            j = i % 3
            if j == 2:
                dyyy = 2*dyyy0
            else:
                dyyy = dyyy0
            yyy = yyy + dyyy

        self.axplot = axplot
        self.selected()

        axplot.axis('off')

        print("tspace",tspace)
        axplot.set_ylim(amin[1]-.2,amax[1]+ tspace)

        self.vtext = vtext
        self.seglines = seglines

        self.adots = np.concatenate(adots)
        self.xycells = xycells

    def selected(self):
        vpos = self.vpos
        for vp in self.selected_branches:
            pos0 = vpos[vp[0]]
            pos1 = vpos[vp[1]]

            left0 = pos0[0]
            bottom0 = pos1[1] + self.off - self.ydel/2.0
            width = pos1[0] - pos0[0]
            height = -2*self.off + self.ydel
            alpha = .2
            print("adding branch",vp)
            p = mpatches.Rectangle((left0, bottom0), width, height,clip_on=False,color="green",alpha=alpha)
            self.axplot.add_patch(p)                            
        
        #plt.show()
