from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QSize
import PyQt5.QtCore as QtCore
from PyQt5.QtGui import QKeySequence,QFont, QColor
from PyQt5.QtWidgets import *

import PyQt5.QtWidgets as QtWidgets

import numpy as np
from numpy.linalg import norm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib import cm

import matplotlib.patches as mpatches

from cytoskel1.subway import subway
from cytoskel1.cgraph import *

import pickle
import os


#from bv import *

import csv
import queue as Q

def seg_dist(x, a0, a1):
    x = np.array(x)
    a0 = np.array(a0)
    a1 = np.array(a1)

    a = a1 - a0
    alen = norm(a)

    a = a/alen

    dx = x - a0

    s = a @ dx

    if s < 0:
        xd = norm(dx)
    elif s > alen:
        xd = norm(x - a1)
    else:
        dx = dx - s * a
        xd = norm(dx)

    return xd
    

class subway_canvas3(FigureCanvas):

    def __init__(self,parent, mwin, width=12, height=9, dpi=220):

        self.parent = parent
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)

        #self.axplot = self.fig.add_axes([xleft,yplot,.8,hplot])
        #self.axplot = self.fig.add_axes()

        eb = self.parent
        self.eb = eb
        print(eb.modes,eb.mode)

        self.axplot = self.fig.add_subplot()
        self.axplot.axis('off')
        self.mwin = mwin
        self.csk = mwin.csk
        csk = self.csk

        cg0 = csk.cg
        self.cg0 = cg0

        """

        self.pkl_file = "cgraph.pkl"

        cg0 = cg0 = csk.load(self.pkl_file)

        if not cg0:
            cg0 = cgraph()
            self.cg0 = cg0
            cg0.get_reduced_graph(self.csk.br_adj)

            csk.dump(cg0,self.pkl_file)

        else:
            self.cg0 = cg0
            print("read pickle")
        """

        #needed?
        self.start = cg0.start


        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.cid = self.mpl_connect('button_press_event', self)

        self.selected_branches = set()

        self.mwin.selected_branches = self.selected_branches

        #uncomment to see events
        #self.installEventFilter(self)

    def eventFilter(self,obj,event):
        print("event",event.type())
        return False

    def compute_initial_figure(self):

        print("initial 0",self.cg0.start)

        self.cg0.forward_graph(self.cg0.reduced_adj)

        print("names",self.mwin.subway_names)

        self.vtext,self.seglines = self.subway1(self.mwin.subway_names)

        print("initial 1",self.cg0.start)        

        #essential to keep this flush, for matplotlib event loop???
        self.flush_events()
        self.mwin.pca_branches = []
        self.draw()



    def __call__(self, event):

        if event.inaxes:
            cpnt = np.array([event.xdata,event.ydata])

            #print("event", event.button,"key",event.key,"end")

            if event.button == 3 and self.eb.mode == "Set Start":
                #print("set start",event.button,"xy",cpnt)

                pos = []
                verts = []
                for v in self.vtext:
                    pos.append(self.vtext[v])
                    verts.append(v)
                pos = np.array(pos)
                dpos = pos - cpnt
                dnorm = norm(dpos,axis=1)

                imin = np.argmin(dnorm)

                self.start = verts[imin]
                self.cg0.forward_graph(self.cg0.reduced_adj,start=verts[imin])
                self.vtext,self.seglines = self.subway1(self.mwin.subway_names,start=verts[imin])
                self.draw()

            elif event.button == 3 and self.eb.mode == "Select Segments":
                #print("branch select")
                vpa = []
                vpdist = [] 
                for vp in self.seglines:
                    seg = self.seglines[vp]
                    vpdist.append(seg_dist(cpnt,seg[0],seg[1]))
                    vpa.append(vp)
                vpdist = np.array(vpdist)
                imin = np.argmin(vpdist)

                #self.mwin.txt.setPlainText("branch"+str(vpa[imin]))

                vp2 = vpa[imin]
                vp2_rev = (vp2[1],vp2[0])

                if vp2 not in self.selected_branches and vp2_rev not in self.selected_branches:
                    self.selected_branches.add(vp2)
                elif vp2 in self.selected_branches:
                    self.selected_branches.remove(vp2)
                else:
                    self.selected_branches.remove(vp2_rev)

                print("selected",self.selected_branches)

                self.start = self.cg0.start
                self.cg0.forward_graph(self.cg0.reduced_adj,start=self.start)                    
                self.vtext,self.seglines = self.subway1(self.mwin.subway_names,self.start)
                self.draw()                    

                #self.mwin.pca_branches.append(vp2)
                s = str(vp2[0]) + " " + str(vp2[1])
                #self.mwin.branch_list.addItem(s)

            #elif event.button == 3 and event.key == "s": #for split
            elif event.button == 3 and self.eb.mode == "Split":                

                dx = cpnt - self.adots

                dist = la.norm(dx,axis=1)
                imin = np.argmin(dist)

                imin = imin % len(self.xycells)

                #print("imin",imin,self.xycells[imin])

                s = "Cell: %d" % self.xycells[imin]

                vpick = self.xycells[imin]

                #self.mwin.eb.clabel.setText(s)
                self.mwin.txt.setPlainText(s)

                self.cg0.split(vpick)

                #self.csk.dump(self.cg0,self.pkl_file+"_1")

                vin = vpick in self.cg0.reduced_adj.keys()

                self.start = vpick
                self.cg0.forward_graph(self.cg0.reduced_adj,start=self.start)                
                self.vtext,self.seglines = self.subway1(self.mwin.subway_names,self.start)
                self.draw()
                


            #elif event.button == 3 and event.key == "i": #for info
            elif event.button == 3 and self.eb.mode == "Cell Info":
                dx = cpnt - self.adots

                dist = la.norm(dx,axis=1)
                imin = np.argmin(dist)

                imin = imin % len(self.xycells)

                #print("imin",imin,self.xycells[imin])



                vpick = self.xycells[imin]

                segments = self.cg0.for_segments

                #bd = self.mwin.bd
                #segments = bd.segments
                for vp in segments:
                    seg = segments[vp]
                    if vpick in seg:
                        print("vpick seg",vpick,vp)
                        v0 = vp[0]
                        v1 = vp[1]
                        iseg = seg.index(vpick)

                #s = "Cell: %d" % self.xycells[imin]

                self.eb.mwin.show_cell_info(self.xycells[imin])

                #self.eb.mwin.txt.setPlainText(s)

                #self.mwin.eb.clabel.setText(s)
                
                
                    
                                
        
    def subway1(self,cnames,start=None,clear=False):

        csk = self.csk

        pcells = list(csk.br_adj.keys())        
        #df_pcells = csk.df_avg2.loc[pcells,cnames]
        df_pcells = csk.df_pcells.loc[:,cnames]

        sub = subway(csk,df_pcells,self.selected_branches)
        sub.subway()

        #fig = plt.figure(figsize=(14,6))
        sub.draw_subway(self.fig,self.axplot)


        self.adots = sub.adots
        self.xycells = sub.xycells        

        print("subway1")
        print(self.selected_branches)
        print("subway1")
        print(type(self.adots),type(self.xycells))
        
        return sub.vtext,sub.seglines



        cg0 = self.cg0
        cg = cg0
        self.cnames = cnames

        do_dump = False

        if not start:
            print("subway 0",cg0.start)
            start = cg0.start

        #cg0.forward_graph(cg0.reduced_adj,start=start)

        self.start = cg0.start
        for_adj = cg0.for_adj

        gap = 0.0
        self.gap = gap
        
        vpos = cg0.vpos_set_x(cg0.start,gap)

        cg.ntrack = len(cnames)
        ntrack = cg.ntrack
        ydel = cg0.vpos_set_y(vpos,cg.ntrack,cg0.start)
        eps0,vsegs = cg0.vpos_eps_shift(vpos,cg0.start)        

        #start of plotting
        xypos = []
        vcol = mc.LineCollection(vsegs,linestyle='dashed')

        cdata = csk.df_avg[cnames]
        seg_cvecs = {}    

        for vp in cg0.for_segments:        
            seg_cvecs[vp] = cdata.loc[cg0.for_segments[vp],:].values
            
        xydots = []
        adots = [] #will be all the dots

        #the cells of the dots
        xycells = []

        scol = []
        seglines = {}
        seg_pos = cg0.seg_positions

        for vp in cg0.for_segments:
            pos0 = vpos[vp[0]]
            spos = seg_pos[vp]
            scell = cg0.for_segments[vp]

            
            scell_1 = scell[1:]
            spos_1 = spos[1:]
            

            x0 = pos0[0]
            x0 = x0 + eps0

            xycells.extend(scell_1)
            for i,s in enumerate(spos_1):
                xydots.append([x0+s,vpos[vp[1]][1]])
                scol.append(seg_cvecs[vp][i+1])

            #a line segment for vp[0] to vp[1]
            seglines[vp] = np.array([[pos0[0],vpos[vp[1]][1]], [pos0[0]+spos[-1],vpos[vp[1]][1]]])


        xydots.append([-eps0,0])
        xycells.append(start)
        scol.append(csk.df_avg.loc[start,cnames])

        xydots = np.array(xydots)

        off = -(cg.ntrack-1)*ydel/2.0
        xydots += (0,off)
        xyt = xydots.T

        amin = np.amin(xyt,axis=1)
        amax = np.amax(xyt,axis=1)

        scol = np.array(scol).T

        xleft = .1
        dyax = .05
        cbh = .02

        yplot = .1+len(cnames)*dyax

        yplot = .05
        hplot = .95 - yplot


        axplot = getattr(self,"axplot",None)
        if axplot:
            axplot.clear()
        else:
            axplot = self.fig.add_axes([xleft,yplot,.8,hplot])
            self.axplot = axplot

        axplot.set_yticks([])
        axplot.set_xticks([])

        #plot the trajectories

        dely = 0.0
        ntracks = scol.shape[0]
        tspace = ntracks*ydel + 1.5*ydel*(ntracks//3)

        cmap = cm.get_cmap('jet')

        br_cells = list(csk.br_adj.keys())

        tvecs = csk.df_avg.loc[br_cells,cnames].values

        max_tot = np.amax(tvecs,axis=0)
        min_tot = np.zeros(max_tot.shape)


        print("scol",scol.shape[0])

        
        for i in range(scol.shape[0]):

            scol2 = (scol[i] - min_tot[i])/(max_tot[i] - min_tot[i])
            rgba = cmap(scol2)
            xydots2 = xydots + (0,dely)
            adots.append(xydots2)       

            axplot.scatter(xyt[0],xyt[1]+dely,s=5,c=rgba)
            j = i % 3
            if j == 2:
                dyyy = 1.5*ydel
            else:
                dyyy = ydel
            dely += dyyy

        
        self.nrow = len(adots)
        adots = np.concatenate(adots)

        self.adots = adots
        self.xycells = xycells

        vtext = SortedDict()
        for v in vpos:
            txt = axplot.text(vpos[v][0]-.7*gap,vpos[v][1],str(v),rotation=90,fontsize=10,verticalalignment='center',horizontalalignment='center')
            vtext[v] = np.array((vpos[v][0]-.7*gap,vpos[v][1]-.1))


        yyy = 0.0
        xxx = -.1
        dyyy0 = .054
        dyyy1 = .02


        #max on averaged trajectories
        max_scol = np.amax(scol,axis=1)

        tfontsize = 12

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


        axplot.axis('off')

        axplot.set_ylim(amin[1]-.2,amax[1]+ tspace)

        #axplot.add_collection(col)
        axplot.add_collection(vcol)

        for vp in self.selected_branches:
            pos0 = vpos[vp[0]]
            pos1 = vpos[vp[1]]

            left0 = pos0[0]
            bottom0 = pos1[1] + off - ydel/2.0
            width = pos1[0] - pos0[0]
            height = -2*off + ydel
            alpha = .2
            print("adding branch",vp)
            p = mpatches.Rectangle((left0, bottom0), width, height,clip_on=False,color="green",alpha=alpha)
            axplot.add_patch(p)                            

        return vtext,seglines

